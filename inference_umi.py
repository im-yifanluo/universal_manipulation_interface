import time
import cv2
import numpy as np
import torch
import dill
import hydra
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import pyrealsense2 as rs
from copy import deepcopy
import copy
import vedo
import os
from diffusion_policy.common.pose_repr_util import compute_relative_pose
from diffusion_policy.common.pose_util import rot6d_to_mat, mat_to_rot6d
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.model.common.rotation_transformer import RotationTransformer, RotationTransformerUMI

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import Image
from rocky_scripts_msgs.msg import StampedFloat32, StampedFloat32MultiArray
from cv_bridge import CvBridge
import message_filters


H = 240
W = 320
FPS = 60
MAX_GRIPPER_WIDTH = 82.
CONTROL_FREQUENCY = 60.
DATA_FREQUENCY = 10.
SLOP = 0.02


def interpolate_between_actions (start_action, end_action, interpolation_factor):
    '''
    This function interpolate between a start action and an end action to generate a trajectory
    with length interpolation_factor, including the end action but not the start action.

    Action has format [x, y, z, rx, ry, rz, gripper_width].
    '''

    # decode
    start_pos = start_action[0:3]
    start_rotvec = start_action[3:6]
    start_width = start_action[6]
    end_pos = end_action[0:3]
    end_rotvec = end_action[3:6]
    end_width = end_action[6]

    # rotation interpolation
    rotations = R.from_rotvec([start_rotvec, end_rotvec])
    key_times = [0, 1]
    slerp = Slerp(key_times, rotations)

    action_interpolated = []

    for i in range (1, int(interpolation_factor)):
        next_pos = (end_pos - start_pos) * i / interpolation_factor + start_pos
        next_rotvec = slerp(float(i) / interpolation_factor).as_rotvec()
        next_width = (end_width - start_width) * i / interpolation_factor + start_width

        next_action = next_pos.tolist() + next_rotvec.tolist() + [next_width]
        action_interpolated.append(next_action.copy())

    action_interpolated.append(end_action.copy())

    return action_interpolated


class DiffusionPolicyInference (Node):
    
    def __init__ (self, ckpt_path, results_path):

        super().__init__('diffusion_policy_inference')


        if not os.path.exists(results_path):
            os.makedirs(results_path)

        # to prevent overwriting
        episode_id = len(os.listdir(results_path))
        self.episode_path = '{}/{}'.format(results_path, episode_id)


        # ===== realsense initialization =====

        # start realsense cameras
        # umi setup only has wrist camera
        self.pipeline_1 = rs.pipeline()
        config_1 = rs.config()
        # config_1.enable_device('348522073794')  # wrist camera
        config_1.enable_stream(rs.stream.color, W*2, H*2, rs.format.bgr8, FPS)
        self.pipeline_1.start(config_1)


        # ===== diffusion policy init =====

        # load checkpoint
        payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)

        # if need to overwrite
        # cfg['policy']['num_inference_steps'] = 10
        # self.n_action_steps = cfg['n_action_steps']
        self.n_action_steps = 8
        self.n_latency_steps = 1  # 0.1 seconds, should be more than enough
        cfg['policy']['n_action_steps'] = self.n_action_steps + self.n_latency_steps  # overwrite

        # for interpolation and latency handling
        self.counter = 0
        self.traj_idx = 0
        self.finished_cur_action_chunk = False
        self.finished_cur_traj = False
        self.next_traj_ready = False
        self.last_action = None  # this is the last action in the previous action chunk
        self.interpolation_factor = CONTROL_FREQUENCY / DATA_FREQUENCY
        self.cur_traj = []
        self.next_traj = []

        # for n_obs_steps (history)
        self.history_timestamps = []
        self.history_obs_dicts = []
        self.history_len = 10  # this is not the same as the number of history steps (2) used for the policy!

        # for trajectory logging
        self.trajectory_received = []
        self.trajectory_executed = []
        self.cur_robot_pose = None
        self.cur_gripper_width = None
        self.timestep_limit = 1200
        self.eef_positions = []
        self.predicted_action_chunks = []
        self.wrist_images = []

        workspace = cls(cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        self.policy = workspace.model
        if cfg.training.use_ema:
            self.policy = workspace.ema_model
        
        self.policy.to(torch.device('cuda:0'))
        self.policy.eval()
        self.device = self.policy.device

        self.rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        self.rot_quat2mat = RotationTransformerUMI(
            from_rep='quaternion',
            to_rep='matrix')
        self.rot_aa2mat = RotationTransformerUMI(
            from_rep='axis_angle',
            to_rep='matrix')
        self.rot_mat2rot6d = RotationTransformerUMI(
            from_rep='matrix',
            to_rep='rotation_6d')


        # ===== ros init =====

        self.cvbridge = CvBridge()

        pose_sub = message_filters.Subscriber(self, StampedFloat32MultiArray, 'robot_pose')
        gripper_width_sub = message_filters.Subscriber(self, StampedFloat32, 'gripper_width')

        ats = message_filters.ApproximateTimeSynchronizer([
            pose_sub,
            gripper_width_sub,
        ], queue_size=1, slop=SLOP)
        ats.registerCallback(self.callback_sub)

        self.robot_trajectory_pub = self.create_publisher(Float32MultiArray, 'robot_target_pose', 1)
        self.gripper_trajectory_pub = self.create_publisher(Float32MultiArray, 'gripper_target_width', 1)
        rate = CONTROL_FREQUENCY
        self.timer = self.create_timer(1./rate, self.callback_pub)

        self.last_inference_timestamp = float(self.get_clock().now().nanoseconds) / 1e9


    def callback_sub (self, pose_msg, gripper_width_msg):

        # always update robot info for logging
        self.cur_robot_pose = np.array(pose_msg.data)
        self.cur_gripper_width = np.array(gripper_width_msg.data)

        # always prepare obs_dict for history
        cur_timestamp = float(self.get_clock().now().nanoseconds) / 1e9

        # get realsense images
        # realsense 1, wrist camera
        frames_1 = self.pipeline_1.wait_for_frames()
        color_frame_1 = frames_1.get_color_frame()
        wrist_image = np.asanyarray(color_frame_1.get_data()).copy()

        # always saves image
        self.wrist_images.append(wrist_image.copy())
        wrist_image = cv2.resize(wrist_image, (W, H))  # images are at higher resolution for visualization

        cv2.imshow('wrist', wrist_image)
        cv2.waitKey(1)

        # convert messages
        robot_pose = np.array(pose_msg.data)
        gripper_width = np.array(gripper_width_msg.data)

        # robot_pose is [x, y, z, rx, ry, rz]
        eef_pos = robot_pose[0:3].copy()
        eef_quat = R.from_rotvec(robot_pose[3:6]).as_quat()

        # gripper_width is not normalized
        gripper_qpos = np.array([gripper_width / MAX_GRIPPER_WIDTH, gripper_width / MAX_GRIPPER_WIDTH])

        # create obs dict
        np_obs_dict = dict()

        # fill in all input required by dp
        np_obs_dict['wrist_camera'] = wrist_image.copy()
        np_obs_dict['eef_pos'] = eef_pos
        np_obs_dict['eef_quat'] = eef_quat
        np_obs_dict['gripper_qpos'] = gripper_qpos

        # swap axis for images and normalize
        np_obs_dict['wrist_camera'] = np.swapaxes(np.swapaxes(np_obs_dict['wrist_camera'], 1, 2), 0, 1).astype(np.float32)/255.

        # only run inference if 
        # 1. the last action chunk (excluding latency steps) has been executed OR
        # 2. inference has not been run
        if self.finished_cur_action_chunk or len(self.cur_traj) == 0:

            # reset
            self.finished_cur_action_chunk = False

            # # get frequency for debug
            # # note: this frequency is how often inference is run, which only happens once per action chunk execution
            # # it is not the same as how fast the model can predict
            # cur_timestamp = float(self.get_clock().now().nanoseconds) / 1e9
            # cur_fps = 1./(cur_timestamp - self.last_inference_timestamp)
            # self.last_inference_timestamp = float(self.get_clock().now().nanoseconds) / 1e9
            # self.get_logger().info("Running inference at {} fps".format(cur_fps))

            # to account for the two steps of observations
            # shapes in np_obs_dict:
            # - wrist_camera: (3, 240, 320)
            # - front_camera: (3, 240, 320)
            # - eef_pos: (3,)
            # - eef_quat: (4,)
            # - gripper_qpos: (2,)
            # shapes to match:
            # - wrist_camera: torch.Size([1, 2, 3, 240, 320])
            # - front_camera: torch.Size([1, 2, 3, 240, 320])
            # - eef_pos: torch.Size([1, 2, 3])
            # - eef_quat: torch.Size([1, 2, 4])
            # - gripper_qpos: torch.Size([1, 2, 2])

            np_obs_dict_stacked = {}

            # if no history available, stack current obs
            if len(self.history_obs_dicts) == 0:
                for key in np_obs_dict:
                    np_obs_dict_stacked[key] = np.stack((np_obs_dict[key], np_obs_dict[key]), axis=0)
            
            # otherwise, get the frame that is the closest to cur_timestamp - 1./DATA_FREQUENCY (0.1s before current frame)
            else:
                frame_idx = np.argmin(np.abs((cur_timestamp - 1./DATA_FREQUENCY) - np.array(self.history_timestamps)))
                # self.get_logger().info('history timestamps: {}'.format(self.history_timestamps))
                # self.get_logger().info('cur_timestamp: {}'.format(cur_timestamp))
                # self.get_logger().info('diffs: {}'.format(cur_timestamp - np.array(self.history_timestamps)))
                # self.get_logger().info('selected index {}'.format(frame_idx))
                np_obs_dict_old = self.history_obs_dicts[frame_idx]
                # first the previous frame, then the current frame
                for key in np_obs_dict:
                    np_obs_dict_stacked[key] = np.stack((np_obs_dict_old[key], np_obs_dict[key]), axis=0)

            # using relative action
            current_pos = copy.copy(np_obs_dict_stacked['eef_pos'][-1])
            current_rot_mat = copy.copy(self.rot_quat2mat.forward(np_obs_dict_stacked['eef_quat'][-1]))
            T_world_baseframe = np.eye(4)
            T_world_baseframe[:3, :3] = current_rot_mat.copy()
            T_world_baseframe[:3, 3] = current_pos.copy()

            # for key in np_obs_dict_stacked:
            #     print(key, np_obs_dict_stacked[key].shape)

            np_obs_dict_stacked['eef_pos'], np_obs_dict_stacked['eef_quat'] = compute_relative_pose(
                pos=np_obs_dict_stacked['eef_pos'],
                rot=np_obs_dict_stacked['eef_quat'],
                base_pos=current_pos,
                base_rot_mat=current_rot_mat,
                rot_transformer_to_mat=self.rot_quat2mat,
                rot_transformer_to_target=self.rot_mat2rot6d
            )

            np_obs_dict_stacked = {
                key: value[np.newaxis, :] for key, value in np_obs_dict_stacked.items()
            }

            # device transfer
            obs_dict = dict_apply(np_obs_dict_stacked, 
                lambda x: torch.from_numpy(x).to(
                    device=self.device))

            # run policy
            with torch.no_grad():
                action_dict = self.policy.predict_action(obs_dict)

            # device_transfer
            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').numpy())

            action_chunk = np_action_dict['action'][0]
            if not np.all(np.isfinite(action_chunk)):
                raise RuntimeError('Nan or Inf action')
            
            # action rotation transformer
            action_pos, action_rot = compute_relative_pose(
                pos=action_chunk[..., :3],
                rot=action_chunk[..., 3: -1],
                base_pos=current_pos,
                base_rot_mat=current_rot_mat,
                rot_transformer_to_mat=self.rot_aa2mat,
                rot_transformer_to_target=self.rot_mat2rot6d,
                backward=True
            )
            action_gripper = action_chunk[..., -1:]
            action_chunk = np.concatenate([action_pos, action_rot, action_gripper], axis=-1)

            # # debug
            # print(np.round(robot_pose, 4))
            # print(np.round(action_chunk, 4))
            # rclpy.shutdown()
            # eef_pos_vis = vedo.Point(robot_pose[0:3], r=10, c='red')
            # traj_vis = vedo.Points(action_chunk[:, 0:3], r=5, c='green')
            # plotter = vedo.Plotter(axes=1)
            # plotter.show(eef_pos_vis, traj_vis)

            # logging
            self.eef_positions.append(eef_pos)
            self.predicted_action_chunks.append(action_chunk)

            # convert gripper width back
            # print('Predicted raw gripper width: {}'.format(action_chunk[:, -1]))
            action_chunk[:, -1] *= MAX_GRIPPER_WIDTH
            # print('Converted gripper width: {}'.format(action_chunk[:, -1]))

            # interpolate trajectory
            self.next_traj = []

            # if this is the first action chunk, there is no latency
            if len(self.cur_traj) == 0:
                # interpolation starts from the start of the first action chunk
                self.next_traj.append(action_chunk[0].tolist())

                for i in range (len(action_chunk)-1):
                    action_interpolated = interpolate_between_actions(action_chunk[i], action_chunk[i+1], self.interpolation_factor)
                    self.next_traj += action_interpolated

            # otherwise, interpolate between the last action and cur_action_chunk[n_latency_steps]
            else:
                # self.last_action is not included as part of the current traj because it was already included in the last traj
                # using action_chunk[self.n_latency_steps + 1] instead of self.n_latency_steps because robot has 0.1s lookahead time
                action_interpolated = interpolate_between_actions(self.last_action, action_chunk[self.n_latency_steps + 1], self.interpolation_factor)
                self.next_traj += action_interpolated

                # # to make actions smoother, interpolate from current robot pose
                # interp_start = np.array(self.cur_robot_pose.tolist() + [self.cur_gripper_width])
                # action_interpolated = interpolate_between_actions(interp_start, action_chunk[self.n_latency_steps], self.interpolation_factor)
                # self.next_traj += action_interpolated

                # interpolate the rest of cur_action_chunk
                for i in range (self.n_latency_steps + 1, len(action_chunk)-1):
                    action_interpolated = interpolate_between_actions(action_chunk[i], action_chunk[i+1], self.interpolation_factor)
                    self.next_traj += action_interpolated

                # self.get_logger().info('eef_pos: {}; 1st point in action_chunk: {}; 1st point in next_traj: {}; last_action: {}'.format(
                #     np.round(eef_pos, 4),
                #     np.round(action_chunk[0, 0:3], 4),
                #     np.round(self.next_traj[0][0:3], 4),
                #     np.round(self.last_action[0:3], 4)
                # ))

            # self.get_logger().info('next_traj length: {}'.format(len(self.next_traj)))
            # self.get_logger().info('eef pos: {}'.format(eef_pos))
            # self.get_logger().info('action chunk: {}'.format(action_chunk[:, 0:3]))

            # updates
            self.next_traj_ready = True
            self.last_action = action_chunk[-1].copy()  # this would be the last latency step

        # update history
        if len(self.history_obs_dicts) >= self.history_len:
            self.history_timestamps.pop(0)
            self.history_obs_dicts.pop(0)
        self.history_timestamps.append(cur_timestamp)
        self.history_obs_dicts.append(deepcopy(np_obs_dict))

    
    def callback_pub (self):

        # determine the trajectory to execute
        # if finished the current trajectory and the next trajectory is ready, update cur_traj to be next_traj
        # this should always be the case, since n_latency_steps * 1./DATA_FREQUENCY > inference time 
        if (self.finished_cur_traj or len(self.cur_traj) == 0) and self.next_traj_ready:
        # if self.next_traj_ready:
            self.cur_traj = self.next_traj.copy()
            # reset
            self.next_traj = []
            self.finished_cur_traj = False
            self.next_traj_ready = False
            self.traj_idx = 0

        # conditions for the robot to not do anything:
        # 1. cur_traj is empty (inference has not started) OR
        # 2. finished cur traj but next traj is not ready
        if len(self.cur_traj) == 0:
            self.get_logger().warning('Inference has not started.', once=True)

        # if finished executing current traj but the next traj is not ready yet, give warning
        elif self.finished_cur_traj and not self.next_traj_ready:
            self.get_logger().warning('Finished current trajectory, but the next trajectory is not ready yet!')

        # otherwise, execute trajectory as usual
        else:  # would not enter this statement if self.finished_cur_traj == True
            cur_action = self.cur_traj[self.traj_idx]
            robot_target_pose = cur_action[0:6]
            gripper_target_width = cur_action[6]

            robot_target_pose_msg = Float32MultiArray()
            robot_target_pose_msg.data = list(map(float, robot_target_pose)) + [float(self.counter)]  # add counter as part of the message to check delay
            self.robot_trajectory_pub.publish(robot_target_pose_msg)

            gripper_target_width_msg = Float32MultiArray()
            gripper_target_width_msg.data = [float(gripper_target_width), float(self.counter)]
            self.gripper_trajectory_pub.publish(gripper_target_width_msg)

            cur_timestamp = float(self.get_clock().now().nanoseconds) / 1e9
            self.get_logger().info("Published trajectory point {} at time {}".format(self.counter, cur_timestamp))
            
            # log trajectory
            # append received trajectory
            self.trajectory_received.append(cur_action)
            
            # append executed trajectory
            self.trajectory_executed.append(self.cur_robot_pose.tolist() + [self.cur_gripper_width])

            if len(self.trajectory_received) >= self.timestep_limit:
                # create episode folder
                if not os.path.exists(self.episode_path):
                    os.makedirs(self.episode_path)

                np.savez(
                    '{}/trajectory.npz'.format(self.episode_path), 
                    trajectory_received = self.trajectory_received,
                    trajectory_executed = self.trajectory_executed,
                )
                np.savez(
                    '{}/predictions.npz'.format(self.episode_path),
                    eef_positions = self.eef_positions,
                    predicted_action_chunks = self.predicted_action_chunks,
                )

                # write video
                wrist_cam_writer = cv2.VideoWriter(
                    '{}/wrist.mp4'.format(self.episode_path), 
                    cv2.VideoWriter_fourcc(*'mp4v') , 
                    FPS, 
                    (W*2, H*2),
                )
                self.get_logger().info('Saving images to mp4...')
                for wrist in self.wrist_images:
                    wrist_cam_writer.write(wrist)
                wrist_cam_writer.release()

                self.get_logger().info('Saved episode data.')
                rclpy.shutdown()
            
            # updates
            self.traj_idx += 1
            self.counter += 1

            # check status
            # if this is the first trajectory, the length of the trajectory should be:
            #   (n_action_steps + n_latency_steps - 1) * interpolation_factor + 1;
            # and the number of steps excluding the latency steps should be:
            #   (n_action_steps - 1) * interpolation_factor + 1.
            # if this is not the first trajectory, the length of the trajectory should be:
            #   (n_action_steps - 1) * interpolation_factor (the -1 is because of 0.1s robot lookahead time)
            # and the number of steps exluding the latency steps should be:
            #   (n_action_steps - 1 - n_latency_steps) * interpolation_factor

            if len(self.cur_traj) == (self.n_action_steps + self.n_latency_steps - 1) * self.interpolation_factor + 1:
                if self.traj_idx == (self.n_action_steps - 1) * self.interpolation_factor + 1:  # use == and not >= to trigger this only once
                    self.finished_cur_action_chunk = True  # this is the signal that inference should run again
                    self.get_logger().info('Signaling next inference at trajectory index {}.'.format(self.traj_idx))
            else:
                if self.traj_idx == (self.n_action_steps - 1 - self.n_latency_steps) * self.interpolation_factor:
                    self.finished_cur_action_chunk = True
                    self.get_logger().info('Signaling next inference at trajectory index {}.'.format(self.traj_idx))

            if self.traj_idx >= len(self.cur_traj):
                self.finished_cur_traj = True


def main (args=None):

    ckpt_path = '/placeholder.ckpt'
    results_path = '/placeholder'

    rclpy.init(args=args)
    dp_inference_node = DiffusionPolicyInference(ckpt_path, results_path)
    rclpy.spin(dp_inference_node)

    dp_inference_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()