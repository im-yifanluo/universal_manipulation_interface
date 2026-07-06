"""
CUSTOM: This is the custom ROS 2 inference script
"""

import time
import argparse
import json
import cv2
import numpy as np
import torch
import dill
import hydra
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import pyrealsense2 as rs
from copy import deepcopy
import os
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.real_world.real_inference_util import get_real_umi_obs_dict, get_real_umi_action

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from rocky_scripts_msgs.msg import StampedFloat32, StampedFloat32MultiArray
import message_filters


H = 240
W = 320
FPS = 60
MAX_GRIPPER_WIDTH = 110.
DATASET_MAX_GRIPPER_WIDTH_M = 0.09
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


def wsg_width_mm_to_policy_m(width_mm):
    # Training zarr was built from normalized HDF5 qpos:
    #   robot0_gripper_width = gripper_qpos * DATASET_MAX_GRIPPER_WIDTH_M
    # where gripper_qpos = wsg_width_mm / MAX_GRIPPER_WIDTH.
    width_normalized = width_mm / MAX_GRIPPER_WIDTH
    width_m = width_normalized * DATASET_MAX_GRIPPER_WIDTH_M
    return np.float32(np.clip(width_m, 0.0, DATASET_MAX_GRIPPER_WIDTH_M))


def policy_width_m_to_wsg_mm(width_m):
    width_normalized = width_m / DATASET_MAX_GRIPPER_WIDTH_M
    width_mm = width_normalized * MAX_GRIPPER_WIDTH
    return float(np.clip(width_mm, 0.0, MAX_GRIPPER_WIDTH))


class DiffusionPolicyInference (Node):
    
    def __init__ (self, ckpt_path, results_path, show_wrist=False):

        super().__init__('diffusion_policy_inference')

        self.show_wrist = show_wrist

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        # to prevent overwriting
        episode_id = 0
        while os.path.exists('{}/{}'.format(results_path, episode_id)):
            episode_id += 1
        self.episode_path = '{}/{}'.format(results_path, episode_id)
        os.makedirs(self.episode_path)
        self.prediction_debug_path = '{}/prediction_debug.jsonl'.format(self.episode_path)
        open(self.prediction_debug_path, 'w').close()


        # ===== realsense initialization =====

        # start realsense cameras
        # umi setup only has wrist camera
        self.pipeline_1 = rs.pipeline()
        config_1 = rs.config()
        # config_1.enable_device('348522073794')  # wrist camera
        config_1.enable_stream(rs.stream.color, W*2, H*2, rs.format.rgb8, FPS)
        self.pipeline_1.start(config_1)


        # ===== diffusion policy init =====

        # load checkpoint
        payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill, weights_only=False)
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)

        # if need to overwrite
        # cfg['policy']['num_inference_steps'] = 10
        self.n_action_steps = 8
        self.n_latency_steps = 1
        self.model_action_steps = self.n_action_steps + self.n_latency_steps
        self.action_down_sample_steps = int(
            cfg.task.shape_meta.action.get('down_sample_steps', 1)
        )
        self.policy_action_frequency = DATA_FREQUENCY
        # DiffusionUnetTimmPolicy returns the full action horizon and has no
        # policy.n_action_steps config key; crop to model_action_steps after inference.

        self.cfg = cfg
        self.obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
        self.action_pose_repr = cfg.task.pose_repr.action_pose_repr
        self.episode_start_pose = None

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

        self.get_logger().info(
            'Using policy action frequency {:.3f} Hz from DATA_FREQUENCY {} Hz. '
            'Checkpoint action.down_sample_steps is {}. Interpolation factor: {}.'.format(
                self.policy_action_frequency,
                DATA_FREQUENCY,
                self.action_down_sample_steps,
                self.interpolation_factor,
            )
        )

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


        # ===== ros init =====

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
        self.cur_robot_pose = np.array(pose_msg.data, dtype=np.float32)
        self.cur_gripper_width = float(np.array(gripper_width_msg.data).reshape(-1)[0])

        # always prepare obs_dict for history
        cur_timestamp = float(self.get_clock().now().nanoseconds) / 1e9

        # get realsense images
        # realsense 1, wrist camera
        frames_1 = self.pipeline_1.wait_for_frames()
        color_frame_1 = frames_1.get_color_frame()
        wrist_image = np.asanyarray(color_frame_1.get_data()).copy()

        # always saves image
        self.wrist_images.append(wrist_image.copy())
        wrist_image_vis = cv2.resize(wrist_image, (W, H))  # images are at higher resolution for visualization

        if self.show_wrist:
            cv2.imshow('wrist', cv2.cvtColor(wrist_image_vis, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        # convert messages
        robot_pose = self.cur_robot_pose.copy()
        gripper_width = self.cur_gripper_width

        # robot_pose is [x, y, z, rx, ry, rz]
        eef_pos = robot_pose[0:3].copy()
        eef_rot_axis_angle = robot_pose[3:6].copy()
        gripper_width_m = wsg_width_mm_to_policy_m(gripper_width)

        camera_meta = self.cfg.task.shape_meta.obs.camera0_rgb
        robot_meta = self.cfg.task.shape_meta.obs.robot0_eef_pos
        gripper_meta = self.cfg.task.shape_meta.obs.robot0_gripper_width

        current_obs_sample = {
            "camera0_rgb": wrist_image,
            "robot0_eef_pos": eef_pos,
            "robot0_eef_rot_axis_angle": eef_rot_axis_angle,
            "robot0_gripper_width": np.array([gripper_width_m], dtype=np.float32),
        }
        history_times = np.array(self.history_timestamps + [cur_timestamp])
        history_samples = self.history_obs_dicts + [current_obs_sample]

        def stack_history(key, meta):
            horizon = int(meta.horizon)
            target_times = (
                cur_timestamp
                - np.arange(horizon - 1, -1, -1) / DATA_FREQUENCY
            )
            indices = [
                int(np.argmin(np.abs(history_times - target_time)))
                for target_time in target_times
            ]
            return np.stack([history_samples[index][key] for index in indices], axis=0)

        env_obs = {
            "camera0_rgb": stack_history("camera0_rgb", camera_meta),
            "robot0_eef_pos": stack_history("robot0_eef_pos", robot_meta),
            "robot0_eef_rot_axis_angle": stack_history("robot0_eef_rot_axis_angle", robot_meta),
            "robot0_gripper_width": stack_history("robot0_gripper_width", gripper_meta),
        }

        if self.episode_start_pose is None:
            self.episode_start_pose = [
                np.concatenate([
                    env_obs["robot0_eef_pos"][-1],
                    env_obs["robot0_eef_rot_axis_angle"][-1],
                ])
            ]

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

            obs_dict_np = get_real_umi_obs_dict(
                env_obs=env_obs,
                shape_meta=self.cfg.task.shape_meta,
                obs_pose_repr=self.obs_pose_rep,
                tx_robot1_robot0=None,
                episode_start_pose=self.episode_start_pose,
            )

            # device transfer
            obs_dict = dict_apply(
                obs_dict_np,
                lambda x: torch.from_numpy(x).unsqueeze(0).to(device=self.device),
            )

            # run policy
            with torch.no_grad():
                action_dict = self.policy.predict_action(obs_dict)

            raw_action = action_dict["action"][0].detach().to("cpu").numpy()
            action_chunk = get_real_umi_action(raw_action, env_obs, self.action_pose_repr)
            if len(action_chunk) > self.model_action_steps:
                self.get_logger().warning(
                    "Policy returned {} action steps after n_action_steps override; "
                    "cropping to {} to match inference_umi_reference.py.".format(
                        len(action_chunk), self.model_action_steps
                    ),
                    once=True,
                )
                action_chunk = action_chunk[:self.model_action_steps]
            action_chunk = action_chunk.copy()

            if not np.all(np.isfinite(action_chunk)):
                raise RuntimeError('Nan or Inf action')

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
            self.predicted_action_chunks.append(action_chunk.copy())

            # Convert UMI policy gripper width in meters back to the WSG command in mm.
            action_chunk[:, -1] = np.array([
                policy_width_m_to_wsg_mm(width_m) for width_m in action_chunk[:, -1]
            ], dtype=np.float32)

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

            try:
                action_pos_steps = np.linalg.norm(np.diff(action_chunk[:, :3], axis=0), axis=1)
                traj_arr = np.asarray(self.next_traj, dtype=np.float32)
                traj_pos_steps = (
                    np.linalg.norm(np.diff(traj_arr[:, :3], axis=0), axis=1)
                    if len(traj_arr) > 1
                    else np.asarray([], dtype=np.float32)
                )
                debug_record = {
                    "time": time.time(),
                    "counter": int(self.counter),
                    "raw_action_steps": int(raw_action.shape[0]),
                    "selected_action_steps": int(action_chunk.shape[0]),
                    "base_data_frequency": float(DATA_FREQUENCY),
                    "action_down_sample_steps": int(self.action_down_sample_steps),
                    "policy_action_frequency": float(self.policy_action_frequency),
                    "interpolation_factor": float(self.interpolation_factor),
                    "eef_pos": eef_pos.astype(float).tolist(),
                    "first_target_pos": action_chunk[0, :3].astype(float).tolist(),
                    "first_target_dist_m": float(np.linalg.norm(action_chunk[0, :3] - eef_pos)),
                    "last_action_dist_m": (
                        None
                        if self.last_action is None
                        else float(np.linalg.norm(self.last_action[:3] - eef_pos))
                    ),
                    "action_step_mean_m": float(np.mean(action_pos_steps)) if len(action_pos_steps) else 0.0,
                    "action_step_max_m": float(np.max(action_pos_steps)) if len(action_pos_steps) else 0.0,
                    "traj_len": int(len(self.next_traj)),
                    "traj_step_mean_m": float(np.mean(traj_pos_steps)) if len(traj_pos_steps) else 0.0,
                    "traj_step_max_m": float(np.max(traj_pos_steps)) if len(traj_pos_steps) else 0.0,
                    "action_chunk_pos": action_chunk[:, :3].astype(float).tolist(),
                }
                with open(self.prediction_debug_path, 'a') as f:
                    f.write(json.dumps(debug_record) + '\n')
            except Exception as e:
                self.get_logger().warning('Prediction debug logging failed: {}'.format(e), once=True)

            # updates
            self.next_traj_ready = True
            self.last_action = action_chunk[-1].copy()  # this would be the last latency step

        # update history
        if len(self.history_obs_dicts) >= self.history_len:
            self.history_timestamps.pop(0)
            self.history_obs_dicts.pop(0)
        self.history_timestamps.append(cur_timestamp)
        self.history_obs_dicts.append(deepcopy(current_obs_sample))

    
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', '--ckpt-path', dest='ckpt_path', required=True)
    parser.add_argument('--results_path', '--results-path', dest='results_path', default='data/turn_knob_ros_logs')
    parser.add_argument('--show_wrist', '--show-wrist', dest='show_wrist', action='store_true')
    parsed_args, ros_args = parser.parse_known_args(args)

    rclpy.init(args=ros_args)
    dp_inference_node = DiffusionPolicyInference(parsed_args.ckpt_path, parsed_args.results_path, parsed_args.show_wrist)
    rclpy.spin(dp_inference_node)

    dp_inference_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
