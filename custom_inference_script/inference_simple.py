import time
import cv2
import numpy as np
import torch
import dill
import hydra
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from gripper_controller import *
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


H = 240
W = 320
STARTING_POSE = np.array([-0.25, -0.45, 0.4, 0, np.pi, 0])
STARTING_GRIPPER_WIDTH = 105.
MAX_GRIPPER_WIDTH = 110.
CONTROL_FREQUENCY = 50.
DATA_FREQUENCY = 10.
SLOP = 0.02


def interpolate_between_poses (start_pos, start_rotvec, end_pos, end_rotvec, interpolation_factor):
    '''
    This function interpolate between a start pose and an end pose to generate a trajectory
    with length interpolation_factor, including the end pose but not the start pose.
    '''

    # rotation interpolation
    rotations = R.from_rotvec([start_rotvec, end_rotvec])
    key_times = [0, 1]
    slerp = Slerp(key_times, rotations)

    pos_interpolated = []
    rotvec_interpolated = []

    for i in range (1, int(interpolation_factor)):
        next_pos = (end_pos - start_pos) * i / interpolation_factor + start_pos
        next_rotvec = slerp(float(i) / interpolation_factor).as_rotvec()

        pos_interpolated.append(next_pos)
        rotvec_interpolated.append(next_rotvec)

    pos_interpolated.append(end_pos)
    rotvec_interpolated.append(end_rotvec)

    return pos_interpolated, rotvec_interpolated


def interpolate_between_widths (start_width, end_width, interpolation_factor):
    width_interpolated = []

    for i in range (1, int(interpolation_factor)):
        next_width = (end_width - start_width) * i / interpolation_factor + start_width
        width_interpolated.append(next_width)
    
    width_interpolated.append(end_width)

    return width_interpolated


class DiffusionPolicyInference (Node):
    
    def __init__ (self, ckpt_path):

        super().__init__('diffusion_policy_inference')

        # simple logging
        self.timestep_limit = 500
        self.trajectory_received = []
        self.trajectory_executed = []
        
        # for interpolation
        self.interpolation_factor = CONTROL_FREQUENCY / DATA_FREQUENCY
        self.last_action = None


        # ===== gripper init =====

        # gripper init
        self.wsg = WSGBinaryDriver(hostname="192.168.1.20", port=1000)
        self.wsg.__enter__()
        self.wsg.ack_fault()

        # gripper pd control
        self.gripper_velocity = 30.
        self.gripper_kp = 15.
        self.gripper_kd = 0.001
        self.gripper_step = 5.  # mm

        # move gripper to starting width
        self.wsg.pre_position(STARTING_GRIPPER_WIDTH, 100)


        # ===== robot init =====

        # rtde initialization
        self.rtde_c = rtde_control.RTDEControlInterface("192.168.1.12")
        self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.12")

        # move the robot to starting position
        self.rtde_c.moveL(STARTING_POSE, 0.1, 0.1, False)  # should be blocking


        # ===== diffusion policy init =====

        # load checkpoint
        payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)

        # # if need to overwrite
        # cfg['policy']['num_inference_steps'] = 10

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


        # ===== ros init =====

        self.cvbridge = CvBridge()
        self.realsense_sub = self.create_subscription(Image, 'realsense_images', self.callback, 1)


    def undo_transform_action(self, action):
        
        # assume always single arm
        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        return uaction


    def callback (self, realsense_msg):

        # convert message
        realsense_images = self.cvbridge.imgmsg_to_cv2(realsense_msg, 'passthrough')

        # left half is wrist cam image, right half is front cam image
        wrist_image = realsense_images[:, :W, :].copy()
        front_image = realsense_images[:, W:, :].copy()

        # robot_pose is [x, y, z, rx, ry, rz]
        robot_pose = np.array(self.rtde_r.getActualTCPPose())
        eef_pos = robot_pose[0:3].copy()
        eef_quat = R.from_rotvec(robot_pose[3:6]).as_quat()

        # get current gripper position
        info = self.wsg.script_query()
        gripper_width = info["position"]  # gripper_width is not normalized
        gripper_qpos = np.array([gripper_width / MAX_GRIPPER_WIDTH, gripper_width / MAX_GRIPPER_WIDTH])

        # create obs dict
        np_obs_dict = dict()

        # fill in all input required by dp
        np_obs_dict['wrist_camera'] = wrist_image
        np_obs_dict['front_camera'] = front_image
        np_obs_dict['eef_pos'] = eef_pos
        np_obs_dict['eef_quat'] = eef_quat
        np_obs_dict['gripper_qpos'] = gripper_qpos

        # swap axis for images and normalize
        np_obs_dict['wrist_camera'] = np.swapaxes(np.swapaxes(np_obs_dict['wrist_camera'], 1, 2), 0, 1).astype(np.float32)/255.
        np_obs_dict['front_camera'] = np.swapaxes(np.swapaxes(np_obs_dict['front_camera'], 1, 2), 0, 1).astype(np.float32)/255.

        # device transfer
        obs_dict = dict_apply(np_obs_dict, 
            lambda x: torch.from_numpy(x).to(
                device=self.device))
        
        # expand and repeat in certain dimensions to match the original tensor shape
        obs_dict = {
            key: value.unsqueeze(0).unsqueeze(0).repeat((1, 2,) + (1,) * len(value.shape)) for key, value in obs_dict.items()
        }

        # run policy
        with torch.no_grad():
            action_dict = self.policy.predict_action(obs_dict)
        self.get_logger().info("Predicted new action chunk.")
        last_action_timestamp = None

        # device_transfer
        np_action_dict = dict_apply(action_dict,
            lambda x: x.detach().to('cpu').numpy())

        action = np_action_dict['action']
        if not np.all(np.isfinite(action)):
            raise RuntimeError('Nan or Inf action')
        action_chunk = self.undo_transform_action(action)[0]  # raw shape is (1, chunk_size, action_dim)

        # convert gripper width back
        action_chunk[:, -1] *= MAX_GRIPPER_WIDTH

        # interpolate trajectory
        trajectory_pos = []
        trajectory_rotvec = []
        trajectory_gripper = []

        if self.last_action is not None:  
            # interpolate between the last executed action and the first action in the action chunk
            trajectory_pos.append(self.last_action[0:3])
            trajectory_rotvec.append(self.last_action[3:6])
            trajectory_gripper.append(self.last_action[6])

            pos_interpolated, rotvec_interpolated = interpolate_between_poses(
                self.last_action[0:3],
                self.last_action[3:6],
                action_chunk[0, 0:3],
                action_chunk[0, 3:6],
                self.interpolation_factor,
            )
            trajectory_pos += pos_interpolated
            trajectory_rotvec += rotvec_interpolated
            trajectory_gripper += interpolate_between_widths(
                self.last_action[6], 
                action_chunk[0, 6], 
                self.interpolation_factor,
            )

        else:
            # start from the start of 1st action chunk
            trajectory_pos.append(action_chunk[0, 0:3])
            trajectory_rotvec.append(action_chunk[0, 3:6])
            trajectory_gripper.append(action_chunk[0, 6])

        # interpolate between current action chunk
        for i in range (len(action_chunk)-1):
            pos_interpolated, rotvec_interpolated = interpolate_between_poses(
                action_chunk[i, 0:3],
                action_chunk[i, 3:6],
                action_chunk[i+1, 0:3],
                action_chunk[i+1, 3:6],
                self.interpolation_factor,
            )
            trajectory_pos += pos_interpolated
            trajectory_rotvec += rotvec_interpolated
            trajectory_gripper += interpolate_between_widths(
                action_chunk[i, 6], 
                action_chunk[i+1, 6], 
                self.interpolation_factor,
            )

        trajectory_pos = np.array(trajectory_pos)
        trajectory_rotvec = np.array(trajectory_rotvec)
        trajectory_gripper = np.array(trajectory_gripper).reshape((len(trajectory_gripper), 1))
        actions = np.hstack((trajectory_pos, trajectory_rotvec, trajectory_gripper))

        # execute actions one by one at data frequency
        for action in actions:

            if last_action_timestamp is not None:
                cur_timestamp = float(self.get_clock().now().nanoseconds) / 1e9
                cur_fps = 1./(cur_timestamp - last_action_timestamp)
                self.get_logger().info("Running inference at {} fps".format(cur_fps))
            last_action_timestamp = float(self.get_clock().now().nanoseconds) / 1e9

            target_pose = action[0:6]
            # if requested robot movement too large, ignore it
            if np.average(np.abs(target_pose[0:3] - np.array(self.rtde_r.getActualTCPPose())[0:3])) > 0.05:
                self.get_logger().warning('Requested robot movement too large! Not executing. Diff: {}'.format(np.round(target_pose - np.array(self.rtde_r.getActualTCPPose()), 3)))
            else:
                self.rtde_c.servoL(target_pose, 0.01, 0.01, 1./CONTROL_FREQUENCY, 0.05, 100)  # params: pose, speed, accel, time, lookahead_time, gain

            target_gripper_width = action[-1]
            # if the requested movement is too large, ignore it
            if abs(target_gripper_width - gripper_width) > 40.:
                self.get_logger().warning('Requested gripper movement too large! Not executing. Diff: {}'.format(target_gripper_width - gripper_width))
            else:
                info = self.wsg.script_position_pd(
                    position = target_gripper_width,
                    velocity = self.gripper_velocity, 
                    blocked_force_limit = 20, 
                    kp = self.gripper_kp, 
                    kd = self.gripper_kd,
                )
                gripper_width = info["position"]

            # append received trajectory
            self.trajectory_received.append(action)
            
            # append executed trajectory
            self.trajectory_executed.append(self.rtde_r.getActualTCPPose() + [gripper_width])

            if len(self.trajectory_received) >= self.timestep_limit:
                np.savez(
                    'trajectory.npz', 
                    trajectory_received = self.trajectory_received,
                    trajectory_executed = self.trajectory_executed,
                )
                self.get_logger().info('Saved received trajectory.')
                rclpy.shutdown()
            
            # sleep until next action execution
            cur_timestamp = float(self.get_clock().now().nanoseconds) / 1e9
            if cur_timestamp - last_action_timestamp < 1./CONTROL_FREQUENCY:
                time.sleep(1./CONTROL_FREQUENCY - (cur_timestamp - last_action_timestamp))
            
        # udpate last action
        self.last_action = actions[-1].copy()


def main (args=None):

    ckpt_path = '/home/jingyixiang/diffusion_policy/data/outputs/2026.01.06/grasp_cube_2/checkpoints/latest.ckpt'

    rclpy.init(args=args)
    dp_inference_node = DiffusionPolicyInference(ckpt_path)
    rclpy.spin(dp_inference_node)

    dp_inference_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()