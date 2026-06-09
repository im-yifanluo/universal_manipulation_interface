import time
import pathlib
import sys
import cv2
import numpy as np
import torch
import dill
import hydra
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation as R

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from umi.real_world.wsg_binary_driver import WSGBinaryDriver
from umi.real_world.real_inference_util import get_real_umi_obs_dict, get_real_umi_action

import pyrealsense2 as rs

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge


H = 240
W = 320

STARTING_JOINT_POS = np.array([
    [0.7975698709487915, -1.6132055721678675, 2.1969855467425745, -2.145191808740133, -1.5728967825519007, -0.7886541525470179],
    [0.7946797013282776, -1.5261031997254868, 2.263378445302145, -2.308000703851217, -1.5659673849688929, -0.783783260975973]
])
STARTING_GRIPPER_WIDTH = 85.
MAX_GRIPPER_WIDTH = 110.
DATASET_MAX_GRIPPER_WIDTH_M = 0.09
DATA_FREQUENCY = 10.
SLOP = 0.02
FPS = 30


def wsg_width_mm_to_policy_m(width_mm):
    width_m = width_mm / 1000
    return np.float32(np.clip(width_m, 0.0, DATASET_MAX_GRIPPER_WIDTH_M))


def policy_width_m_to_wsg_mm(width_m):
    width_mm = width_m * 1000
    return float(np.clip(width_mm, 0.0, MAX_GRIPPER_WIDTH))


class DiffusionPolicyInference ():
    
    def __init__ (self, ckpt_path):

        self.close_cur_episode = False
        self.last_action_timestamp = None
        

        # ===== gripper init =====

        # gripper init
        self.wsg = WSGBinaryDriver(hostname="192.168.1.20", port=1000)
        self.wsg.__enter__()
        self.wsg.ack_fault()
        self.wsg.homing(positive_direction=True, wait=True)

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
        for joint_pos in STARTING_JOINT_POS:
            self.rtde_c.moveJ(joint_pos, 0.4, 1, False)


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

        self.cfg = cfg
        self.obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
        self.n_action_steps = int(cfg.get("n_action_steps", cfg.task.shape_meta.action.horizon))
        self.episode_start_pose = None

        self.policy.to(torch.device('cuda:0'))
        self.policy.eval()
        self.device = self.policy.device

        self.rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')


        # ===== ros init =====

        # self.cvbridge = CvBridge()
        # self.realsense_sub = self.create_subscription(Image, 'realsense_images', self.callback, 1)
        
        # ===== UMI init =====
        self.action_pose_repr = cfg.task.pose_repr.action_pose_repr
        
        # ===== Camera init =====
        # start realsense cameras
        # umi setup only has wrist camera
        self.pipeline_1 = rs.pipeline()
        config_1 = rs.config()
        config_1.enable_stream(rs.stream.color, W*2, H*2, rs.format.rgb8, FPS)
        self.pipeline_1.start(config_1)


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


    def callback (self):

        # # convert message
        # realsense_images = self.cvbridge.imgmsg_to_cv2(realsense_msg, 'passthrough')

        # # left half is wrist cam image, right half is front cam image
        # wrist_image = realsense_images[:, :W, :].copy()
        # front_image = realsense_images[:, W:, :].copy()

        # # robot_pose is [x, y, z, rx, ry, rz]
        # robot_pose = np.array(self.rtde_r.getActualTCPPose())
        # eef_pos = robot_pose[0:3].copy()
        # eef_quat = R.from_rotvec(robot_pose[3:6]).as_quat()

        # # get current gripper position
        # info = self.wsg.script_query()
        # gripper_width = info["position"]  # gripper_width is not normalized
        # gripper_qpos = np.array([gripper_width / MAX_GRIPPER_WIDTH, gripper_width / MAX_GRIPPER_WIDTH])

        # # create obs dict
        # np_obs_dict = dict()

        # # fill in all input required by dp
        # np_obs_dict['wrist_camera'] = wrist_image
        # np_obs_dict['front_camera'] = front_image
        # np_obs_dict['eef_pos'] = eef_pos
        # np_obs_dict['eef_quat'] = eef_quat
        # np_obs_dict['gripper_qpos'] = gripper_qpos

        # # swap axis for images and normalize
        # np_obs_dict['wrist_camera'] = np.swapaxes(np.swapaxes(np_obs_dict['wrist_camera'], 1, 2), 0, 1).astype(np.float32)/255.
        # np_obs_dict['front_camera'] = np.swapaxes(np.swapaxes(np_obs_dict['front_camera'], 1, 2), 0, 1).astype(np.float32)/255.
        
        # get realsense images
        # realsense 1, wrist camera
        frames_1 = self.pipeline_1.wait_for_frames()
        color_frame_1 = frames_1.get_color_frame()
        wrist_image = np.asanyarray(color_frame_1.get_data()).copy()

        # Modified UMI inference OBS build
        camera_horizon = self.cfg.task.shape_meta.obs.camera0_rgb.horizon
        robot_horizon = self.cfg.task.shape_meta.obs.robot0_eef_pos.horizon
        gripper_horizon = self.cfg.task.shape_meta.obs.robot0_gripper_width.horizon

        # robot_pose is [x, y, z, rx, ry, rz]
        robot_pose = np.array(self.rtde_r.getActualTCPPose())
        eef_pos = robot_pose[0:3].copy()

        # WSG reports width in mm; UMI observations use meters.
        info = self.wsg.script_query()
        gripper_width = info["position"]
        gripper_width_m = wsg_width_mm_to_policy_m(gripper_width)
        print(f"Gripper width: {gripper_width:.3f} mm -> policy {gripper_width_m:.4f} m")

        eef_pos = eef_pos.astype(np.float32)
        eef_rot_axis_angle = robot_pose[3:6].astype(np.float32)
        env_obs = {
            "camera0_rgb": np.repeat(wrist_image[None], camera_horizon, axis=0),
            "robot0_eef_pos": np.repeat(eef_pos[None], robot_horizon, axis=0),
            "robot0_eef_rot_axis_angle": np.repeat(eef_rot_axis_angle[None], robot_horizon, axis=0),
            "robot0_gripper_width": np.full((gripper_horizon, 1), gripper_width_m, dtype=np.float32),
        }

        if self.episode_start_pose is None:
            self.episode_start_pose = [
                np.concatenate([
                    env_obs["robot0_eef_pos"][-1],
                    env_obs["robot0_eef_rot_axis_angle"][-1],
                ])
            ]

        obs_dict_np = get_real_umi_obs_dict(
            env_obs=env_obs,
            shape_meta=self.cfg.task.shape_meta,
            obs_pose_repr=self.obs_pose_rep,
            tx_robot1_robot0=None,
            episode_start_pose=self.episode_start_pose,
        )


        # device transfer and add batch dimension
        obs_dict = dict_apply(
            obs_dict_np,
            lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device),
        )

        # run policy
        with torch.no_grad():
            action_dict = self.policy.predict_action(obs_dict)
        # self.get_logger().info("Predicted new action chunk.")
        self.last_action_timestamp = None

        # # device_transfer
        # np_action_dict = dict_apply(action_dict,
        #     lambda x: x.detach().to('cpu').numpy())

        # action = np_action_dict['action']
        # if not np.all(np.isfinite(action)):
        #     raise RuntimeError('Nan or Inf action')
        # action_chunk = self.undo_transform_action(action)[0]  # raw shape is (1, chunk_size, action_dim)

        # # convert gripper width back
        # action_chunk[:, -1] *= MAX_GRIPPER_WIDTH
        
        raw_action = action_dict["action_pred"][0].detach().to("cpu").numpy()
        action_chunk = get_real_umi_action(raw_action, env_obs, self.action_pose_repr)
        action_chunk = action_chunk[:self.n_action_steps]

        if not np.all(np.isfinite(action_chunk)):
            raise RuntimeError("Nan or Inf action")

        # print("Predicted action chunk:", action_chunk)

        # execute actions one by one at data frequency
        for action in action_chunk:

            cur_timestamp = time.monotonic()
            if self.last_action_timestamp is not None:
                cur_fps = 1. / (cur_timestamp - self.last_action_timestamp)
            self.last_action_timestamp = cur_timestamp

            target_pose = action[0:6]
            # if requested robot movement too large, ignore it
            if not np.average(np.abs(target_pose[0:3] - np.array(self.rtde_r.getActualTCPPose())[0:3])) > 0.05:
                self.rtde_c.servoL(target_pose, 0.01, 0.01, 1./DATA_FREQUENCY, 0.05, 100)  # params: pose, speed, accel, time, lookahead_time, gain
            
            target_gripper_width = policy_width_m_to_wsg_mm(action[-1])
            print(f"Predicted gripper: {action[-1]:.4f} m -> target {target_gripper_width:.2f} mm")

            # if the requested movement is too large, ignore it
            if not abs(target_gripper_width - gripper_width) > 40.:
                info = self.wsg.script_position_pd(
                    position = target_gripper_width,
                    velocity = self.gripper_velocity, 
                    blocked_force_limit = 20, 
                    kp = self.gripper_kp, 
                    kd = self.gripper_kd,
                )
                gripper_width = info["position"]
            else:
                print(f"Skipping gripper target; requested jump is {target_gripper_width - gripper_width:.2f} mm")
            
            # sleep until next action execution
            cur_timestamp = time.monotonic()
            if cur_timestamp - self.last_action_timestamp < 1./DATA_FREQUENCY:
                time.sleep(1./DATA_FREQUENCY - (cur_timestamp - self.last_action_timestamp))


def main (args=None):

    ckpt_path = 'checkpoints/turn_knob/latest(2).ckpt'
    

    dp_inference_node = DiffusionPolicyInference(ckpt_path)
    while True:
        dp_inference_node.callback()

    # rclpy.init(args=args)
    # rclpy.spin(dp_inference_node)

    # dp_inference_node.destroy_node()
    # rclpy.shutdown()

if __name__ == '__main__':
    main()
