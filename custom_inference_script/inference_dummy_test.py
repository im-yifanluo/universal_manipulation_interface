import time
import cv2
import numpy as np
import torch
import dill
import hydra
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.model.common.rotation_transformer import RotationTransformer


class DiffusionPolicyExperiment:

    def __init__(self, ckpt_path):
        
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

        self.rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

    
    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
    

    def dummy_input_inference (self):

        while True:
            device = self.policy.device

            # create obs dict
            np_obs_dict = dict()

            # dummy input
            np_obs_dict['wrist_camera'] = (np.ones((240, 320, 3))*255).astype(np.uint8)
            np_obs_dict['front_camera'] = (np.ones((240, 320, 3))*255).astype(np.uint8)
            np_obs_dict['eef_pos'] = np.zeros((3,))
            np_obs_dict['eef_quat'] = np.array([0., 0., 0., 1.])  # scalar last
            np_obs_dict['gripper_qpos'] = np.array([0., 0.])  # normalized between 0 and 1

            # swap axis for images and normalize
            np_obs_dict['wrist_camera'] = np.swapaxes(np.swapaxes(np_obs_dict['wrist_camera'], 1, 2), 0, 1).astype(np.float32)/255.
            np_obs_dict['front_camera'] = np.swapaxes(np.swapaxes(np_obs_dict['front_camera'], 1, 2), 0, 1).astype(np.float32)/255.

            # device transfer
            obs_dict = dict_apply(np_obs_dict, 
                lambda x: torch.from_numpy(x).to(
                    device=device))
            
            # expand and repeat in certain dimensions to match the original tensor shape
            obs_dict = {
                key: value.unsqueeze(0).unsqueeze(0).repeat((1, 2,) + (1,) * len(value.shape)) for key, value in obs_dict.items()
            }

            # run policy
            cur_time = time.time()
            with torch.no_grad():
                action_dict = self.policy.predict_action(obs_dict)
            print('model inference time: {}'.format(time.time() - cur_time))

            # device_transfer
            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').numpy())

            action = np_action_dict['action']
            if not np.all(np.isfinite(action)):
                raise RuntimeError('Nan or Inf action')
            
            # step env
            env_action = self.undo_transform_action(action)

            print(env_action)


def main (args=None):

    ckpt_path = '/home/jingyixiang/diffusion_policy/data/outputs/2026.01.05/22.55.06_train_diffusion_unet_image_ur5e_grasp_cube_image/checkpoints/latest.ckpt'
    exp = DiffusionPolicyExperiment(ckpt_path = ckpt_path)
    exp.dummy_input_inference()


if __name__ == '__main__':
    main()