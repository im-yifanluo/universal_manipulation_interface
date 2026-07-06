'''
Offline inference from a saved eval_once.py inference_input_XXXXXX.npz.
'''

# ===== Imports =====
import argparse
import os
import pathlib
import sys
import json
import re

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.real_world.real_inference_util import get_real_umi_action

OmegaConf.register_new_resolver("eval", eval, replace=True)
register_codecs()

# ===== Variables =====

CKPT_PATH = "checkpoints/latest.ckpt"
INPUT_NPZ_PATH = "data/eval_once_inference_inputs/debug_grasp_cube_53/inference_input_000003.npz"
OUTPUT_DIR = "data/offline_inference_robot_data_logs/debug_grasp_cube_53"
DEVICE = "cuda:0"

# ===== Inference Functions =====

def load_policy(ckpt_path, device):
    # load checkpoint
    payload = torch.load(
        open(ckpt_path, "rb"),
        map_location="cpu",
        pickle_module=dill,
    )

    # cfg is the saved Hydra config from training
    cfg = payload["cfg"]

    # build workspace class from cfg._target_
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace

    # load model weights into workspace
    workspace.load_payload(
        payload,
        exclude_keys=None,
        include_keys=None,
    )

    # choose model
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    # move to GPU and eval mode
    policy.to(torch.device(device))
    policy.eval()

    return policy, cfg

def get_npz_index(npz_path):
    match = re.search(r"inference_input_(\d+)\.npz$", npz_path.name)
    if match is None:
        return npz_path.stem
    return match.group(1)


def load_npz_inputs(npz_path, cfg):
    data = np.load(npz_path)
    obs_dict_np = {}

    input_key_map = {
        "camera0_rgb": "camera0_rgb_policy",
        "robot0_eef_pos": "policy_robot0_eef_pos",
        "robot0_eef_rot_axis_angle": "policy_robot0_eef_rot_axis_angle",
        "robot0_gripper_width": "policy_robot0_gripper_width",
        "robot0_eef_rot_axis_angle_wrt_start": (
            "policy_robot0_eef_rot_axis_angle_wrt_start"
        ),
    }

    for key, attr in cfg.task.shape_meta.obs.items():
        if attr.get("ignore_by_policy", False):
            continue
        if key not in input_key_map:
            raise KeyError(f"No saved NPZ input mapping for obs key: {key}")
        npz_key = input_key_map[key]
        if npz_key not in data.files:
            raise KeyError(f"Missing expected NPZ input array: {npz_key}")
        obs_dict_np[key] = data[npz_key]

    env_obs = {
        "robot0_eef_pos": data["robot0_eef_pos"],
        "robot0_eef_rot_axis_angle": data["robot0_eef_rot_axis_angle"],
        "robot0_gripper_width": data["robot0_gripper_width"],
    }

    return data, obs_dict_np, env_obs


def predict_from_npz(policy, cfg, npz_path, device):
    data, obs_dict_np, env_obs = load_npz_inputs(npz_path, cfg)
    obs_dict = dict_apply(
        obs_dict_np,
        lambda x: torch.from_numpy(x).unsqueeze(0).to(device),
    )

    with torch.no_grad():
        result = policy.predict_action(obs_dict)

    raw_action = result["action_pred"][0].detach().to("cpu").numpy()
    action_pose_repr = cfg.task.pose_repr.action_pose_repr
    full_horizon_prediction = get_real_umi_action(
        raw_action,
        env_obs,
        action_pose_repr,
    )

    return data, obs_dict_np, raw_action, full_horizon_prediction


def get_output_path(npz_path, output_dir):
    npz_index = get_npz_index(npz_path)
    if output_dir is None:
        output_dir = pathlib.Path("data") / "offline_inference_robot_data_logs" / npz_path.parent.name
    else:
        output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"offline_pred_{npz_index}.jsonl"


def write_prediction(output_path, ckpt_path, npz_path, cfg, data, obs_dict_np,
                     raw_action, full_horizon_prediction):
    current_pose = data["current_pose"]
    record = {
        "phase": "offline_policy",
        "ckpt_path": str(ckpt_path),
        "input_npz_path": str(npz_path),
        "npz_index": get_npz_index(npz_path),
        "obs_keys": list(obs_dict_np.keys()),
        "action_pose_repr": cfg.task.pose_repr.action_pose_repr,
        "current_pose": current_pose.tolist(),
        "raw_action_pred": raw_action.tolist(),
        "full_horizon_prediction": full_horizon_prediction.tolist(),
    }

    with open(output_path, "w") as f:
        f.write(json.dumps(record) + "\n")

# ===== Main function =====

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=CKPT_PATH)
    parser.add_argument("--npz", default=INPUT_NPZ_PATH)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    ckpt_path = pathlib.Path(args.ckpt)
    npz_path = pathlib.Path(args.npz)
    device = torch.device(args.device)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ input not found: {npz_path}")

    print("ckpt:", ckpt_path)
    print("npz:", npz_path)
    print("device:", device)

    policy, cfg = load_policy(ckpt_path, device)
    data, obs_dict_np, raw_action, full_horizon_prediction = predict_from_npz(
        policy,
        cfg,
        npz_path,
        device,
    )

    output_path = get_output_path(npz_path, args.output_dir)
    write_prediction(
        output_path,
        ckpt_path,
        npz_path,
        cfg,
        data,
        obs_dict_np,
        raw_action,
        full_horizon_prediction,
    )

    print("raw_action_pred shape:", raw_action.shape)
    print("full_horizon_prediction shape:", full_horizon_prediction.shape)
    print("saved:", output_path)


if __name__ == "__main__":
    os.chdir(str(ROOT_DIR))
    main()
