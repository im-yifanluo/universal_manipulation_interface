"""
Offline deployment simulation: feed the model obs from the training zarr exactly like
inference_umi.py does, compare predicted chunks to ground truth, and check the seam:
does chunk A's step 8 sit ahead of chunk B's steps 0/1/2 when obs are truthful (no robot lag)?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch, dill, hydra, zarr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.real_world.real_inference_util import get_real_umi_obs_dict, get_real_umi_action
register_codecs()

CKPT = "checkpoints/turn_knob/latest(5).ckpt"
ZARR = "data/complete_knob_training_dataset.zarr.zip"
EPISODES = [1, 5, 9]
STRIDE = 7          # rows between simulated inferences (0.7s, matches deployment cadence)
OUT = "data/offline_seam_check.png"

# ---- load model ----
payload = torch.load(open(CKPT, "rb"), pickle_module=dill, weights_only=False)
cfg = payload["cfg"]
workspace = hydra.utils.get_class(cfg._target_)(cfg)
workspace: BaseWorkspace
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
policy = workspace.ema_model if cfg.training.use_ema else workspace.model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
policy.to(device).eval()
obs_pose_repr = cfg.task.pose_repr.obs_pose_repr
action_pose_repr = cfg.task.pose_repr.action_pose_repr

# ---- load data ----
root = zarr.group(zarr.ZipStore(ZARR, mode="r"))
ee = root["meta/episode_ends"][:]

def predict(s, t):
    """simulate one deployment inference at absolute row s+t (obs = rows t-1, t)"""
    idx = [s + t - 1, s + t]
    env_obs = {
        "camera0_rgb": root["data/camera0_rgb"].oindex[idx],
        "robot0_eef_pos": root["data/robot0_eef_pos"].oindex[idx],
        "robot0_eef_rot_axis_angle": root["data/robot0_eef_rot_axis_angle"].oindex[idx],
        "robot0_gripper_width": root["data/robot0_gripper_width"].oindex[idx],
    }
    start = np.concatenate([root["data/robot0_eef_pos"][s], root["data/robot0_eef_rot_axis_angle"][s]])
    obs_np = get_real_umi_obs_dict(env_obs=env_obs, shape_meta=cfg.task.shape_meta,
        obs_pose_repr=obs_pose_repr, tx_robot1_robot0=None, episode_start_pose=[start])
    obs = dict_apply(obs_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
    with torch.no_grad():
        raw = policy.predict_action(obs)["action"][0].cpu().numpy()
    return get_real_umi_action(raw, env_obs, action_pose_repr)   # (16, 7) absolute [pose6, grip]

# ---- run over episodes ----
all_err = {}; seams = []; chunks_by_ep = {}
for ep in EPISODES:
    s, e = (0 if ep == 0 else ee[ep-1]), ee[ep]
    ts = list(range(2, e - s - 17, STRIDE))
    chunks = {t: predict(s, t) for t in ts}
    chunks_by_ep[ep] = (s, e, chunks)
    gtp = root["data/robot0_eef_pos"][s:e]
    gtg = root["data/robot0_gripper_width"][s:e, 0]
    for t, c in chunks.items():
        for k in range(16):
            all_err.setdefault(k, []).append((
                np.linalg.norm(c[k, :3] - gtp[t + k]) * 1000,
                abs(c[k, 6] - gtg[t + k]) * 1000))
    # seam: consecutive inferences 7 rows apart; project on gt motion direction
    for t, t2 in zip(ts[:-1], ts[1:]):
        A, B = chunks[t], chunks[t2]
        d = gtp[t2 + 2] - gtp[t2 - 2]; n = np.linalg.norm(d)
        if n < 1e-6: continue
        u = d / n
        pj = lambda p: float(np.dot(p, u)) * 1000.0  # mm along motion direction
        seams.append(dict(A7=pj(A[7, :3]), A8=pj(A[8, :3]),
                          B0=pj(B[0, :3]), B1=pj(B[1, :3]), B2=pj(B[2, :3]),
                          gA8=A[8, 6] * 1000, gB0=B[0, 6] * 1000, gB2=B[2, 6] * 1000))

# ---- report ----
print(f"model: {CKPT}   episodes {EPISODES}   {sum(len(c[2]) for c in chunks_by_ep.values())} inferences")
print("\nprediction error vs ground truth (mean over all chunks):")
print("  horizon step:   " + "  ".join(f"{k:>5}" for k in [0, 1, 2, 4, 8, 12, 15]))
print("  pos err (mm):   " + "  ".join(f"{np.mean([a for a,_ in all_err[k]]):5.1f}" for k in [0, 1, 2, 4, 8, 12, 15]))
print("  grip err (mm):  " + "  ".join(f"{np.mean([g for _,g in all_err[k]]):5.1f}" for k in [0, 1, 2, 4, 8, 12, 15]))
sm = {k: np.mean([s[k] for s in seams]) for k in seams[0]}
fwd = np.mean([(s["B2"] - s["A8"]) > 0 for s in seams]) * 100
print(f"\nSEAM with truthful obs ({len(seams)} boundaries), projected on motion direction (mm):")
print(f"  A8 - B0 = {sm['A8']-sm['B0']:+6.2f}   (on-robot this was ~ +lag, B anchored behind)")
print(f"  A8 - B1 = {sm['A8']-sm['B1']:+6.2f}   (expect ~0: both forecast the same instant)")
print(f"  B2 - A8 = {sm['B2']-sm['A8']:+6.2f}   forward {fwd:.0f}% of boundaries (on-robot: backward 74%)")
print(f"  gripper: A8 - B0 = {sm['gA8']-sm['gB0']:+6.2f} mm,  B2 - A8 = {sm['gB2']-sm['gA8']:+6.2f} mm")

# ---- graph episode 1 ----
ep = EPISODES[0]; s, e, chunks = chunks_by_ep[ep]
gtp = root["data/robot0_eef_pos"][s:e]; gtg = root["data/robot0_gripper_width"][s:e, 0] * 1000
pc = gtp - gtp.mean(0); _, _, vt = np.linalg.svd(pc, full_matrices=False)
proj = pc @ vt[0] * 1000
tt = np.arange(len(gtp)) / 10
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
ax1.plot(tt, proj, "k-", lw=2, alpha=.5, label="ground truth")
ax2.plot(tt, gtg, "k-", lw=2, alpha=.5, label="ground truth")
for i, (t, c) in enumerate(chunks.items()):
    col = plt.cm.tab10(i % 10)
    ck = np.arange(16)
    ax1.plot((t + ck) / 10, ((c[:, :3] - gtp.mean(0)) @ vt[0]) * 1000, "-o", color=col, ms=2.5, lw=1)
    ax2.plot((t + ck) / 10, c[:, 6] * 1000, "-o", color=col, ms=2.5, lw=1)
ax1.set_ylabel("eef pos, main axis (mm)"); ax1.legend(fontsize=9); ax1.grid(alpha=.25)
ax1.set_title(f"offline deployment simulation — ep{ep}, one predicted 16-step chunk every 0.7s (truthful obs)")
ax2.set_ylabel("gripper width (mm)"); ax2.grid(alpha=.25)
ax2.set_xlabel("time (s)")
ks = sorted(all_err)
ax3.plot(ks, [np.mean([a for a, _ in all_err[k]]) for k in ks], "-o", label="pos err (mm)")
ax3.plot(ks, [np.mean([g for _, g in all_err[k]]) for k in ks], "-o", label="gripper err (mm)")
ax3.set_xlabel("horizon step k"); ax3.set_ylabel("mean |pred - gt| (mm)")
ax3.legend(fontsize=9); ax3.grid(alpha=.25); ax3.set_title("prediction error vs horizon depth (all episodes)")
plt.tight_layout(); plt.savefig(OUT, dpi=110)
print("\nsaved", OUT)
