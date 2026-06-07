# =============== Imports ===============
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import json
import pathlib
import click
import zarr
import pickle
import h5py
import numpy as np
import cv2
import av
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial.transform import Rotation
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
from umi.common.cv_util import get_image_transform
register_codecs()

# =============== Helpers ===============
# %%
def resolve_hdf5_paths(input_path: str) -> list[pathlib.Path]:
    """Resolve a parent folder into a deterministic list of HDF5 episode files."""
    path = pathlib.Path(os.path.expanduser(input_path)).absolute()

    if path.is_dir():
        hdf5_paths = sorted(path.glob("*.hdf5")) + sorted(path.glob("*.h5"))
    elif path.is_file() and path.suffix.lower() in {".hdf5", ".h5"}:
        hdf5_paths = [path]
    else:
        raise FileNotFoundError(f"No HDF5 file or folder found at {path}")

    if len(hdf5_paths) == 0:
        raise FileNotFoundError(f"No .hdf5 or .h5 files found in {path}")

    return hdf5_paths


def episode_sort_key(name: str):
    """Sort episode_2 before episode_10."""
    stem = name.rstrip("/").split("/")[-1]
    prefix, _, suffix = stem.rpartition("_")
    if prefix == "episode" and suffix.isdigit():
        return int(suffix)
    return stem


def resolve_hdf5_episode_specs(hdf5_paths: list[pathlib.Path]) -> list[dict]:
    """
    Expand HDF5 files into episode specs.

    Supports both:
    - one episode per file with /obs/... at the root
    - one file containing /data/episode_*/obs/...
    """
    episode_specs = []
    for path in hdf5_paths:
        with h5py.File(path, "r") as h5_file:
            if "data" in h5_file and isinstance(h5_file["data"], h5py.Group):
                episode_keys = [
                    f"data/{name}"
                    for name, obj in h5_file["data"].items()
                    if isinstance(obj, h5py.Group) and "obs/eef_pos" in obj
                ]
                episode_keys = sorted(episode_keys, key=episode_sort_key)
                if len(episode_keys) > 0:
                    for episode_key in episode_keys:
                        episode_specs.append({
                            "path": path,
                            "episode_key": episode_key,
                        })
                    continue

        episode_specs.append({
            "path": path,
            "episode_key": None,
        })

    return episode_specs


def get_episode_group(h5_file: h5py.File, episode_key):
    if episode_key is None:
        return h5_file
    return h5_file[episode_key]


def describe_h5_group(h5_group) -> str:
    return f"{h5_group.file.filename}:/{h5_group.name.strip('/')}"

# %%
def build_lowdim_episode(
    h5_file,
    gripper_width_scale: float,
    quat_format: str = "xyzw",
) -> tuple[dict[str, np.ndarray], int]:
    """
    Read and convert one HDF5 demo's low-dimensional data.

    Returns the episode dict passed to ReplayBuffer.add_episode and the episode length.
    This function should:
    - validate required HDF5 datasets
    - convert /obs/eef_pos -> robot0_eef_pos
    - convert /obs/eef_quat -> robot0_eef_rot_axis_angle
    - convert /obs/gripper_qpos -> robot0_gripper_width
    - compute repeated robot0_demo_start_pose and robot0_demo_end_pose
    """

    # Validation
    source_name = describe_h5_group(h5_file)
    required_keys = ("obs/eef_pos", "obs/eef_quat", "obs/gripper_qpos")
    for key in required_keys:
        if key not in h5_file:
            raise KeyError(f"{source_name}: missing /{key}")

    eef_pos = h5_file["obs/eef_pos"][:]
    eef_quat = h5_file["obs/eef_quat"][:]
    gripper_qpos = h5_file["obs/gripper_qpos"][:]

    if eef_pos.ndim != 2 or eef_pos.shape[1] != 3:
        raise ValueError(f"{source_name}: /obs/eef_pos must be (T, 3), got {eef_pos.shape}")
    if eef_quat.ndim != 2 or eef_quat.shape[1] != 4:
        raise ValueError(f"{source_name}: /obs/eef_quat must be (T, 4), got {eef_quat.shape}")
    if gripper_qpos.ndim != 2 or gripper_qpos.shape[1] < 1:
        raise ValueError(f"{source_name}: /obs/gripper_qpos must be (T, >=1), got {gripper_qpos.shape}")

    episode_length = eef_pos.shape[0]
    if episode_length == 0:
        raise ValueError(f"{source_name}: empty episode")
    if eef_quat.shape[0] != episode_length or gripper_qpos.shape[0] != episode_length:
        raise ValueError(
            f"{source_name}: lowdim lengths do not match "
            f"eef_pos={eef_pos.shape[0]}, eef_quat={eef_quat.shape[0]}, "
            f"gripper_qpos={gripper_qpos.shape[0]}"
        )

    # EEF position conversion
    eef_pos = eef_pos.astype(np.float32)

    # EEF quaternion conversion
    eef_quat = eef_quat.astype(np.float64)
    if quat_format == "wxyz":
        eef_quat = eef_quat[:, [1, 2, 3, 0]]
    elif quat_format != "xyzw":
        raise ValueError(f"Unsupported quat_format {quat_format!r}")

    quat_norm = np.linalg.norm(eef_quat, axis=-1, keepdims=True)
    if np.any(quat_norm <= 1e-8):
        raise ValueError(f"{source_name}: /obs/eef_quat contains near-zero quaternion")
    eef_quat = eef_quat / quat_norm
    eef_rot_axis_angle = Rotation.from_quat(eef_quat).as_rotvec().astype(np.float32)

    # Gripper width conversion
    gripper_width = (gripper_qpos[:, :1].astype(np.float32) * gripper_width_scale).astype(np.float32)

    eef_pose = np.concatenate([eef_pos, eef_rot_axis_angle], axis=-1).astype(np.float32)

    # Demo start and end pose conversion
    demo_start_pose = np.empty_like(eef_pose)
    demo_start_pose[:] = eef_pose[0]
    demo_end_pose = np.empty_like(eef_pose)
    demo_end_pose[:] = eef_pose[-1]

    episode_data = {
        "robot0_eef_pos": eef_pos,
        "robot0_eef_rot_axis_angle": eef_rot_axis_angle,
        "robot0_gripper_width": gripper_width,
        "robot0_demo_start_pose": demo_start_pose,
        "robot0_demo_end_pose": demo_end_pose,
    }
    return episode_data, int(episode_length)


def append_lowdim_episodes(
    replay_buffer: ReplayBuffer,
    episode_specs: list[dict],
    camera_key: str,
    gripper_width_scale: float,
    quat_format: str = "xyzw",
) -> list[dict]:
    """
    First pass over HDF5 files.

    Adds each whole demo to ReplayBuffer with add_episode(...) and returns image specs:
    {"path": pathlib.Path, "camera_key": str, "buffer_start": int, "buffer_end": int}
    """
    camera_key = camera_key.strip("/")
    image_specs = []

    for spec in tqdm(episode_specs, desc="lowdim episodes"):
        path = pathlib.Path(spec["path"])
        episode_key = spec["episode_key"]
        with h5py.File(path, "r") as h5_file:
            episode_group = get_episode_group(h5_file, episode_key)
            source_name = describe_h5_group(episode_group)
            episode_data, episode_length = build_lowdim_episode(
                h5_file=episode_group,
                gripper_width_scale=gripper_width_scale,
                quat_format=quat_format,
            )

            if camera_key not in episode_group:
                raise KeyError(f"{source_name}: missing /{camera_key}")
            camera = episode_group[camera_key]
            if not isinstance(camera, h5py.Dataset):
                raise TypeError(f"{source_name}: /{camera_key} is not a dataset")
            if camera.ndim != 4 or camera.shape[-1] != 3:
                raise ValueError(f"{source_name}: /{camera_key} must be (T, H, W, 3), got {camera.shape}")
            if camera.shape[0] != episode_length:
                raise ValueError(
                    f"{source_name}: /{camera_key} length {camera.shape[0]} != lowdim length {episode_length}"
                )
            if camera.dtype != np.uint8:
                raise ValueError(f"{source_name}: /{camera_key} must be uint8, got {camera.dtype}")

            buffer_start = int(replay_buffer.n_steps)
            replay_buffer.add_episode(data=episode_data, compressors=None)
            buffer_end = int(replay_buffer.n_steps)
            if (buffer_end - buffer_start) != episode_length:
                raise RuntimeError(
                    f"{path}: added {buffer_end - buffer_start} steps, expected {episode_length}"
                )

        image_specs.append({
            "path": path,
            "episode_key": episode_key,
            "camera_key": camera_key,
            "buffer_start": buffer_start,
            "buffer_end": buffer_end,
        })

    return image_specs

# %%
def write_camera0_rgb(
    replay_buffer: ReplayBuffer,
    image_specs: list[dict],
    out_res: tuple[int, int],
    compression_level: int,
    bgr_to_rgb: bool,
    camera_name: str = "camera0_rgb",
) -> None:
    """
    Creates /data/camera0_rgb with JPEG-XL compression, then streams wrist-camera
    frames into the zarr array by global timestamp.
    """
    if len(image_specs) == 0:
        raise ValueError("No image specs to write.")

    out_w, out_h = out_res
    total_frames = int(replay_buffer["robot0_eef_pos"].shape[0])
    expected_total_frames = max(int(spec["buffer_end"]) for spec in image_specs)
    if expected_total_frames != total_frames:
        raise ValueError(
            f"Image specs end at {expected_total_frames}, but replay buffer has {total_frames} frames."
        )

    img_compressor = JpegXl(level=compression_level, numthreads=1)
    image_array = replay_buffer.data.require_dataset(
        name=camera_name,
        shape=(total_frames, out_h, out_w, 3),
        chunks=(1, out_h, out_w, 3),
        compressor=img_compressor,
        dtype=np.uint8,
    )

    for spec in tqdm(image_specs, desc="image episodes"):
        path = pathlib.Path(spec["path"])
        camera_key = str(spec["camera_key"]).strip("/")
        buffer_start = int(spec["buffer_start"])
        buffer_end = int(spec["buffer_end"])
        episode_length = buffer_end - buffer_start

        with h5py.File(path, "r") as h5_file:
            episode_group = get_episode_group(h5_file, spec.get("episode_key"))
            source_name = describe_h5_group(episode_group)
            if camera_key not in episode_group:
                raise KeyError(f"{source_name}: missing /{camera_key}")
            camera = episode_group[camera_key]
            if camera.shape[0] != episode_length:
                raise ValueError(
                    f"{source_name}: /{camera_key} length {camera.shape[0]} != image spec length {episode_length}"
                )

            in_h, in_w = camera.shape[1], camera.shape[2]
            resize_tf = get_image_transform(
                in_res=(in_w, in_h),
                out_res=out_res,
                bgr_to_rgb=bgr_to_rgb,
            )

            for frame_idx in tqdm(range(episode_length), desc=path.name, leave=False):
                img = resize_tf(camera[frame_idx])
                if img.shape != (out_h, out_w, 3):
                    raise ValueError(
                        f"{path}: converted frame has shape {img.shape}, expected {(out_h, out_w, 3)}"
                    )
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                image_array[buffer_start + frame_idx] = img

# %%
def save_replay_buffer(
    replay_buffer: ReplayBuffer,
    output_path: pathlib.Path,
) -> None:
    """Save final ReplayBuffer to a zarr zip with ReplayBuffer.save_to_store."""
    with zarr.ZipStore(str(output_path), mode="w") as zip_store:
        replay_buffer.save_to_store(store=zip_store)


def validate_output_zarr(
    output_path: pathlib.Path,
    expected_episode_lengths: list[int],
    camera_name: str = "camera0_rgb",
) -> None:
    """Check final zarr keys, first dimensions, and cumulative episode_ends."""
    required_data_keys = {
        camera_name,
        "robot0_eef_pos",
        "robot0_eef_rot_axis_angle",
        "robot0_gripper_width",
        "robot0_demo_start_pose",
        "robot0_demo_end_pose",
    }
    expected_episode_ends = np.cumsum(expected_episode_lengths, dtype=np.int64)
    if expected_episode_ends.size == 0:
        raise ValueError("No expected episodes were provided for validation.")
    expected_total_frames = int(expected_episode_ends[-1])

    with zarr.ZipStore(str(output_path), mode="r") as zip_store:
        root = zarr.group(zip_store)
        if "data" not in root:
            raise KeyError(f"{output_path}: missing /data group")
        if "meta" not in root:
            raise KeyError(f"{output_path}: missing /meta group")
        if "episode_ends" not in root["meta"]:
            raise KeyError(f"{output_path}: missing /meta/episode_ends")

        missing_data_keys = required_data_keys - set(root["data"].keys())
        if missing_data_keys:
            raise KeyError(f"{output_path}: missing /data keys {sorted(missing_data_keys)}")

        episode_ends = root["meta"]["episode_ends"][:]
        if not np.array_equal(episode_ends, expected_episode_ends):
            raise ValueError(
                f"{output_path}: episode_ends mismatch. "
                f"Expected {expected_episode_ends}, got {episode_ends}"
            )

        for key in required_data_keys:
            value = root["data"][key]
            if value.shape[0] != expected_total_frames:
                raise ValueError(
                    f"{output_path}: /data/{key} length {value.shape[0]} "
                    f"!= expected {expected_total_frames}"
                )


@click.command()
@click.option("-i", "--input", required=True)
@click.option("-o", "--output", required=True, help="Output dataset.zarr.zip path.")
@click.option("-or", "--out_res", type=str, default="224,224", help="Output WIDTH,HEIGHT.")
@click.option("--camera-key", type=str, default="obs/wrist_camera", help="HDF5 camera dataset.")
@click.option("-cl", "--compression_level", type=int, default=99)
@click.option("--gripper-width-scale", type=float, default=0.09)
@click.option("--quat-format", type=click.Choice(["xyzw", "wxyz"]), default="xyzw")
@click.option("--bgr-to-rgb", is_flag=True, default=False)
@click.option("--overwrite", is_flag=True, default=False)
def main(
    input: str,
    output: str,
    out_res: str,
    camera_key: str,
    compression_level: int,
    gripper_width_scale: float,
    quat_format: str,
    bgr_to_rgb: bool,
    overwrite: bool,
) -> None:
    """
    Convert HDF5 demos into a UMI-compatible zarr zip.

    Intended flow:
    1. resolve HDF5 files
    2. create ReplayBuffer.create_empty_zarr(...)
    3. append lowdim episodes with ReplayBuffer.add_episode(...)
    4. create and stream camera0_rgb frames
    5. save with ReplayBuffer.save_to_store(...)
    """
    output_path = pathlib.Path(os.path.expanduser(output)).absolute()
    if output_path.exists() and not overwrite:
        click.confirm(f"Output file {output_path} exists. Overwrite?", abort=True)

    out_res_tuple = tuple(int(x) for x in out_res.split(","))
    hdf5_paths = resolve_hdf5_paths(input)
    episode_specs = resolve_hdf5_episode_specs(hdf5_paths)

    replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
    image_specs = append_lowdim_episodes(
        replay_buffer=replay_buffer,
        episode_specs=episode_specs,
        camera_key=camera_key,
        gripper_width_scale=gripper_width_scale,
        quat_format=quat_format,
    )

    write_camera0_rgb(
        replay_buffer=replay_buffer,
        image_specs=image_specs,
        out_res=out_res_tuple,
        compression_level=compression_level,
        bgr_to_rgb=bgr_to_rgb,
        camera_name="camera0_rgb",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_replay_buffer(
        replay_buffer=replay_buffer,
        output_path=output_path,
    )
    validate_output_zarr(
        output_path=output_path,
        expected_episode_lengths=[
            int(spec["buffer_end"]) - int(spec["buffer_start"])
            for spec in image_specs
        ],
        camera_name="camera0_rgb",
    )

    click.echo(f"Wrote {len(image_specs)} episodes to {output_path}")


if __name__ == "__main__":
    main()
