import pathlib
import sys

import zarr

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

register_codecs()

path = "custom_data_pipeline/dataset.zarr.zip"

with zarr.ZipStore(path, mode="r") as store:
    root = zarr.group(store)
    print(root.tree())

    print("\nDATA")
    for key, arr in root["data"].items():
        print(key, arr.shape, arr.dtype)

    print("\nMETA")
    for key, arr in root["meta"].items():
        value = arr[:] if arr.shape else arr[()]
        print(key, arr.shape, arr.dtype, value)
