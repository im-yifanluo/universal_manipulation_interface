import h5py

path = "alternative_data_workspace/episode_1.hdf5"

# with h5py.File(path, "r") as f:
#     def show(name, obj):
#         if isinstance(obj, h5py.Dataset):
#             print(name, obj.shape, obj.dtype)
#         else:
#             print(name, "GROUP")

#     f.visititems(show)


with h5py.File(path, "r") as f:
    eef_pos = f["obs/eef_pos"][:]
    print(eef_pos.shape)
    print(eef_pos[:5])