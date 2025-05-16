import h5py
import numpy as np

file_path = '/home/tonyz/code_bases/UoB/data/raw/recording_2022-08-17_trial2-arm/1_LF.mat'

def print_hdf5_item_info(item, indent=""):
    """ Prints information about a single HDF5 item (Dataset or Group). """
    if isinstance(item, h5py.Dataset):
        print(f"{indent}Dataset: {item.name}, Shape: {item.shape}, Dtype: {item.dtype}")
        # Optionally print data if it's small or a reference
        if item.dtype == h5py.ref_dtype or np.prod(item.shape) < 10:
             print(f"{indent}  Data: {item[()]}")
        # If it's a reference, try to print the target
        if item.dtype == h5py.ref_dtype:
            try:
                # Try dereferencing, assuming single reference per element
                # This might need adjustment if it's an array of references
                target = item.file[item[()][0]] # Example: Dereference first reference
                print(f"{indent}  -> Target: {target.name}, Type: {type(target)}")
                if isinstance(target, h5py.Dataset):
                     print(f"{indent}     Target Shape: {target.shape}, Dtype: {target.dtype}")
                elif isinstance(target, h5py.Group):
                     print(f"{indent}     Target is a Group.")
                     # Optionally recurse into the target group if needed
                     # print_hdf5_structure(target, indent + "     ")
            except Exception as e:
                print(f"{indent}  -> Could not dereference: {e}")
    elif isinstance(item, h5py.Group):
        print(f"{indent}Group: {item.name}")
        print_hdf5_structure(item, indent + "  ")
    else:
         print(f"{indent}Unknown item type: {item.name}, Type: {type(item)}")

def print_hdf5_structure(group, indent=""):
    """ Recursively prints the structure of an HDF5 group. """
    for key, item in group.items():
        print_hdf5_item_info(item, indent)


try:
    with h5py.File(file_path, 'r') as f:
        print(f"Structure starting from ImgData in {file_path}:")
        if 'ImgData' in f:
            imgdata_item = f['ImgData']
            print_hdf5_item_info(imgdata_item) # Print info about the top-level ImgData item
            # If it's a reference dataset, maybe try dereferencing its target?
            if isinstance(imgdata_item, h5py.Dataset) and imgdata_item.dtype == h5py.ref_dtype:
                 print(f"Attempting to dereference target of /ImgData:")
                 try:
                     target = f[imgdata_item[0][0]] # Assuming shape (1,1) or similar
                     print_hdf5_item_info(target, indent="  ")
                 except Exception as e:
                     print(f"  Could not dereference target: {e}")
        else:
            print("'/ImgData' not found in the file.")

except Exception as e:
    print(f"Error reading {file_path}: {e}")
