import h5py

def inspect_h5_file(file_path):
    with h5py.File(file_path, 'r') as h5_file:
    
        print("Keys in the HDF5 file:")
        h5_file.visit(print)

if __name__ == "__main__":
    h5_file_path = 'C:/Users/Admin/Desktop/Brain tumor/brain_tumor_segmentation/data/BraTS2020_training_data/content/data/volume_1_slice_0.h5'
    inspect_h5_file(h5_file_path)
