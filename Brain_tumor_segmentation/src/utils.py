import h5py
import os
import numpy as np
import pandas as pd

def load_h5_file_in_batches(h5_file_path, batch_size=32):
    with h5py.File(h5_file_path, 'r') as h5_file:
        num_slices = h5_file['image'].shape[2]
        for i in range(0, num_slices, batch_size):
            image_batch = np.array(h5_file['image'][:, :, i:i + batch_size], dtype=np.float32)
            mask_batch = np.array(h5_file['mask'][:, :, i:i + batch_size], dtype=np.float32)
            yield image_batch, mask_batch


def correct_paths(metadata, old_base_path, new_base_path):
    #
    metadata['slice_path'] = metadata['slice_path'].str.replace(old_base_path, new_base_path)
    return metadata

def preprocess_h5_data(csv_file, data_path, old_base_path, batch_size=32):
    data = pd.read_csv(csv_file)
    X_train, y_train = [], []
    
    for _, row in data.iterrows():
        h5_file_path = os.path.join(data_path, row['slice_path'].replace(old_base_path, ''))
        for image_batch, mask_batch in load_h5_file_in_batches(h5_file_path, batch_size):
            X_train.append(image_batch)
            y_train.append(mask_batch)
    
    X_train = np.concatenate(X_train, axis=2)  
    y_train = np.concatenate(y_train, axis=2) 
    return X_train, y_train
