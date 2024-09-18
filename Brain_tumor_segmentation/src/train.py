from model import hybrid_3dcnn_attention_model
from utils import preprocess_h5_data
from tensorflow.keras.optimizers import Adam

def train_model(csv_file, data_path, old_base_path, epochs=50, batch_size=4):

    X_train, y_train = preprocess_h5_data(csv_file, data_path, old_base_path)


    model = hybrid_3dcnn_attention_model(input_shape=X_train.shape[1:])


    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)


    model.save('models/brain_tumor_segmentation_model.h5')

if __name__ == "__main__":
    csv_file = '../data/BraTS2020_training_data/content/data/meta_data.csv'
    data_path = 'C:/Users/Admin/Desktop/Brain tumor/brain_tumor_segmentation/data/BraTS2020_training_data/content/data/'
    old_base_path = '/content/data/'
    train_model(csv_file, data_path, old_base_path)
