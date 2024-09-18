from utils import preprocess_h5_data
import tensorflow as tf

def evaluate_model(model_path, csv_file, data_path):
    
    X_test, y_test = preprocess_h5_data(csv_file, data_path)
    model = tf.keras.models.load_model(model_path)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

if __name__ == "__main__":
    model_path = 'models/brain_tumor_segmentation_model.h5'
    csv_file = '../data/BraTS2020_training_data/content/data/meta_data.csv'
    data_path = '../data/BraTS2020_training_data/content/data/'
    evaluate_model(model_path, csv_file, data_path)
