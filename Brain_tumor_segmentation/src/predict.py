from tensorflow.keras.models import load_model
from utils import load_data

csv_path = 'C:/Users/Admin/Desktop/Brain tumor/brain_tumor_segmentation/data/BraTS2020_training_data/content/data/new_scan.csv'

X_new, _ = load_data(csv_path)

model = load_model("brain_tumor_segmentation_model.h5")

predictions = model.predict(X_new)