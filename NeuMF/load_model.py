import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from NeuMF import neumf


from sklearn.preprocessing import LabelEncoder

# === Step 1: Reload item encoder ===
train_df = pd.read_csv("C:/Users/alexp/Licenta/Model/NeuMF/data/neumf_interactions_with_clean_item_ids.csv")
item_encoder = LabelEncoder()
item_encoder.fit(train_df["item_id"])

# === Step 2: Load trained model and updated car metadata ===
model = tf.keras.models.load_model("C:/Users/alexp/Licenta/Model/NeuMF/data/neumf_model.keras")
car_df = pd.read_csv("C:/Users/alexp/Licenta/Model/NeuMF/data/genmodel_metadata_with_clean_item_id.csv")

# Only keep valid trained items
valid_items = set(item_encoder.classes_)
car_df = car_df[car_df["item_id"].isin(valid_items)].copy()

