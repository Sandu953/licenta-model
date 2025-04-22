import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from neumf import NeuMF

# === Step 1: Reload training item encoder ===
train_df = pd.read_csv("C:\\Users\\alexp\\Licenta\\Model\\NeuMF\\data\\neumf_training_data.csv")
item_encoder = LabelEncoder()
item_encoder.fit(train_df["item_id"])

# === Step 2: Load model and item metadata ===
model = tf.keras.models.load_model("data\\neumf_model.keras")
brand_model_df = pd.read_csv("data/car_metadata_with_item_id.csv")

# Only keep items the model was trained on
valid_items = set(item_encoder.classes_)
brand_model_df = brand_model_df[brand_model_df["item_id"].isin(valid_items)].copy()


# === Step 3: Session-based recommendation function ===
def recommend_from_recent_items_only(recent_df, model, item_encoder, brand_model_df, top_k=10):
    # 1. Combine brand and model to match item_id format
    recent_df["item_id"] = recent_df["brand"] + "_" + recent_df["model"].str.replace(" ", "")

    # 2. Filter out unknown items (not seen during training)
    valid_items = set(item_encoder.classes_)
    recent_item_ids = [item for item in recent_df["item_id"] if item in valid_items]

    if len(recent_item_ids) == 0:
        print("⚠️ No valid items from recent interactions found.")
        return pd.DataFrame()

    # 3. Encode and get embeddings for recent items
    recent_encoded = item_encoder.transform(recent_item_ids)
    item_embedding_layer = model.gmf_item  # or mlp_item
    recent_vectors = item_embedding_layer(tf.convert_to_tensor(recent_encoded)).numpy()

    # 4. Session vector
    session_vector = np.mean(recent_vectors, axis=0)

    # 5. Score all items seen in training
    all_item_ids = brand_model_df["item_id"].values
    all_item_encoded = item_encoder.transform(all_item_ids)
    all_vectors = item_embedding_layer(tf.convert_to_tensor(all_item_encoded)).numpy()
    scores = np.dot(all_vectors, session_vector)

    # 6. Top-K recommendations
    top_indices = np.argsort(scores)[-top_k:][::-1]
    top_items = all_item_ids[top_indices]

    return brand_model_df[brand_model_df["item_id"].isin(top_items)][["brand", "model", "item_id"]]


test_sessions = {
    "Sporty": [
        {"brand": "BMW", "model": "M6"},
        {"brand": "Porsche", "model": "911"},
        {"brand": "Jaguar", "model": "F-Type"},
        {"brand": "Mercedes-Benz", "model": "AMG GT"},
        {"brand": "Alfa Romeo", "model": "Giulia"}
    ],
    "Family_Comfort": [
        {"brand": "Audi", "model": "A6 Allroad"},
        {"brand": "Volkswagen", "model": "Passat Variant"},
        {"brand": "Mercedes-Benz", "model": "E-Class"},
        {"brand": "BMW", "model": "5 Series"},
        {"brand": "Skoda", "model": "Superb Combi"}
    ],
    "EV_Luxury": [
        {"brand": "Tesla", "model": "Model S"},
        {"brand": "BMW", "model": "iX"},
        {"brand": "Porsche", "model": "Taycan"},
        {"brand": "Audi", "model": "E-Tron"},
        {"brand": "Mercedes-Benz", "model": "EQC"}
    ],
    "Offroad_Utility": [
        {"brand": "Land Rover", "model": "Discovery"},
        {"brand": "Jeep", "model": "Grand Cherokee"},
        {"brand": "Toyota", "model": "Land Cruiser"},
        {"brand": "Audi", "model": "Q7"},
        {"brand": "Mercedes-Benz", "model": "GLS"}
    ],
    "Blended_Sport_Lux": [
        {"brand": "BMW", "model": "X6"},
        {"brand": "Audi", "model": "SQ5"},
        {"brand": "Mercedes-Benz", "model": "GLE Coupe"},
        {"brand": "Porsche", "model": "Cayenne"},
        {"brand": "Jaguar", "model": "E-Pace"}
    ]
}

# Run recommendations for each profile
all_recommendations = []
for profile, session in test_sessions.items():
    recent_df = pd.DataFrame(session)
    recs = recommend_from_recent_items_only(recent_df, model, item_encoder, brand_model_df, top_k=6)
    recs["source_profile"] = profile
    all_recommendations.append(recs)

final_recs_df = pd.concat(all_recommendations, ignore_index=True)

def pretty_print_recommendations(df):
    grouped = df.groupby("source_profile")
    for profile, group in grouped:
        print(f"\n🔹 {profile}")
        for _, row in group.iterrows():
            print(f"   - {row['brand']} {row['model']}")

pretty_print_recommendations(final_recs_df)

