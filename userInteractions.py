import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from NeuMF import neumf

# === Step 1: Reload item encoder ===
train_df = pd.read_csv("data/neumf_interactions_with_clean_item_ids.csv")
item_encoder = LabelEncoder()
item_encoder.fit(train_df["item_id"])

# === Step 2: Load trained model and updated car metadata ===
model = tf.keras.models.load_model("data/neumf_model.keras")
car_df = pd.read_csv("data/genmodel_metadata_with_clean_item_id.csv")

# Only keep valid trained items
valid_items = set(item_encoder.classes_)
car_df = car_df[car_df["item_id"].isin(valid_items)].copy()

# === Recommendation function ===
def recommend_from_recent_items_only(recent_df, model, item_encoder, car_df, live_intventory_set,top_k):
    recent_df["item_id"] = recent_df["brand"].str.lower().str.replace(" ", "") + "_" + recent_df["model"].str.lower().str.replace(" ", "")

    valid_items = set(item_encoder.classes_)
    recent_item_ids = [item for item in recent_df["item_id"] if item in valid_items]

    if len(recent_item_ids) == 0:
        print("⚠️ No valid models from recent interactions found.")
        return pd.DataFrame()

    recent_encoded = item_encoder.transform(recent_item_ids)
    item_embedding_layer = model.gmf_item
    recent_vectors = item_embedding_layer(tf.convert_to_tensor(recent_encoded)).numpy()
    session_vector = np.mean(recent_vectors, axis=0)

    filtered_df = car_df[car_df["item_id"].isin(live_inventory_set)].copy()
    all_item_ids = filtered_df["item_id"].values
    all_encoded = item_encoder.transform(all_item_ids)
    all_vectors = item_embedding_layer(tf.convert_to_tensor(all_encoded)).numpy()
    scores = np.dot(all_vectors, session_vector)

    top_indices = np.argsort(scores)[-top_k:][::-1]
    top_items = all_item_ids[top_indices]

    # Filter car_df using live inventory
    # car_df = car_df[car_df["item_id"].isin(live_inventory_set)].copy()

    return filtered_df[filtered_df["item_id"].isin(top_items)][["Maker", "Genmodel", "item_id"]]

# === Test session ===
test_sessions = {
    "Sporty": [
        {"brand": "BMW", "model": "6 Series"},
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
        {"brand": "Mercedes-Benz", "model": "Gls class"}
    ],
    "Blended_Sport_Lux": [
        {"brand": "BMW", "model": "X6"},
        {"brand": "Audi", "model": "SQ5"},
        {"brand": "Mercedes-Benz", "model": "GLE Coupe"},
        {"brand": "Porsche", "model": "Cayenne"},
        {"brand": "Jaguar", "model": "E-Pace"}
    ]
}

live_inventory_set = {
    "bmw_6series",
    "porsche_911",
    "jaguar_ftype",
    "mercedes-benz_amggt",
    "alfaromeo_giulia",
    "audi_a6allroad",
    "volkswagen_passatvariant",
    "mercedes-benz_eclass",
    "bmw_5series",
    "skoda_superbcombi",
    "tesla_models",
    "bmw_ix",
    "porsche_taycan",
    "audi_etron",
    "mercedes-benz_eqc",
    "landrover_discovery",
    "jeep_grandcherokee",
    "toyota_landcruiser",
    "audi_q7",
    "mercedes-benz_gls",
    "bmw_x6",
    "audi_sq5",
    "mercedes-benz_glecoupe",
    "porsche_cayenne",
    "jaguar_epace"
}

# Run all sessions
all_recs = []
for profile, session in test_sessions.items():
    recent_df = pd.DataFrame(session)
    recs = recommend_from_recent_items_only(recent_df, model, item_encoder, car_df,live_inventory_set, top_k=6)
    recs["source_profile"] = profile
    all_recs.append(recs)

results_df = pd.concat(all_recs, ignore_index=True)

def pretty_print_recommendations(df):
    grouped = df.groupby("source_profile")
    for profile, group in grouped:
        print(f"\n🔹 {profile}")
        for _, row in group.iterrows():
            print(f"   - {row['Maker']} {row['Genmodel']}")

pretty_print_recommendations(results_df)

