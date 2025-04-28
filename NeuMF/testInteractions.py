from userInteractions import *
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from NeuMF import neumf


# === Step 1: Reload item encoder ===
train_df = pd.read_csv("../data/neumf_interactions_with_clean_item_ids.csv")
item_encoder = LabelEncoder()
item_encoder.fit(train_df["item_id"])

# === Step 2: Load trained model and updated car metadata ===
model = tf.keras.models.load_model("../data/neumf_model.keras")
car_df = pd.read_csv("../data/genmodel_metadata_with_clean_item_id.csv")

# Only keep valid trained items
valid_items = set(item_encoder.classes_)
car_df = car_df[car_df["item_id"].isin(valid_items)].copy()


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
        {"brand": "Mercedes-Benz", "model": "E Class"},
        {"brand": "BMW", "model": "5 Series"},
        {"brand": "Skoda", "model": "Superb"},

    ],
    "EV_Luxury": [
        {"brand": "Tesla", "model": "Model S"},
        {"brand": "BMW", "model": "i3"},
        {"brand": "BMW", "model": "i3"},
        {"brand": "volkswagen", "model": "E-golf"},
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
    ],
    "Economy":[
        {"brand": "BMW", "model": "3 series"},
        {"brand": "BMW", "model": "3 series"},
        {"brand": "BMW", "model": "3 series"},
        {"brand": "BMW", "model": "3 series"},
        {"brand": "BMW", "model": "3 series"},
        {"brand": "BMW", "model": "3 series"},
        {"brand": "BMW", "model": "3 series"},
        {"brand": "BMW", "model": "3 series"},
        {"brand": "BMW", "model": "3 series"},
        {"brand": "BMW", "model": "3 series"},
        {"brand": "BMW", "model": "3 series"},
        {"brand": "BMW", "model": "3 series"},
        {"brand": "BMW", "model": "3 series"},
        {"brand": "BMW", "model": "3 series"},
        {"brand": "BMW", "model": "3 series"},
        {"brand": "Audi", "model": "A4"},
        {"brand": "Audi", "model": "A2"},
        {"brand": "Audi", "model": "A3"},
        {"brand": "Dacia", "model": "Duster"},
        {"brand": "Dacia", "model": "Sandero"}

    ]
}

live_inventory_set = {
    "bmw_6series",
    "bmw_3series",
    "bmw_3series",
    "bmw_i3",
    "bmw_3series",
    "dacia_duster",
    "dacia_duster",
    "dacia_duster",
    "dacia_sandero",
    "dacia_sandero",
    "dacia_sandero",
    "audi_a1",
    "audi_a2",
    "audi_a4",
    "audi_a5",
    "audi_a3",
    "porsche_911",
    "jaguar_ftype",
    "mercedes-benz_amggt",
     "alfaromeo_giulia",
    "audi_a6allroad",
    "volkswagen_passatvariant",
    "volkswagen_e-golf",
    "mercedes-benz_eclass",
    "bmw_5series",
    "skoda_superb",
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
    recs = recommend_cars_from_recent_interactions(recent_df,model, item_encoder, car_df, live_inventory_set, top_k=10)
    recs["source_profile"] = profile
    all_recs.append(recs)

results_df = pd.concat(all_recs, ignore_index=True)

def pretty_print_recommendations(df):
    grouped = df.groupby("source_profile")
    for profile, group in grouped:
        print(f"\n🔹 {profile}")
        for _, row in group.iterrows():
            print(f"   - {row['Maker']} {row['Genmodel']} {row['score']}")

pretty_print_recommendations(results_df)
