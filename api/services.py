from api.repo import *
import pandas as pd
import tensorflow as tf
import numpy as np
from NeuMF.load_model import model, item_encoder, car_df


def recommend_cars_for_user(user_id: int):
    recent_df = UserRepository.get_recent_interactions(user_id)
    live_inventory = InventoryRepository.get_live_inventory()

    recommended_df = recommend_cars_from_recent_interactions(
        recent_df,model, item_encoder, car_df, live_inventory, top_k=6
    )

    # Map into CarRecommendation list
    recommended_cars = []
    for _, row in recommended_df.iterrows():
        recommended_cars.append({
            "maker": row["Maker"],
            "genmodel": row["Genmodel"],
            "item_id": row["item_id"],
            "score": float(row["score"])
        })
    return recommended_cars
