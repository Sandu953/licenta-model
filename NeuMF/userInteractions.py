import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from NeuMF import neumf


def recommend_cars_from_recent_interactions(recent_df,model, item_encoder, car_df, live_inventory, top_k):
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

    filtered_df = car_df[car_df["item_id"].isin(live_inventory)].copy()
    all_item_ids = filtered_df["item_id"].values
    all_encoded = item_encoder.transform(all_item_ids)
    all_vectors = item_embedding_layer(tf.convert_to_tensor(all_encoded)).numpy()
    scores = np.dot(all_vectors, session_vector)

    top_indices = np.argsort(scores)[-top_k:][::-1]
    top_items = all_item_ids[top_indices]
    top_scores = scores[top_indices]

    ranked_df = pd.DataFrame({"item_id": top_items, "score": top_scores})
    ranked_df = ranked_df.merge(filtered_df, on="item_id", how="left")

    return ranked_df[["Maker", "Genmodel", "item_id", "score"]]


