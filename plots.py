import pandas as pd
import numpy as np
import random
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from train_neumf import NeuMF

# Load trained model
#model = load_model("neumf_model.h5", compile = True)  # change path if needed

model = load_model("neumf_model.h5", custom_objects={'NeuMF': NeuMF})

# Load and encode the dataset
df = pd.read_csv("neumf_training_data.csv")
user_encoder = {uid: idx for idx, uid in enumerate(df['user_id'].unique())}
item_encoder = {iid: idx for idx, iid in enumerate(df['item_id'].unique())}
df['user'] = df['user_id'].map(user_encoder)
df['item'] = df['item_id'].map(item_encoder)

# Build positive test set (1 pos/user) and sample negatives
users = df['user'].unique()
all_items = list(df['item'].unique())

test_data = []
for user in users:
    user_pos_items = df[(df['user'] == user) & (df['label'] == 1)]['item'].tolist()
    if not user_pos_items:
        continue
    pos_item = random.choice(user_pos_items)

    # Sample 99 negatives
    user_all_items = set(df[df['user'] == user]['item'])
    neg_candidates = list(set(all_items) - user_all_items)
    neg_items = random.sample(neg_candidates, min(99, len(neg_candidates)))

    candidates = [(user, pos_item, 1)] + [(user, neg, 0) for neg in neg_items]
    test_data.append(candidates)

# Flatten list
flat_test_data = [item for sublist in test_data for item in sublist]

# Prepare input for prediction
users_test = np.array([x[0] for x in flat_test_data])
items_test = np.array([x[1] for x in flat_test_data])
labels_test = np.array([x[2] for x in flat_test_data])

# Predict scores
scores = model.predict([users_test, items_test], verbose=0).flatten()

# Restructure by user
user_groups = {}
for (user, item, label, score) in zip(users_test, items_test, labels_test, scores):
    user_groups.setdefault(user, []).append((item, label, score))

# Evaluate HR@K and NDCG@K
K_range = list(range(1, 11))
hr_at_k = []
ndcg_at_k = []

for K in K_range:
    hr_total = 0
    ndcg_total = 0
    total_users = len(user_groups)

    for user, entries in user_groups.items():
        sorted_entries = sorted(entries, key=lambda x: x[2], reverse=True)
        top_k = sorted_entries[:K]
        labels = [x[1] for x in top_k]

        hr_total += int(1 in labels)
        ndcg_total += ndcg_score([[x[1] for x in entries]], [[x[2] for x in entries]], k=K)

    hr_at_k.append(hr_total / total_users)
    ndcg_at_k.append(ndcg_total / total_users)

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K_range, hr_at_k, marker='o')
plt.title("Hit Rate @ K")
plt.xlabel("K")
plt.ylabel("HR@K")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(K_range, ndcg_at_k, marker='o', color='orange')
plt.title("NDCG @ K")
plt.xlabel("K")
plt.ylabel("NDCG@K")
plt.grid(True)

plt.tight_layout()
plt.savefig("neumf_eval_real.png")
plt.show()
