import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ndcg_score

# === Load and encode data ===
df = pd.read_csv('C:\\Users\\alexp\\Licenta\\Model\\NeuMF\\neumf_training_data.csv')

user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
df['user'] = user_encoder.fit_transform(df['user_id'])
df['item'] = item_encoder.fit_transform(df['item_id'])

num_users = df['user'].nunique()
num_items = df['item'].nunique()

X = df[['user', 'item']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === NeuMF model (paper spec) ===
class NeuMF(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(NeuMF, self).__init__()
        self.gmf_user = tf.keras.layers.Embedding(num_users, embedding_dim)
        self.gmf_item = tf.keras.layers.Embedding(num_items, embedding_dim)

        self.mlp_user = tf.keras.layers.Embedding(num_users, embedding_dim)
        self.mlp_item = tf.keras.layers.Embedding(num_items, embedding_dim)

        self.mlp_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu')
        ])

        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        user_input, item_input = inputs[:, 0], inputs[:, 1]

        gmf_vec = tf.multiply(self.gmf_user(user_input), self.gmf_item(item_input))

        mlp_vec = tf.concat([self.mlp_user(user_input), self.mlp_item(item_input)], axis=1)
        mlp_vec = self.mlp_layers(mlp_vec)

        final_vec = tf.concat([gmf_vec, mlp_vec], axis=1)
        return self.output_layer(final_vec)

# === Instantiate and compile model ===
model = NeuMF(num_users, num_items)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# === Training ===
X_train_tensor = tf.convert_to_tensor(X_train.values, dtype=tf.int32)
y_train_tensor = tf.convert_to_tensor(y_train.values, dtype=tf.float32)

history = model.fit(
    X_train_tensor,
    y_train_tensor,
    batch_size=256,
    epochs=30,
    validation_split=0.1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True
        )
    ]
)

# === Evaluation ===
X_test_tensor = tf.convert_to_tensor(X_test.values, dtype=tf.int32)
y_test_tensor = tf.convert_to_tensor(y_test.values, dtype=tf.float32)
loss, acc = model.evaluate(X_test_tensor, y_test_tensor)
print(f"Test Accuracy: {acc:.4f}")

# === HR@K and NDCG@K ===
y_pred_scores = model.predict(X_test_tensor).flatten()

test_results = pd.DataFrame({
    "user": X_test["user"].values,
    "item": X_test["item"].values,
    "label": y_test.values,
    "score": y_pred_scores
})

K_range = list(range(1, 11))
hr_at_k, ndcg_at_k = [], []

grouped = test_results.groupby("user")

for K in K_range:
    hr_total, ndcg_total = 0, 0

    for _, group in grouped:
        ranked = group.sort_values("score", ascending=False)
        top_k_labels = ranked["label"].values[:K]
        all_labels = ranked["label"].values
        all_scores = ranked["score"].values

        hr_total += int(1 in top_k_labels)

        if np.sum(all_labels) > 0:
            ndcg_total += ndcg_score([all_labels], [all_scores], k=K)

    hr_at_k.append(hr_total / len(grouped))
    ndcg_at_k.append(ndcg_total / len(grouped))

# === Plots ===

# === Combined Plot: HR@K, NDCG@K, and Loss Over Epochs ===
plt.figure(figsize=(18, 5))

# Subplot 1: HR@K
plt.subplot(1, 3, 1)
plt.plot(K_range, hr_at_k, marker='o', label="HR@K", color='royalblue')
plt.title("Hit Rate @ K")
plt.xlabel("K")
plt.ylabel("HR@K")
plt.grid(True)

# Subplot 2: NDCG@K
plt.subplot(1, 3, 2)
plt.plot(K_range, ndcg_at_k, marker='o', label="NDCG@K", color='orange')
plt.title("NDCG @ K")
plt.xlabel("K")
plt.ylabel("NDCG@K")
plt.grid(True)

# Subplot 3: Loss Over Epochs
plt.subplot(1, 3, 3)
plt.plot(history.history['loss'], label='Training Loss', color='red')
plt.plot(history.history['val_loss'], label='Validation Loss', color='green')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Binary Crossentropy Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("neumf_combined_plot.png")
plt.show()

