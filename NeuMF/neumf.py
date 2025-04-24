import tensorflow as tf

# === NeuMF model (paper spec) ===
@tf.keras.utils.register_keras_serializable()
class NeuMF(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        self.gmf_user = tf.keras.layers.Embedding(num_users, embedding_dim)
        self.gmf_item = tf.keras.layers.Embedding(num_items, embedding_dim)

        self.mlp_user = tf.keras.layers.Embedding(num_users, embedding_dim)
        self.mlp_item = tf.keras.layers.Embedding(num_items, embedding_dim)

        self.mlp_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2)
        ])

        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        user_input, item_input = inputs[:, 0], inputs[:, 1]

        gmf_vec = tf.multiply(self.gmf_user(user_input), self.gmf_item(item_input))

        mlp_vec = tf.concat([self.mlp_user(user_input), self.mlp_item(item_input)], axis=1)
        mlp_vec = self.mlp_layers(mlp_vec)

        final_vec = tf.concat([gmf_vec, mlp_vec], axis=1)
        return self.output_layer(final_vec)

    def get_config(self):
        return {
            "num_users": self.num_users,
            "num_items": self.num_items,
            "embedding_dim": self.embedding_dim,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

