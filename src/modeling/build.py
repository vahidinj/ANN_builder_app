# import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import train_test_split
# # from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


def build_ann(
    X_train: np.array,
    y_train: np.array,
    layers_units: list[int],
    output_units: int,
    hidden_activation: str,
    output_activation: str,
    loss: str,
    batch_size: int,
    epochs: int,
) -> tf.keras.models.Sequential:
    """
    Builds an ANN with a variable number of units per layer.

    Args:
        layers_units (list[int]): List of integers specifying the number of units in each layer.
        activation (str): Activation function for the hidden layers.
        output_units (int): Number of units in the output layer.
        output_activation (str): Activation function for the output layer.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs for training.

    Returns:
        tf.keras.models.Sequential: Compiled ANN model.
    """
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)))

    for units in layers_units:
        ann.add(tf.keras.layers.Dense(units=units, activation=str(hidden_activation)))

    ann.add(
        tf.keras.layers.Dense(units=output_units, activation=str(output_activation))
    )
    ann.compile(
        optimizer="adam",
        loss=str(loss),
    )

    ann.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

    model_filename = "models/ann_model.pkl"
    with open(model_filename, "wb") as file:
        pickle.dump(ann, file)


# df = pd.read_csv("data/processed/Churn.csv")
# df.head()

# num_cols = df.drop("Exited", axis=1).select_dtypes(include=["float", "int"]).columns
# cat_cols = df.drop("Exited", axis=1).select_dtypes("object").columns


# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", StandardScaler(), num_cols),
#         ("cat", OneHotEncoder(), cat_cols),
#     ]
# )


# X = df.drop("Exited", axis=1)
# y = df["Exited"]


# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.30, random_state=42
# )

# X_train = preprocessor.fit_transform(X_train)
# X_test = preprocessor.transform(X_test)


# X_train = np.array(X_train)
# X_test = np.array(X_test)
# y_train = np.array(y_train)
# y_test = np.array(y_test)


# ann = tf.keras.models.Sequential()
# ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
# ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
# ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
# ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# ann.compile(
#     optimizer="adam",
#     loss="binary_crossentropy",
#     metrics=[
#         tf.keras.metrics.BinaryAccuracy(name="accuracy"),
#         tf.keras.metrics.Precision(name="precision"),
#         tf.keras.metrics.Recall(name="recall"),
#         tf.keras.metrics.AUC(name="auc"),
#     ],
# )


# ann.fit(X_train, y_train, batch_size=32, epochs=160)
