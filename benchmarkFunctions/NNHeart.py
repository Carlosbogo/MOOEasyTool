import tensorflow as tf
import sobol_seq
from pymoo.core.problem import Problem
from pymoo.core.problem import ElementwiseProblem
import numpy as np

from models.GaussianProcess import GaussianProcess
from utils.EPAlgorithm import EP
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("heart_disease_health_indicators_BRFSS2015.csv")
X = data.drop("HeartDiseaseorAttack", axis=1)
y = data["HeartDiseaseorAttack"]

X_used, _, y_used, _ = train_test_split(X, y, test_size=0.50, random_state=42, stratify=y)
X_train_val, X_test, y_train_val, y_test = train_test_split(X_used, y_used, test_size=0.3, random_state=42, stratify=y_used)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=42, stratify=y_train_val)

def fit_nn_heart_precision(c):
    print(f"C = {c}")
    
    checkpoint_prec_path = f"./Checkpoints/model__dropout_precision{c[0]}_{c[1]}"
    
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(21, 1)),
      tf.keras.layers.Dense(500, activation='relu'),
      tf.keras.layers.Dropout(c[0]),
      tf.keras.layers.Dense(500, activation='relu'),
      tf.keras.layers.Dense(500, activation='relu'),
      tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
            optimizer=tf.keras.optimizers.Adam(c[1]/10),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        )

    with tf.device("/GPU:1"):
        model.fit(
                X_train,
                y_train,
                epochs=6,
                validation_data=(X_val, y_val),
            )
    
    model.save_weights(checkpoint_prec_path)
    
    y_pred = np.around(model.predict(X_test))
    
    print(f"Predicted: {np.unique(y_pred, return_counts = True)}")
    
    return [-precision_score(y_test, y_pred)]*2


def fit_nn_heart_recall(c):
    print(f"C = {c}")
    
    checkpoint_recall_path = f"./Checkpoints/model__dropout_recall_{c[0]}_{c[1]}"
    
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(21, 1)),
      tf.keras.layers.Dense(500, activation='relu'),
      tf.keras.layers.Dropout(c[0]),
      tf.keras.layers.Dense(500, activation='relu'),
      tf.keras.layers.Dense(500, activation='relu'),
      tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
            optimizer=tf.keras.optimizers.Adam(c[1]/10),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        )

    with tf.device("/GPU:1"):
        model.fit(
                X_train,
                y_train,
                epochs=6,
                validation_data=(X_val, y_val),
            )
    
    y_pred = np.around(model.predict(X_test))
    
    model.save_weights(checkpoint_recall_path)
    
    print(f"Predicted: {np.unique(y_pred, return_counts = True)}")
    
    return 2*[-recall_score(y_test, y_pred)]


def fit_nn_heart(c):
    print(f"C = {c}")
    
    checkpoint_path = f"./Checkpoints/model_dropout{c[0]}_{c[1]}"
    
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(21, 1)),
      tf.keras.layers.Dense(500, activation='relu'),
      tf.keras.layers.Dropout(c[0]),
      tf.keras.layers.Dense(500, activation='relu'),
      tf.keras.layers.Dense(500, activation='relu'),
      tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
            optimizer=tf.keras.optimizers.Adam(c[1]/10),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        )

    with tf.device("/GPU:1"):
        model.fit(
                X_train,
                y_train,
                epochs=6,
                validation_data=(X_val, y_val),
            )
    
    y_pred = np.around(model.predict(X_val))
    
    model.save_weights(checkpoint_path)
    
    print(f"Predicted: {np.unique(y_pred, return_counts = True)}")
    
    return (-precision_score(y_val, y_pred), -recall_score(y_val, y_pred))


def fit_logreg_heart(c):
    logreg = LogisticRegression(C = c[0], max_iter = 300, random_state=42, solver="saga")
    
    logreg.fit(X_train, y_train)
    
    y_pred = logreg.predict(X_val)
    
    return (-precision_score(y_val, y_pred), -recall_score(y_val, y_pred))