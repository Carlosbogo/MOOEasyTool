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

data = pd.read_csv("heart_disease_health_indicators_BRFSS2015.csv")
X = data.drop("HeartDiseaseorAttack", axis=1)
y = data["HeartDiseaseorAttack"]

y_1 = y[y == 1]
y_used = pd.concat([y_1, y[y == 0].sample(len(y_1))])
X_used = X.iloc[y_used.index]

X_train_val, X_test, y_train_val, y_test = train_test_split(X_used, y_used, test_size=0.1, random_state=42, stratify=y_used)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=42, stratify=y_train_val)

tf.config.set_visible_devices([], 'GPU')

def fit_nn_heart_neurons_prob(c):
    print(f"C = {c}")
    
    B_1 = np.random.binomial(size=1, n=1, p=c[0]%1)
    theta_1 = int(np.floor(c[0])) - int(np.floor(c[0]))%10 + B_1*10
    
    B_2 = np.random.binomial(size=1, n=1, p=c[1]%1)
    theta_2 = int(np.floor(c[1])) - int(np.floor(c[1]))%10 + B_2*10
    
    checkpoint_path = f"./Checkpoints/model_prob_{theta_1}_{theta_2}.h5"
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
    
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(21, 1)),
      tf.keras.layers.Dense(theta_1, activation='relu'),
      tf.keras.layers.Dropout(c[2]/2000),
      tf.keras.layers.Dense(theta_2, activation='relu'),
      tf.keras.layers.Dropout(c[3]/2000),
      tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
            optimizer=tf.keras.optimizers.Adam(c[4]/100000),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        )

    with tf.device("/GPU:1"):
        model.fit(
                X_train,
                y_train,
                epochs=6,
                validation_data=(X_val, y_val)
            )
    
    #model.save_weights(checkpoint_path)

    
    y_pred_val = np.around(model.predict(X_val))
    
    return (-precision_score(y_val, y_pred_val), -recall_score(y_val, y_pred_val))