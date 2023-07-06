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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.calc_pareto import get_pareto_undominated_by

class Custom_Loss(tf.keras.losses.Loss):
    def __init__(self, c):
        super().__init__()
        self.c = c
        
    def call(self, y_true, y_pred):        
        mae = tf.keras.losses.mean_absolute_error(y_true, y_pred)
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
        return mae + mse

data = pd.read_csv("kc_house_data.csv")
X = data.drop(["price", "id", "date"], axis=1)
y = data["price"]

# X_used, _, y_used, _ = train_test_split(X, y, test_size=0.9, random_state=42)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=42)

# Normalized y
# norm_y = (y - y.mean())/y.std()
# X_train_val, X_test, y_train_val, y_test = train_test_split(X, norm_y, test_size=0.1, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=42)

tf.config.set_visible_devices([], 'GPU')

def fit_nn_houses_regression_prob(c):
    print(f"C = {c}")
    
    B_1 = np.random.binomial(size=1, n=1, p=c[0]%1)
    theta_1 = int(np.floor(c[0]) + B_1)
    
    B_2 = np.random.binomial(size=1, n=1, p=c[1]%1)
    theta_2 = int(np.floor(c[1]) + B_2)
    
    print(f"C = [{theta_1}, {theta_2}]")
    #checkpoint_path = f"./Checkpoints/reg_model_{int(np.around(c[0]))}_{int(np.around(c[1]))}"
    
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(18, 1)),
      tf.keras.layers.Dense(theta_1, activation='relu'),
      tf.keras.layers.Dense(theta_2, activation='relu'),
      tf.keras.layers.Dense(1, activation='relu')
    ])

    model.compile(
        loss=Custom_Loss(c[0]),
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['mae', 'mse'],
    )


    model.fit(
        X_train,
        y_train,
        epochs=6,
        validation_data=(X_val, y_val)
    )

    y_pred = model.predict(X_val)
    
    errors = np.array([mean_absolute_error(y_val, y_pred)/y.mean(), mean_squared_error(y_val, y_pred)/(y.mean()**2)])
    
    errors = np.asarray(errors).astype('float32')

    return errors