import tensorflow as tf
import sobol_seq
from pymoo.core.problem import Problem
from pymoo.core.problem import ElementwiseProblem

from models.GaussianProcess import GaussianProcess
from utils.EPAlgorithm import EP
import tensorflow as tf

def fit_nn_mnist(ds, c1, c2):
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l1(c1)),
      tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(c2)),
      tf.keras.layers.Dense(10)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        ds[0],
        epochs=6,
        validation_data=ds[1],
    )
    
    y_pred = model.predict(ds[2][0])
    
    return (tf.keras.metrics.Recall(ds[2][1], y_pred), tf.keras.metrics.Precision(ds[2][1], y_pred))