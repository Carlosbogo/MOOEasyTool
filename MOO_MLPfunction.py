from tensorflow.keras.layers import Dropout, Flatten, Input, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from keras.utils import np_utils
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

cifar10 = tf.keras.datasets.cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train/255.
X_test = X_test/255.
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def MLP_model(l2_reg=0, dropout = 0, img_width=32, img_height=32, img_channels=3, num_classes=10, n_hidden=2):
    
    input_image = Input(shape=(img_width,img_height,img_channels))
    x = Flatten()(input_image)
    
    for layer in reversed(range(n_hidden)):
        x = Dense(int((img_height*img_width*img_channels+num_classes)*(layer+1)/(n_hidden+1)), 
            activation='relu', 
            kernel_regularizer=l2(l2_reg))(x)
        if dropout>0:
            x=Dropout(dropout)(x)
        
    out= Dense(num_classes, activation='softmax',kernel_regularizer=l2(l2_reg))(x)
    model = Model(inputs = input_image, outputs = out)
    
    return model

def trainModel(model, learning_rate, X_train=X_train, y_train=y_train, epochs=2, batch_size=128):

    model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=learning_rate), metrics=['accuracy'])

    history=model.fit(
        x = X_train, 
        y = y_train, 
        batch_size = batch_size,
        validation_split=0.2,
        epochs = epochs
    )
    
    return history

def evalMLPCIFAR1(x):
    """
        input: x:
            x[0]=learning_rate (log)
        output: y:
            y[0]=val_accuracy
            y[1]=overfitting error=accuracy-val_accuracy
    """

    # model = MLP_model(l2_reg=0)    
    print(10**x)

    model = MLP_model(l2_reg=0, dropout=0)    
    history = trainModel(model, 10**x[0])

    val_acc = np.max(history.history['val_accuracy'])
    overfitting = np.max(history.history['accuracy']) - np.max(history.history['val_accuracy'])

    return np.array([-val_acc, overfitting])

def evalMLPCIFAR2(x):
    """
        input: x:
            x[0]=learning_rate (log)
            x[1]=l2_reg (log)
        output: y:
            y[0]=val_accuracy
            y[1]=overfitting error=accuracy-val_accuracy
    """

    # model = MLP_model(l2_reg=0)    
    print(10**x)

    model = MLP_model(l2_reg=10**x[1], dropout=0)    
    history = trainModel(model, 10**x[0])

    val_acc = np.max(history.history['val_accuracy'])
    overfitting = np.max(history.history['accuracy']) - np.max(history.history['val_accuracy'])

    return np.array([-val_acc, overfitting])

def evalMLPCIFAR3(x):
    """
        input: x:
            x[0]=learning_rate (log)
            x[1]=l2_reg (log)
            x[2]=dropout
        output: y:
            y[0]=val_accuracy
            y[1]=overfitting error=accuracy-val_accuracy
    """

    # model = MLP_model(l2_reg=0)    
    print(10**x)

    model = MLP_model(l2_reg=10**x[1], dropout=x[2])    
    history = trainModel(model, 10**x[0])

    val_acc = np.max(history.history['val_accuracy'])
    overfitting = np.max(history.history['accuracy']) - np.max(history.history['val_accuracy'])

    return np.array([-val_acc, overfitting])


