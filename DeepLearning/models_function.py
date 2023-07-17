import tensorflow as tf
import keras
from keras import Sequential, layers, optimizers

def get_relu_softmax():
    model = Sequential([
        layers.Dense(512, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def get_relu_softmax2(learning_rate):
    model = Sequential([
        layers.Dense(512, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    of = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=of, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def large_model_creation(model_kind):
    if(model_kind == 1):
        model = keras.Sequential([layers.Dense(10, activation='softmax')])
    elif(model_kind == 2):
        model = keras.Sequential([
            layers.Dense(96, activation='relu'),
            layers.Dense(96, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
    
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model