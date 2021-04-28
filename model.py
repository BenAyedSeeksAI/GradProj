from keras.models import Model,Sequential
from keras import optimizers
from keras.layers import Input,Conv1D,Conv2D,BatchNormalization,MaxPooling1D,LSTM,Dense,Activation,Layer
from keras.layers import Dropout, Flatten, LeakyReLU,AveragePooling2D, Bidirectional
from keras.utils import to_categorical
from keras import backend
import argparse
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


def Model():
    model = Sequential()
    
    #CNNL1
    model.add(Conv1D(filters = 64,kernel_size = (3),strides=1,padding='same',input_shape=(128,1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size = (4),padding='same'))
    
    #CNNL2
    model.add(Conv1D(filters = 64, kernel_size= 3,strides=1,padding='same',))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size = (4),padding='same'))
    
    #CNNL3
    model.add(Conv1D(filters = 128, kernel_size= 3,strides=1,padding='same',))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size = (4),padding='same'))
    
    #CNNL4
    model.add(Conv1D(filters = 128, kernel_size= 3,strides=1,padding='same',))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size =(4),padding='same'))
    
    
    model.add(Flatten())
    #RNN
    #model.add(LSTM(units=32)) 
        
    #FC
    model.add(Dense(units=8,activation = 'softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=0.0005, rho=0.9, epsilon=1e-06, decay=0.0),
              metrics=['accuracy'])
    model.summary()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8)
    return model

def CNN3DModel():
    backend.set_image_data_format('channels_first')
    print(backend.image_data_format())
    model = Sequential()
    
    #First Layer
    model.add(Conv2D(filters=128 , kernel_size=(7,9),data_format='channels_first',input_shape=(3,50,127)))
    model.add(LeakyReLU())
    model.add(AveragePooling2D(pool_size=(2,2)))
    
    #Second Layer
    model.add(Conv2D(filters=64 , kernel_size=(7,9)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(AveragePooling2D(pool_size=(2,2)))
    
    
    
    #Third Layer
    #model.add(Bidirectional(LSTM(units=128)))
    model.add(Flatten())
    
    #Fully Connected
    model.add(Dense(units=8,activation = 'softmax'))
    
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.001, epsilon=1e-06, decay=0.0),
              metrics=['accuracy'])
    model.summary()
    return model
    
    
    