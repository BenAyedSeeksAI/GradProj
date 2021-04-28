import librosa
import numpy as np
from ExtractFeature import ALLMFCC, MFCC, SPEC, LoadData, convertNp
from ExtractFeature import PrepareTrainTestData,  PrepareTrainTestData2
from model import Model, CNN3DModel
from Constants import PATH, N_Mfcc
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras import backend 
Audios,Labels = LoadData(PATH)
#Mfcc feature Test
#mfcc = MFCC(Audios,Nmfcc=N_Mfcc)
#mfcc,Labels = convertNp(mfcc,Labels)

#Spectogram feature test
# spec = SPEC(Audios)
# spec,Labels = convertNp(spec,Labels)

A3Dmfcc = ALLMFCC(Audios,N_Mfcc)
A3Dmfcc,Labels = convertNp(A3Dmfcc,Labels)


# x_train, x_test,y_train,y_test = PrepareTrainTestData(spec,Labels)
x_train, x_test,y_train,y_test = PrepareTrainTestData2(A3Dmfcc,Labels)

# model = Model()
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8)
# rcnn =model.fit(x_train, y_train, batch_size=16, epochs=200, validation_data=(x_test, y_test))
# rcnn =model.fit(x_train, y_train, batch_size=16, epochs=200, validation_data=(x_test, y_test),callbacks=[es])

model = CNN3DModel()
rcnn =model.fit(x_train, y_train, batch_size=16, epochs=200, validation_data=(x_test, y_test))

plt.figure()
plt.plot(rcnn.history['accuracy'])
plt.plot(rcnn.history['val_accuracy'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()