import librosa
import numpy as np
import glob
import os
from Constants import EMOTIONS
from ProgressBarCode import progress
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.utils import to_categorical

def MFCC(audioData, Nmfcc):
    result = []
    for data in audioData:
        mfcc = np.mean(librosa.feature.mfcc(data,n_mfcc=Nmfcc).T,axis = 0)
        result.append(mfcc)
    return result

def ALLMFCC(audioData, NMfcc):
    result = []
    for data in audioData:
        mfcc = librosa.feature.mfcc(y = data[:64745],n_mfcc = NMfcc)
        mfccdelta = librosa.feature.delta(mfcc)
        mfccdelta2 = librosa.feature.delta(mfcc, order = 2)
        result.append(np.array([mfcc,mfccdelta,mfccdelta2]))
    return result        
    

def SPEC(audioData):
    result = []
    for data in audioData:
        mel = np.mean(librosa.feature.melspectrogram(data).T, axis = 0)
        result.append(mel)
    return result


def LoadData(path):
    print("Loading Audio Files ...")
    counter = 0
    Audios,Labels = [],[]
    for file in glob.glob(path):
        counter +=1
        progress(counter,1440)
        try:
            name = os.path.basename(file)
            audio, sRate = librosa.load(file)
            Audios.append(audio)
            label= EMOTIONS[name.split('-')[2]]
            Labels.append(label)
        except ValueError:
            continue
    return Audios , Labels

def convertNp(Features, Labels):
    return np.array(Features),np.array(Labels)

def PrepareTrainTestData(Features, Labels):
    X_train, X_test,Y_train, Y_test = train_test_split(Features,Labels, test_size=0.33, random_state=42)
    X_traincnn = np.expand_dims(X_train, axis=2)
    X_testcnn = np.expand_dims(X_test, axis=2)
    le = preprocessing.LabelEncoder()
    le.fit(["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"])
    Y_train = le.transform(Y_train) 
    Y_test = le.transform(Y_test)
    Y_traincnn= to_categorical(Y_train, num_classes=8)
    Y_testcnn= to_categorical(Y_test, num_classes=8)
    return X_traincnn,X_testcnn,Y_traincnn, Y_testcnn
            
def PrepareTrainTestData2(Features, Labels):
    X_train, X_test,Y_train, Y_test = train_test_split(Features,Labels, test_size=0.33, random_state=42)
    le = preprocessing.LabelEncoder()
    le.fit(["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"])
    Y_train = le.transform(Y_train) 
    Y_test = le.transform(Y_test)
    Y_traincnn= to_categorical(Y_train, num_classes=8)
    Y_testcnn= to_categorical(Y_test, num_classes=8)
    return X_train,X_test,Y_traincnn, Y_testcnn


        
    
        