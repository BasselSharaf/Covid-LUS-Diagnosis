import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
import math
from sklearn import preprocessing
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

my_data = pd.read_csv('../data/Five_Frames_Per_patient.csv')
patients_data=pd.read_csv('../data/videos_data.csv')
data_array = my_data.to_numpy()
patients_array=patients_data["FileName"].to_numpy()
images_path = '../data/Five_Frames_Per_patient/'
no_of_test_patients=math.floor(patients_array.size*0.2)
images_data = []
models = []

df=my_data
data_array=df.to_numpy()
for i in data_array:
    img = cv2.imread(images_path + i[0] + '.' + i[2], cv2.IMREAD_GRAYSCALE)
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img_2d = img.reshape(256 ,256,1)
    images_data.append(img_2d)

X=np.asarray(images_data)
y = np.asarray(my_data['Label'])
number = preprocessing.LabelEncoder()
y=number.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

#reshape data to fit model
y_train=to_categorical(y_train,3)
y_test=to_categorical(y_test,3)

#create model
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=4, activation='relu', input_shape=(256,256,1)))
model.add(Conv2D(32, kernel_size=4, activation='relu'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit(X_train, y_train, epochs=3)