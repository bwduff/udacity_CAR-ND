import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Dropout, Cropping2D
from datetime import datetime

#PARAMETERS AND PATHS
EPOCHS = 3
BATCH_SIZE = 128
MODEL_SAVEFILE_NAME = "nvNet"
Training_Data_Paths = ['../windows_sim/collected_data/',
                       '../windows_sim/supplemental_data/alt_path_clean/',
                       '../windows_sim/supplemental_data/alt_path_clean-2/',
                       '../windows_sim/supplemental_data/alt_path_clean-3/',
                       '../windows_sim/supplemental_data/wrong_way_training/']

LEFT_CAMERA_BIAS = 0.225
RIGHT_CAMERA_BIAS = -0.225

#New write
lines, images,measurements =[],[],[]
for set in Training_Data_Paths:
    with open(set+'driving_log.csv') as csvfile:
        print("Loading file: ", set)
        reader = csv.reader(csvfile)
        for line in reader:
            for i in range(3):
                source_path = line[i]
                source_path = source_path.replace('\\','/')
                filename = source_path.split('/')[-1]
                current_path = set + 'IMG/' + filename
                image = cv2.imread(current_path)
                images.append(image)
                measurement = float(line[3])
                if( i == 1):
                    measurement += LEFT_CAMERA_BIAS
                elif(i == 2):
                    measurement += RIGHT_CAMERA_BIAS
                measurements.append(measurement)
                # print("---")
                # print("Iteration: ",i)
                # print("Path: ",current_path)
                # print(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

#MODEL DEFINITION
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.35))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#MODEL TRAINING PIPELINE
adam = optimizers.adam(lr=0.0005)
model.compile(loss='mse',optimizer=adam)
print("Beginning training")
history_object = model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=EPOCHS,batch_size=BATCH_SIZE)

modelPath = (MODEL_SAVEFILE_NAME+datetime.now().strftime("%Y%m%d-%H%M%S")+".h5")
model.save(modelPath)
print("Model saved as: ", modelPath)

print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()