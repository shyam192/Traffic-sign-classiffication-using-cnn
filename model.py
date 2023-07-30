import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from visualkeras import layered_view

img_data = []
img_labels = []
img_classes = 43        #types of signals(43)
cur_path = "C:/Users/shyam/OneDrive/Desktop/Python-Project-Traffic-Sign-Classification/Traffic sign classification/Dataset"

#Retrieving the images and their labels 
for i in range(img_classes):
    path = os.path.join(cur_path,'train',str(i))
    # print(path)
    train_img = os.listdir(path)

    for a in train_img:
        try:
            img = Image.open(path + '\\'+ a)
            #img.show()
            #break
            img = img.resize((30,30))              #resizing for neural network feeding
            img = np.array(img)                    #converts the image data to numerical data for computation
            sim = Image.fromarray(img)             #create an image object from a NumPy array.
            img_data.append(img)
            img_labels.append(i)
        except Exception as e:
            print("Error loading image",e)
            
#Converting lists into numpy arrays
img_data = np.array(img_data)                      #images of different classes
img_labels = np.array(img_labels)                  #folder containing images based on classification
print(img_data.shape, img_labels.shape)            #dimension of the array


#img_data.shape  = (39209, 30, 30, 3) 
#(39209) indicate number of images in training dataset.
# (30,30) indicates the size of the image(Height,Width).
# (3) indicates the number of colour channels  in each image. 


#img_label.shape = (39209,)
#one-dimensional NumPy array that stores 39209 elements..



#Splitting training and testing dataset
#test - 20% 
#tarining - 80%
X_train, X_test, y_train, y_test = train_test_split(img_data, img_labels, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# #Converting the labels into one hot encoding
y_train = to_categorical(y_train, 43)   #43 catogiries - 43length with 43 elements array (0 or 1).
y_test = to_categorical(y_test, 43)

#One hot encoding is a technique used to represent categorical data in a way that can be easily used for machine learning algorithms. 
# In one hot encoding, we convert a categorical variable with k possible categories into a binary vector of length k, where each element of the vector corresponds to a category and is either 0 or 1. 
# Specifically, we create a new binary variable for each category and assign a value of 1 to the corresponding variable and 0 to all the other variables

# #Building the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))
best_weights_callback = ModelCheckpoint(filepath='C:/Users/shyam/OneDrive/Desktop/Python-Project-Traffic-Sign-Classification/Traffic sign classification/model/best_50weights1.h5', save_weights_only=True, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
best_model_callback = ModelCheckpoint(filepath='C:/Users/shyam/OneDrive/Desktop/Python-Project-Traffic-Sign-Classification/Traffic sign classification/model/best_50model1.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', verbose=1)
layered_view(model, legend=True)


 #Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

epochs = 35
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test),callbacks=[best_weights_callback, best_model_callback, early_stopping_callback])
model.save("C:/Users/shyam/OneDrive/Desktop/Python-Project-Traffic-Sign-Classification/Traffic sign classification/model/my_model150.h5")
# #plotting graphs for accuracy 
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig("C:/Users/shyam/OneDrive/Desktop/Python-Project-Traffic-Sign-Classification/Traffic sign classification/plots/50t_accVSv_acc.png")
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig("C:/Users/shyam/OneDrive/Desktop/Python-Project-Traffic-Sign-Classification/Traffic sign classification/plots/50t_lossVSv_loss.png")
plt.show()

# #testing accuracy on test dataset
# from sklearn.metrics import accuracy_score

# y_test = pd.read_csv('Test.csv')

# labels = y_test["ClassId"].values
# imgs = y_test["Path"].values

# data=[]

# for img in imgs:
#     image = Image.open(img)
#     image = image.resize((30,30))
#     data.append(np.array(image))

# X_test=np.array(data)

# pred = model.predict_classes(X_test)

# #Accuracy with the test data
# from sklearn.metrics import accuracy_score
# print(accuracy_score(labels, pred))
