import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
from PIL import Image
from glob import glob
from tkinter import *


model = load_model("C:/Users/shyam/OneDrive/Desktop/Python-Project-Traffic-Sign-Classification/Traffic sign classification/model/best_model1.h5")
classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',      
            3:'Speed limit (50km/h)',       
            4:'Speed limit (60km/h)',      
            5:'Speed limit (70km/h)',    
            6:'Speed limit (80km/h)',      
            7:'End of speed limit (80km/h)',     
            8:'Speed limit (100km/h)',    
            9:'Speed limit (120km/h)',     
           10:'No passing',   
           11:'No passing veh over 3.5 tons',     
           12:'Right-of-way at intersection',     
           13:'Priority road',    
           14:'Yield',     
           15:'Stop',       
           16:'No vehicles',       
           17:'Veh > 3.5 tons prohibited',       
           18:'No entry',       
           19:'General caution',     
           20:'Dangerous curve left',      
           21:'Dangerous curve right',   
           22:'Double curve',      
           23:'Bumpy road',     
           24:'Slippery road',       
           25:'Road narrows on the right',  
           26:'Road work',    
           27:'Traffic signals',      
           28:'Pedestrians',     
           29:'Children crossing',     
           30:'Bicycles crossing',       
           31:'Beware of ice/snow',
           32:'Wild animals crossing',      
           33:'End speed + passing limits',      
           34:'Turn right ahead',     
           35:'Turn left ahead',       
           36:'Ahead only',      
           37:'Go straight or right',      
           38:'Go straight or left',      
           39:'Keep right',     
           40:'Keep left',      
           41:'Roundabout mandatory',     
           42:'End of no passing',      
           43:'End no passing veh > 3.5 tons' }


# Capture an image from the laptop camera
cap = cv2.VideoCapture(0)

# Capture a single frame from the camera
ret, frame = cap.read()

# Save the captured frame to a file
cv2.imwrite('C:/Users/shyam/OneDrive/Desktop/Python-Project-Traffic-Sign-Classification/Traffic sign classification/captured_image.jpg', frame)

# Resize the captured frame to 30x30 pixels
resized_frame = cv2.resize(frame, (30, 30))

# Convert the resized frame to a NumPy array of shape (1, 30, 30, 3)
image = np.expand_dims(resized_frame, axis=0)

# Display the saved image
saved_image = cv2.imread('C:/Users/shyam/OneDrive/Desktop/Python-Project-Traffic-Sign-Classification/Traffic sign classification/captured_image.jpg')
cv2.imshow('saved_image', saved_image)
cv2.waitKey(0)

# Release the camera and close the windows
cap.release()
cv2.destroyAllWindows()

# Print the shape of the image array
print(image.shape)


# Predict the traffic signal from the resized image
pred = np.argmax(model.predict(image), axis=-1)[0]
signal_name = classes[pred+1]
print(signal_name)
