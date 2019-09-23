# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import requests
import numpy as np
import tensorflow as tf
from scipy.misc import imsave,imread
from flask import Flask, request, jsonify
from tensorflow.keras.datasets import fashion_mnist
from scipy.misc import imsave
import json


#Load the data just to select few test images
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()


#Randomly selecting few test images for prediction
for i in range(5):
    imsave('C:/Users/5030994/.spyder-py3/uploads/{}.png'.format(i),arr=x_test[i])
    
    

#Loading the model architecture and weights which we saved earlier
with open('C:\\Users\\5030994\\fashion_model.json','r') as f:
    model_json = f.read()    
model = tf.keras.models.model_from_json(model_json)    
model.load_weights('C:\\Users\\5030994\\fashion_model.h5')


#Defining the Flask Framework
app = Flask(__name__)

@app.route('/subbu/<string:img_name>',methods=['POST'])
def pred_image(img_name):
    path = 'C:\\Users\\5030994\\Model_Deploy\\test_Images\\'
    image= imread(path+img_name)
    pred = model.predict([image.reshape(1,28*28)])
    classes= ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    result = {'Predicted test label': classes[np.argmax(pred[0])]}
    return json.dumps(result)
    

app.run(port=5000, debug=False)
    
    
    
    
    