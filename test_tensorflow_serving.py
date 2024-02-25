import numpy as np
import os
import subprocess
import cv2
import json
import requests
from tqdm import tqdm
from detect import detect
from PIL import Image


model_image_size = (608, 608)

def predict(frame):

    #image = Image.fromarray(frame)  # Convertir el frame de OpenCV a formato de imagen de PIL
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    #image.show()

    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)


    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.


    image_data = np.expand_dims(image_data, 0)  # Agregar dimensión de batch


    data = json.dumps({"signature_name": "serving_default", 
                       "instances": image_data.tolist()})

    HEADERS = {'content-type': 'application/json'}
    MODEL1_API_URL = 'http://localhost:8605/v1/models/yolo_model:predict'


    json_response = requests.post(MODEL1_API_URL, data=data, headers=HEADERS)
    predictions = json.loads(json_response.text)['predictions']

    return detect(predictions, image)
