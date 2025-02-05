import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Classificação de imagens
model = tf.keras.models.load_model('modelo_frutas_mobilenet.h5')
input_size = (160, 160)
input_folder = './imagens_para_classificar'
output_folder = './output'
os.makedirs(output_folder, exist_ok=True)

# Classes de frutas
target_classes = ['maça', 'banana', 'uva', 'manga', 'abacaxi']

for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    
    img = image.load_img(img_path, target_size=input_size)
    img_array = image.img_to_array(img) / 255.0  # Normalização
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)
    
    label = target_classes[class_index] if confidence > 0.5 else 'fruta não identificada'
    
    img_cv = cv2.imread(img_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_position = (20, 40)
    font_scale = 1
    color = (0, 255, 0) if label != 'fruta não identificada' else (0, 0, 255)
    thickness = 2
    
    cv2.putText(img_cv, label, text_position, font, font_scale, color, thickness)
    output_path = os.path.join(output_folder, img_name)
    cv2.imwrite(output_path, img_cv)

print(f'Classificação concluída! Imagens salvas em "{output_folder}".')