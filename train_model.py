import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image

# Caminhos dos dados
train_data = './train'
test_data = './test'

# Parâmetros
input_shape = (160, 160, 3)
batch_size = 32
epochs = 40  # MobileNetV2 converge rápido!

# Carrega MobileNetV2 pré-treinado (sem as camadas finais)
base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
base_model.trainable = False  # Mantém os pesos congelados

# Modelo final
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(5, activation='softmax')  # 5 classes de frutas
])

# Compilação
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Pré-processamento de imagens
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data, target_size=(160, 160), batch_size=batch_size, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_data, target_size=(160, 160), batch_size=batch_size, class_mode='categorical')

# Callbacks para melhorar o treinamento
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
]

# Treinamento
model.fit(train_generator, validation_data=test_generator, epochs=epochs, callbacks=callbacks)

# Avaliação do modelo
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Acurácia no conjunto de teste: {test_accuracy:.4f}')

# Salvar modelo treinado
model.save('modelo_frutas_mobilenet.h5')