import tensorflow as tf
import matplotlib.pyplot as plt
from config import *
import numpy as np
import os
from keras.models import Sequential
from models import a_convnets, alexnet, resnet, vgg16, vgg19
import time
from prepare_data import get_datasets
from models.vgg16 import VGG16
from models.a_convnets import A_ConvNets
from PIL import Image
import cv2

# 判断是否使用gpu进行训练
gpu_ok = tf.test.is_gpu_available()
print("tf_version:", tf.__version__)
print("use GPU", gpu_ok)

# train_generator, valid_generator, train_num, valid_num = get_datasets()
# 加载模型，引入模型的参数
best_model = A_ConvNets()

# input = tf.random.uniform(shape = (1,224,224,3),dtype = tf.float32)
# best_model.build(input_shape = input)
# best_model=DenseNet(img_rows=image_height,img_cols=image_width,color_type=channels,num_classes=NUM_CLASSES)
best_model.load_weights('VGG-16 pre-training.h5')

# 加载带参数的模型
# best_model = tf.keras.models.load_model('VGG-16 pre-training.h5')

# 复用原模型的前若干层
l_layer = len(best_model.layers)
print(l_layer)
best_model.summary()

# new_model = Sequential(best_model.layers[0:l_layer-6])
# new_model = Sequential(best_model.layers[:])
# new_model.trainable = False
# 构造新的输出层
# new_model.add(tf.keras.layers.Dense(units=4096, activation=tf.keras.activations.relu))
# new_model.add(tf.keras.layers.Dropout(rate=0.5))
# new_model.add(tf.keras.layers.Dense(units=4096, activation=tf.keras.activations.relu))
# new_model.add(tf.keras.layers.Dropout(rate=0.5))
# new_model.add(tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax))
# new_model.summary()
# 控制前面层不动
for i in range(l_layer - 3):
    best_model.layers[i].trainable = False
# # compile模型
# new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,beta_1 = 0.9,beta_2 = 0.99,decay=0.00001),
#                   loss=tf.keras.losses.categorical_crossentropy,
#                   metrics=['accuracy'])
# boundaries = [50, 100, 200, 300]
# values = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
# piece_wise_constant_decay = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
#     boundaries=boundaries, values=values, name=None)
best_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
time0 = time.time()

tl_train_dir = "D:/Dataset/train"
tl_valid_dir = "D:/Dataset/valid"

b_size = 256
tl_train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)
tl_train_generator = tl_train_datagen.flow_from_directory(tl_train_dir,
                                                          target_size=(88, 88),
                                                          color_mode="rgb",
                                                          batch_size=b_size,
                                                          seed=1,
                                                          shuffle=True,
                                                          class_mode="categorical")

tl_valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)
tl_valid_generator = tl_valid_datagen.flow_from_directory(tl_valid_dir,
                                                          target_size=(88, 88),
                                                          color_mode="rgb",
                                                          batch_size=b_size,
                                                          seed=1,
                                                          shuffle=True,
                                                          class_mode="categorical")
tl_train_num = tl_train_generator.samples
tl_valid_num = tl_valid_generator.samples



history = best_model.fit_generator(tl_train_generator,
                                  epochs=800,
                                  steps_per_epoch=tl_train_num // b_size,
                                  validation_data=tl_valid_generator,
                                  validation_steps=tl_valid_num // b_size)

train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']

val_loss = history.history['val_loss']
epochs = range(1, len(train_acc) + 1)

plt.figure()
plt.subplot(121)
plt.title('Accuracy')
plt.plot(epochs, train_acc, 'blue', label='Train')
plt.plot(epochs, val_acc, 'red', label='Val')
plt.xlabel('epoch')

plt.subplot(122)
plt.title('Loss')
plt.plot(epochs, train_loss, 'blue', label='Train')
plt.plot(epochs, val_loss, 'red', label='Val')
plt.xlabel('epoch')

plt.legend(loc="best")
plt.savefig("transfer.png")
plt.show()
best_model.save_weights('VGG-16 transfer.h5')