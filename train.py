import tensorflow as tf
from config import EPOCHS, BATCH_SIZE, model_dir
from prepare_data import get_datasets
from models.alexnet import AlexNet
from models.vgg16 import VGG16
from models.vgg19 import VGG19
import matplotlib.pylab as plt
from tensorflow.keras.applications.resnet50 import ResNet50
from PIL import Image, ImageEnhance
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys

def get_model():
    
    model = VGG16 ( weights=None, classes=4 )  # 训练模型input_shape, num_classes,
    model.compile ( loss='categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    metrics=['accuracy'])


    # model.compile(loss=tf.keras.losses.categorical_crossentropy,
    #               optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    #               metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    train_generator, valid_generator, test_generator, \
    train_num, valid_num, test_num = get_datasets()

    # Use command tensorboard --logdir "log" to start tensorboard
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='log')
    callback_list = [tensorboard]

    model = get_model()
    model.summary()

    #start training
    history=model.fit(train_generator,
                        epochs=EPOCHS,
                        steps_per_epoch=train_num // BATCH_SIZE,
                        validation_data=valid_generator,
                        validation_steps=valid_num // BATCH_SIZE,
                        callbacks=callback_list)

    # history = model.fit_generator(aug.flow(train_num,valid_num,batch_size=BATCH_SIZE),
    #                     validation_data=valid_generator,steps_per_epoch=train_num//batch_size,
    #                     epochs=EPOCHS,verbose=1)

    acc = history.history['accuracy']
    loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    epochs = range(1,len(acc)+1)

    plt.style.use("ggplot")
    plt.title('Accuracy and Loss')
    #plt.plot(epochs, acc, 'blue', label='Training acc')
    #plt.plot(epochs,loss,'red',label='Training loss')
    plt.plot(epochs, val_acc, 'green', label='val_acc')
    # plt.plot (epochs, val_loss, 'black', label='val_loss' )
    a = history.history["val_acc"]
    np.savetxt ( "data.txt", a, fmt='%f', delimiter=',' )
    plt.legend()
    plt.show()

