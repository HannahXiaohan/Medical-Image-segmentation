from __future__ import print_function

import configparser
import os
import tensorflow as tf
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from data import load_train_data

K.clear_session()
K.set_image_data_format('channels_last')  # TF dimension ordering in this code


img_rows = 448
img_cols = 448

smooth = 1e-5


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def similarity(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    return ((intersection) / (K.sum(K.abs(y_true_f)+ K.abs(y_pred_f)) - intersection))



def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5), conv4])
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6), conv3])
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7), conv2])
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8), conv1])
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(learning_rate = 1e-5), loss=dice_coef_loss, metrics=[similarity, dice_coef])

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)

    counter = 0
    for i in range(imgs.shape[0]):
        imgs_p[counter] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)
        counter += 1

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

def train():

    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    imgs_train, imgs_id_train, imgs_mask_train = load_train_data()
   # imgs_train, imgs_mask_train = load_train_data(imgs_train, imgs_mask)
    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    model = get_unet()
    callbacks = [tf.keras.callbacks.ModelCheckpoint('weights.h5', monitor='loss', save_best_only=True),
                 tf.keras.callbacks.EarlyStopping(monitor='dice_coef', patience=25, min_delta=0, mode='max'),
                 tf.keras.callbacks.CSVLogger('model_1_logs.csv')]

    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    # Fitting model; increment epochs appropriately (default epochs=300, default batch_size=8)
    history = model.fit(imgs_train, imgs_mask_train, batch_size=8, epochs=300, verbose=1, shuffle=False,
              callbacks=callbacks)
    weights = 'weights.h5'
    model.save_weights(weights)


    # list all data in history
    print(history.history.keys())

    plt.plot(history.history['dice_coef'])
    #plt.plot(history.history['val_dice_coef'])
    plt.title('model accuracy')
    plt.ylabel('dice_coef')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig('Accuracy.png')

    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig('Loss.png')

def train_and_predict_with_validation(imgs_train, imgs_mask_train,imgs_val, imgs_mask_val, imgs_test,imgs_mask_test, weights, pred_dir):
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    imgs_train, imgs_mask_train = load_train_data(imgs_train, imgs_mask_train, imgs_id_train)
    imgs_validate, imgs_id_validate, imgs_mask_validate = load_validate_data(imgs_train_org, imgs_mask_train, imgs_id_train)
    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Loading and preprocessing validation data...')
    print('-'*30)

    imgs_validate = preprocess(imgs_validate)
    imgs_mask_validate = preprocess(imgs_mask_validate)

    imgs_validate = imgs_validate.astype('float32')
    mean = np.mean(imgs_validate)  # mean for data centering
    std = np.std(imgs_validate)  # std for data normalization

    imgs_validate -= mean
    imgs_validate /= std

    imgs_mask_validate = imgs_mask_validate.astype('float32')
    imgs_mask_validate /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)

    imgs_test = preprocess(imgs_test)
    imgs_mask_test = preprocess(imgs_mask_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    imgs_mask_test = imgs_mask_test.astype('float32')
    imgs_mask_test /= 255.  # scale masks to [0, 1]


    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    callbacks = [tf.keras.callbacks.ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True),
                 tf.keras.callbacks.EarlyStopping(monitor='val_dice_coef', patience=25, min_delta=0, mode='max'),
                 tf.keras.callbacks.CSVLogger('model_1_logs.csv')]

    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    # Fitting model; increment epochs appropriately (default epochs=300, default batch_size=8)
    history = model.fit(imgs_train, imgs_mask_train, batch_size=8, epochs=15, verbose=1, shuffle=False,
              validation_data=(imgs_validate, imgs_mask_validate),
              callbacks=callbacks)

    model.save_weights(weights)


    # list all data in history
    print(history.history.keys())

    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('model accuracy')
    plt.ylabel('dice_coef')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig('Loss.png')

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    # model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test_pred = model.predict(imgs_test, verbose=1)

    # np.save('imgs_mask_test.npy', imgs_mask_test_pred)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test_pred, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)


    print('-' * 30)
    print('Evaluating the model on test data...')
    print('-' * 30)

    score = model.evaluate(imgs_test, imgs_mask_test, batch_size = 1)
    print('\ndice_coef: ', score[2]*100)
    print('similarity: ', score[1]*100, '\n')
    
def train_and_predict(imgs_train,imgs_mask_train,imgs_test, imgs_id_test, imgs_mask_test, pred_dir, weights):
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    #imgs_train, imgs_mask_train = load_train_data(imgs_train, imgs_mask_train, imgs_id_tes)

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Loading and preprocessing validation data...')
    print('-'*30)
    #imgs_validate, imgs_id_validate, imgs_mask_validate = load_validate_data()

    #imgs_validate = preprocess(imgs_validate)
    #imgs_mask_validate = preprocess(imgs_mask_validate)

   # imgs_validate = imgs_validate.astype('float32')
   # mean = np.mean(imgs_validate)  # mean for data centering
   # std = np.std(imgs_validate)  # std for data normalization

    #imgs_validate -= mean
    #imgs_validate /= std

    #imgs_mask_validate = imgs_mask_validate.astype('float32')
    #imgs_mask_validate /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)

    imgs_test = preprocess(imgs_test)
    imgs_mask_test = preprocess(imgs_mask_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    imgs_mask_test = imgs_mask_test.astype('float32')
    imgs_mask_test /= 255.  # scale masks to [0, 1]


    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    callbacks = [tf.keras.callbacks.ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True),
                 tf.keras.callbacks.EarlyStopping(monitor='val_dice_coef', patience=25, min_delta=0, mode='max'),
                 tf.keras.callbacks.CSVLogger('model_1_logs.csv')]

    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    # Fitting model; increment epochs appropriately (default epochs=300, default batch_size=8)
    history = model.fit(imgs_train, imgs_mask_train, batch_size=8, epochs=15, verbose=1, shuffle=False,
              validation_data=(imgs_validate, imgs_mask_validate),
              callbacks=callbacks)
    model.save_weights(weights)


    # list all data in history
    print(history.history.keys())

    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('model accuracy')
    plt.ylabel('dice_coef')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig('Loss.png')

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    # model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test_pred = model.predict(imgs_test, verbose=1)

    # np.save('imgs_mask_test.npy', imgs_mask_test_pred)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test_pred, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)


    print('-' * 30)
    print('Evaluating the model on test data...')
    print('-' * 30)

    score = model.evaluate(imgs_test, imgs_mask_test, batch_size = 1)
    print('\ndice_coef: ', score[2]*100)
    print('similarity: ', score[1]*100, '\n')


if __name__ == '__main__':
   train()
