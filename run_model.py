import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
#from data import load_test_data
from skimage.io import imsave, imread

K.clear_session()
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 448
img_cols = 448

smooth = 1e-5


def create_test_data():
    from skimage.io import imread
    image_rows = 448
    image_cols = 448

    test_data_path = 'input/'
    images = os.listdir(test_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total,), dtype=object)

    i = 0
    print('-' * 30)
    print('Loading input images...')
    print('-' * 30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        img_id = image_name.split('.')[0]
        img = imread(os.path.join(test_data_path, image_name), as_gray=True)
        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 10 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    return imgs, imgs_id

def load_test_data():
    data_path =	'/Users/jiangxiaohan/Desktop/Code/'
    imgs_test = np.load(data_path+'imgs_test.npy', allow_pickle = True)
    #imgs_mask_test = np.load(data_path+'imgs_mask_test.npy', allow_pickle = True)
    imgs_id = np.load(data_path+'imgs_id_test.npy', allow_pickle = True)
    return imgs_test, imgs_id

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
    return -dice_coef(y_true, y_pred)


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
    run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)
    #tf.compat.v1.disable_eager_execution()
    model.compile(optimizer=Adam(learning_rate=1e-5), loss=dice_coef_loss, metrics=[similarity, dice_coef])

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)

    counter = 0
    for i in range(imgs.shape[0]):
        imgs_p[counter] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)
        counter += 1

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def predict():
    #gpu_options = tf.compat.v1.GPUOptions(allow_growth=True, allocator_type='BFC', per_process_gpu_memory_fraction=0.40)
    #session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
	    try:
        # Currently, memory growth needs to be the same across GPUs
		    for gpu in gpus:
			    tf.config.experimental.set_memory_growth(gpu, True)
		    logical_gpus = tf.config.list_logical_devices('GPU')
		    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
		    print(e)
    
    print('-' * 30)
    print('Loading and preprocessing input data...')
    print('-' * 30)
    imgs_test, imgs_id_test = load_test_data()

    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    mean = np.mean(imgs_test)  # mean for data centering
    std = np.std(imgs_test)  # std for data normalization
    imgs_test -= mean
    imgs_test /= std

    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    model = get_unet()
    model.load_weights('/Users/jiangxiaohan/Desktop/Code/weights.h5')

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)
    imgs_mask_test_pred = model.predict(imgs_test, batch_size=16, verbose=1)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = '/Users/jiangxiaohan/Desktop/Code/predict/'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test_pred, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)


if __name__ == '__main__':
    predict()
