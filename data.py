from __future__ import print_function

import os
import numpy as np

from skimage.io import imsave, imread

data_path = '/Users/jiangxiaohan/Desktop/Code/output/'

image_rows = 448
image_cols = 448


def create_train_data():
    train_data_path = os.path.join(data_path)
    images = os.listdir(train_data_path)
    total = len(images) // 2

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=object)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.png'
        img_id = image_name.split('.')[0]
        img = imread(os.path.join(train_data_path, image_name), as_gray=True)
        img_mask = imread(os.path.join(train_data_path, image_mask_name), as_gray=True)

        img = np.array([img])
        img_mask = np.array([img_mask])

        # print(i)
        # print(image_name)
        # print(img.shape)

        # try:
        #     imgs[i] = img
        #     imgs_mask[i] = img_mask
        #     imgs_id[i] = img_id
        # except:
        #     print(i)
        #     print(image_name)
        #     print(img.shape)
        #     continue

        imgs[i] = img
        imgs_mask[i] = img_mask
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('/Users/jiangxiaohan/Desktop/Code/imgs_train.npy', imgs)
    np.save('/Users/jiangxiaohan/Desktop/Code/imgs_mask_train.npy', imgs_mask)
    np.save('/Users/jiangxiaohan/Desktop/Code/imgs_id_train.npy', imgs_id)
    print('Saving to .npy files done.')


def load_train_data():
    data_path = '/Users/jiangxiaohan/Desktop/Code/output/'
    imgs_train = np.load(data_path+'imgs_train.npy', allow_pickle = True)
    imgs_mask_train = np.load(data_path+'imgs_mask_train.npy', allow_pickle = True)
    imgs_id = np.load(data_path+'imgs_id_train.npy', allow_pickle = True)
    return imgs_train, imgs_id, imgs_mask_train


def create_test_data():
    images = os.listdir(data_path)
    #total = len(images) // 2
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    #imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=object)

    i = 0
    print('-'*30)
    print('Creating testing images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        #image_mask_name = image_name.split('.')[0] + '_mask.png'
        img_id = image_name.split('.')[0]
        img = imread(os.path.join(data_path, image_name), as_gray=True)
    #    img_mask = imread(os.path.join(data_path, image_mask_name), as_gray=True)

        img = np.array([img])
      #  img_mask = np.array([img_mask])

        imgs[i] = img
     #   imgs_mask[i] = img_mask
        imgs_id[i] = img_id

        if i % 10 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('/Users/jiangxiaohan/Desktop/Code/imgs_test.npy', imgs)
    #np.save(masks_name, imgs_mask)
    np.save('/Users/jiangxiaohan/Desktop/Code/imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    data_path =	'/Users/jiangxiaohan/Desktop/Code/'
    imgs_test = np.load(data_path+'imgs_test.npy', allow_pickle = True)
    #imgs_mask_test = np.load(data_path+'imgs_mask_test.npy', allow_pickle = True)
    imgs_id = np.load(data_path+'imgs_id_test.npy', allow_pickle = True)
    return imgs_test, imgs_id


def create_validate_data():
    train_data_path = os.path.join(data_path, 'val')
    images = os.listdir(train_data_path)
    total = len(images) // 2

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=object)

    i = 0
    print('-'*30)
    print('Creating validation images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.png'
        img_id = image_name.split('.')[0]
        img = imread(os.path.join(train_data_path, image_name), as_gray=True)
        img_mask = imread(os.path.join(train_data_path, image_mask_name), as_gray=True)

        img = np.array([img])
        img_mask = np.array([img_mask])

        # try:
        #     imgs[i] = img
        #     imgs_mask[i] = img_mask
        #     imgs_id[i] = img_id
        # except:
        #     print(i)
        #     print(image_name)
        #     print(img.shape)
        #     continue

        imgs[i] = img
        imgs_mask[i] = img_mask
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_validate.npy', imgs)
    np.save('imgs_mask_validate.npy', imgs_mask)
    np.save('imgs_id_validate.npy', imgs_id)
    print('Saving to .npy files done.')


def load_validate_data():
    data_path =	'/Users/jiangxiaohan/Desktop/Code/'
    imgs_validate = np.load(data_path+'imgs_validate.npy', allow_pickle = True)
    imgs_mask_validate = np.load(data_path+'imgs_mask_validate.npy', allow_pickle = True)
    imgs_id = np.load(data_path+'imgs_id_validate.npy', allow_pickle = True)
    return imgs_validate, imgs_id, imgs_mask_validate

if __name__ == '__main__':
    #create_train_data()
    create_test_data()
    #create_validate_data()

