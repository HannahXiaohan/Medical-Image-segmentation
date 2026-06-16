## Creating Train / Val / Test folders
import os
import numpy as np
import shutil
import random
import glob
import numpy as np
import fnmatch

root_dir = '/Users/jiangxiaohan/Desktop/Code/Data/' # data root path
set_dir = ['train', 'test', 'val']

val_ratio = 0.15
test_ratio = 0.05

for cls in set_dir:
    os.makedirs(root_dir + cls,exist_ok=True)
    os.makedirs(root_dir + cls,exist_ok=True)
    os.makedirs(root_dir + cls,exist_ok=True)

# Creating partitions of the data after shuffeling
src = root_dir + cls # Folder to copy images from

allFileNames = os.listdir(root_dir)

i = 0
fullList = []
for fileName in allFileNames:
	if fileName.endswith(".png"):
		if 'mask' in fileName:
			continue	
		#print(fileName)
		fullList.append(fileName)

np.random.shuffle(fullList)

train_FileNames, val_FileNames, test_FileNames = np.split(np.array(fullList),
                                                          [int(len(fullList)* (1 - (val_ratio + test_ratio))), 
                                                           int(len(fullList)* (1 - test_ratio))])	

train_FileNames = [name for name in train_FileNames.tolist()]
val_FileNames = [name for name in val_FileNames.tolist()]
test_FileNames = [name for name in test_FileNames.tolist()]

print('--------Train File Names--------')
print(train_FileNames)

print('Total images: ', len(allFileNames))
print('Training: ', len(train_FileNames))
print('Validation: ', len(val_FileNames))
print('Testing: ', len(test_FileNames))

# Copy-pasting images
for name in train_FileNames:
    shutil.copy(root_dir + name, root_dir + 'train/')
    mask_img_name = name.split('.')[0] + '_mask.png'
    shutil.copy(root_dir + mask_img_name, root_dir + 'train/')

for name in val_FileNames:
    shutil.copy(root_dir + name, root_dir +'val/')
    mask_img_name = name.split('.')[0] + '_mask.png'
    shutil.copy(root_dir + mask_img_name, root_dir + 'val/')

for name in test_FileNames:
    shutil.copy(root_dir + name, root_dir +'test/')
    mask_img_name = name.split('.')[0] + '_mask.png'
    shutil.copy(root_dir + mask_img_name, root_dir + 'test/')

count = len(fnmatch.filter(os.listdir(root_dir +'train'), '*.*'))
print('Train Count:', count)

count = len(fnmatch.filter(os.listdir(root_dir +'val'), '*.*'))
print('Val Count:', count)

count = len(fnmatch.filter(os.listdir(root_dir +'test'), '*.*'))
print('Test Count:', count)

print("-----Done-----")

