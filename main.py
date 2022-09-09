

import numpy as np
import os
import sys, time
import json
import platform
import shutil
import cv2 as cv
import re

from uti.BRIND_aug import augment_brind, gamma_data
from uti import data_manager as dm
from uti.utls import *

IS_LINUX = True if platform.system()=='Linux' else False

def data_list_maker(img_dir='train_imgs/aug', gt_dir='train_gt/aug', dataset_name='BRIND'):


    img_base_dir =img_dir
    gt_base_dir = gt_dir
    files_idcs = []
    simple_list = False
    if simple_list:

        for full_path in os.listdir(img_base_dir):
            file_name = os.path.splitext(full_path)[0]
            files_idcs.append(
                (os.path.join(img_base_dir + '/' + file_name + '.jpg'),
                 os.path.join(gt_base_dir + '/' + file_name + '.png'),))
    # save files
    else:
        for dir_name in os.listdir(img_base_dir):
            # img_dirs = img_base_dir + '/' + dir_name
            img_dirs = img_base_dir + '/' + dir_name
            for full_path in os.listdir(img_dirs):
                file_name = os.path.splitext(full_path)[0]
                files_idcs.append(
                    (os.path.join(img_dirs + '/' + file_name + '.jpg'),
                     os.path.join(gt_base_dir + '/' + dir_name + '/' + file_name + '.png'),))
    # save files
    print(os.path.join(img_dirs + '/' + file_name + '.jpg'))
    print(os.path.join(gt_base_dir + '/' + dir_name + '/' + file_name + '.png'))
    save_path = 'train_pair.lst'
    with open(save_path, 'w') as txtfile:
        json.dump(files_idcs, txtfile)

    print("Saved in> ", save_path)

    # Check the files

    with open(save_path) as f:
        recov_data = json.load(f)
    idx = np.random.choice(200, 1)
    tmp_files = recov_data[15]
    img = cv.imread(tmp_files[0])
    gt = cv.imread(tmp_files[1])
    print(f"Image size {img.shape}, file name {tmp_files[0]}")
    print(f"GT size {gt.shape}, file name {tmp_files[1]}")



if __name__ == '__main__':

    print("Below you will find the basic operation to run: \n")
    print("Op 1. data augmentation")
    print("Op 2: Dataset list maker .txt in json")
    print("Then you are ready to run ")

    # Data augmentation

    base_dir = None
    dataset = 'BRIND'
    augment_both = True  # to augment the input and target
    augment_brind(base_dir=base_dir, augment_both=augment_both, use_all_augs=True)

    # List maker
    print("Dataset list maker is going to run in 10 sec.")
    time.sleep(10)

    dataset_name = 'BRIND'
    img_base_dir = 'train_imgs/aug'
    gt_base_dir = 'train_gt/aug'

    data_list_maker(img_dir=img_base_dir,gt_dir=gt_base_dir)


