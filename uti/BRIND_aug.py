

import numpy as np
import os
import cv2 as cv
import shutil
import imutils
from scipy import ndimage

from uti.utls import image_normalization, gamma_correction, make_dirs, cv_imshow

def cv_imshow(img,title='image'):
    print(img.shape)
    cv.imshow(title,img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def scale_img(img,scl=1.):

    scaled_img = cv.resize(img, dsize=(0,0),fx=scl,fy=scl)
    return scaled_img


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv.warpAffine(image, M, (nW, nH))

def experimental_rotation(img, angle=90):
    # rot_img = imutils.rotate(img,degree) # to rotate but not ensure taken all previous image data
    # keep image shape
    rot_img = imutils.rotate_bound(img,angle) # to rotate but ensure taken all previous image data
    #
    return rot_img

def rotated_img_extractor(x=None, gt=None,img_width=None, img_height=None,i=None, two_data=False):
    #         [12, 20,   78, 90, 100, 120, 168,180, 190, 202,  268  270, 280,  290, 348 ]
    if two_data:
        if img_width==img_height:
            # for images whose sizes are the same

            if i % 90 == 0:
                adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                rot_gt = cv.warpAffine(gt, adjus_M, (img_height, img_width))
                # print("just for check 90: ", i)

            elif i % 12 == 0:
                if i == 57:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x, (100, 100), (720 - 100, 720 - 100), (0, 0, 255), (2))
                    rot_x = rot_x[100:720 - 100, 100:720 - 100, :]
                    rot_gt = rot_gt[100:720 - 100, 100:720 - 100]
                    # print("just for check 19: ", i, rot_x.shape)
                elif i == 285:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x, (75, 75), (720 - 75, 720 - 75), (0, 0, 255), (2))
                    rot_x = rot_x[75:720 - 75, 75:720 - 75, :]
                    rot_gt = rot_gt[75:720 - 75, 75:720 - 75]
                    # print("just for check 19: ", i, rot_x.shape)

                else:

                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x,(95,95),(720-95,720-95),(0,0,255),(2) )
                    rot_x = rot_x[95:720 - 95, 95:720 - 95, :]
                    rot_gt = rot_gt[95:720 - 95, 95:720 - 95]
                    # print("just for check 19: ", i, rot_x.shape)

            elif i % 23 == 0:
                if i == 161:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x,(85,85),(720-85,720-85),(0,0,255),(2) )
                    rot_x = rot_x[85:720 - 85, 85:720 - 85, :]
                    rot_gt = rot_gt[85:720 - 85, 85:720 - 85]
                    # print("just for check 23: ", i, rot_x.shape)
                else:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    rot_gt = cv.warpAffine(gt, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x, (105, 105), (720 - 105, 720 - 105), (0, 0, 255), (2))
                    rot_x = rot_x[105:720 - 105, 105:720 - 105, :]
                    rot_gt = rot_gt[105:720 - 105, 105:720 - 105]
                    # print("just for check 23:", i, rot_x.shape)

            return rot_x, rot_gt
        else:
            # # for images whose sizes are ***not*** the same *********************************
            img_size = img_width if img_width < img_height else img_height
            h,w,c = x.shape
            if i % 90 == 0:
                rot_x = ndimage.rotate(x, i)
                rot_gt = ndimage.rotate(gt, i)
                # cv.rectangle(a, (300, 100), (img_size+200, img_size+200), (0, 0, 255), (2))

            elif i  in [12,168,190,348]:
                #  [12i, 20,   78, 90, 100, 120, 168,180, 192i, 202,  268  270, 280,  290, 348 ]
                rot_x = ndimage.rotate(x, i)
                rot_gt = ndimage.rotate(gt, i)
                rh,rw,_=rot_x.shape
                mih = rh-h
                miw = rw-w
                n_w = w-miw
                n_h = h-mih
                m_w = rw//2
                m_h = rh//2
                # cv.rectangle(a, (150, 150), (img_size+30, img_size-70), (0, 0, 255), (2))
                ad=2
                rot_x = rot_x[m_h-n_h//2-ad:(m_h-n_h//2)+n_h+ad,m_w-n_w//2-ad:(m_w-n_w//2)+n_w+ad]
                rot_gt = rot_gt[m_h-n_h//2-ad:(m_h-n_h//2)+n_h+ad,m_w-n_w//2-ad:(m_w-n_w//2)+n_w+ad]
                # print("just for check 19: ", i, rot_x.shape)
            elif i in [20,200]:
                rot_x = ndimage.rotate(x, i)
                rot_gt = ndimage.rotate(gt, i)
                rh, rw, _ = rot_x.shape
                rot_size = w // 2 if w > h else h // 2
                m_w = rw // 2
                m_h = rh // 2
                ad_x = 10 if w > h else 2
                ad_y = 2 if w > h else 10
                # cv.rectangle(a, (150, 150), (img_size+30, img_size-70), (0, 0, 255), (2))
                rot_x = rot_x[m_h - rot_size // 2-ad_y:(m_h - rot_size // 2) + rot_size+ad_y,
                        m_w - rot_size // 2-ad_x:(m_w - rot_size // 2) + rot_size+ad_x]
                rot_gt = rot_gt[m_h - rot_size // 2-ad_y:(m_h - rot_size // 2) + rot_size+ad_y,
                        m_w - rot_size // 2-ad_x:(m_w - rot_size // 2) + rot_size+ad_x]
                # print("just for check 23: ", i, rot_x.shape)

            elif i in [78,100,268,280]:
                rot_x = ndimage.rotate(x, i)
                rot_gt = ndimage.rotate(gt, i)
                rh, rw, _ = rot_x.shape

                if w>h:
                    ad = 0
                    rot_w = int((rh // 2) - w * 0.03)
                    rot_h = int(((w // 4) * 3) - w * 0.03)
                    m_w = rw // 2 if w > h else rh // 2
                    m_h = rh // 2 if w > h else rw // 2
                    rot_x = rot_x[m_h - rot_h // 2 - ad:(m_h - rot_h // 2) + rot_h + ad,
                            m_w - rot_w // 2 - ad:(m_w - rot_w // 2) + rot_w + ad]
                    rot_gt = rot_gt[m_h - rot_h // 2 - ad:(m_h - rot_h // 2) + rot_h + ad,
                             m_w - rot_w // 2 - ad:(m_w - rot_w // 2) + rot_w + ad]
                else:
                    ad_x = 35
                    ad_y = 20
                    mih = rh - h
                    miw = rw - w
                    n_w = w - miw
                    n_h = h - mih
                    m_w = rw // 2
                    m_h = rh // 2
                    rot_x = rot_x[m_h - m_h // 2 - ad_y:m_h+m_h//2 + ad_y,
                            m_w-m_w//2 - ad_x:m_w+m_w//2 + ad_x]
                    rot_gt = rot_gt[m_h - m_h // 2 - ad_y:m_h+m_h//2 + ad_y,
                            m_w-m_w//2 - ad_x:m_w+m_w//2 + ad_x]

            else:
                # 290, 110
                rot_x = ndimage.rotate(x, i)
                rot_gt = ndimage.rotate(gt, i)
                rh, rw, _ = rot_x.shape

                rot_size = w // 2 if w > h else h // 2
                m_w = rw // 2
                m_h = rh // 2
                ad_x = 3 if w > h else 7
                ad_y = 7 if w > h else 3
                # cv.rectangle(a, (150, 150), (img_size+30, img_size-70), (0, 0, 255), (2))
                rot_x = rot_x[m_h - rot_size // 2 - ad_y:(m_h - rot_size // 2) + rot_size + ad_y,
                        m_w - rot_size // 2 - ad_x:(m_w - rot_size // 2) + rot_size + ad_x]
                rot_gt = rot_gt[m_h - rot_size // 2 - ad_y:(m_h - rot_size // 2) + rot_size + ad_y,
                         m_w - rot_size // 2 - ad_x:(m_w - rot_size // 2) + rot_size + ad_x]
                # print("just for check 23: ", i, rot_x.shape)
            return rot_x,rot_gt
    else:
        # For  NIR imagel but just NIR (ONE data)
        if img_height==img_width:

            if i % 90 == 0:
                adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                # print("just for check 90: ", i)

            elif i % 19 == 0:
                if i == 57:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x, (100, 100), (720 - 100, 720 - 100), (0, 0, 255), (2))
                    rot_x = rot_x[100:720 - 100, 100:720 - 100, :]
                    # print("just for check 19: ", i, rot_x.shape)
                elif i == 285:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x, (75, 75), (720 - 75, 720 - 75), (0, 0, 255), (2))
                    rot_x = rot_x[75:720 - 75, 75:720 - 75, :]
                    # print("just for check 19: ", i, rot_x.shape)

                else:

                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x,(95,95),(720-95,720-95),(0,0,255),(2) )
                    rot_x = rot_x[95:720 - 95, 95:720 - 95, :]
                    # print("just for check 19: ", i, rot_x.shape)

            elif i % 23 == 0:
                if i == 161:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x,(85,85),(720-85,720-85),(0,0,255),(2) )
                    rot_x = rot_x[85:720 - 85, 85:720 - 85, :]
                    # print("just for check 23: ", i, rot_x.shape)
                elif i==207:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x, (105, 105), (720 - 105, 720 - 105), (0, 0, 255), (2))
                    rot_x = rot_x[105:720 - 105, 105:720 - 105, :]
                    # print("just for check 23:", i, rot_x.shape)
                else:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height, img_width))
                    # cv.rectangle(rot_x, (105, 105), (720 - 105, 720 - 105), (0, 0, 255), (2))
                    rot_x = rot_x[105:720 - 105, 105:720 - 105, :]
                    # print("just for check 23:", i, rot_x.shape)
            else:
                print("Error line 221 in dataset_manager")
                return

        else:
            # when the image size are not the same
            img_size = img_width if img_width < img_height else img_height
            if i % 90 == 0:
                if i == 180:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + 250, img_width))
                    # a = np.copy(rot_x)
                    rot_x = rot_x[10:img_size - 90, 10:img_size + 110, :]

                else:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + 450, img_width))
                    # a = np.copy(rot_x)
                    rot_x = rot_x[100:img_size + 200, 300:img_size + 200, :]

            elif i % 19 == 0:
                if i == 57:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + i + 5, img_width))
                    # a = np.copy(rot_x)
                    # #                 x    y             x           y
                    # cv.rectangle(a, (275, 275), (img_size+55, img_size+55), (0, 0, 255), (2))
                    #                   y                   x
                    rot_x = rot_x[275:img_size + 55, 275:img_size + 55, :]
                elif i == 285:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + i, img_width))
                    # a = np.copy(rot_x)
                    # cv.rectangle(a, (355, 100), (img_size+205, img_size-50), (0, 0, 255), (2))
                    rot_x = rot_x[100:img_size - 50, 355:img_size + 205, :]
                elif i == 19:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + 200, img_width))
                    # a = np.copy(rot_x)
                    # cv.rectangle(a, (150, 150), (img_size+30, img_size-70), (0, 0, 255), (2))
                    rot_x = rot_x[150:img_size - 70, 150:img_size + 30, :]

                else:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + 250, img_width))
                    # a = np.copy(rot_x)
                    # cv.rectangle(a, (400, 115), (img_size+180, img_size-105), (0, 0, 255), (2))
                    rot_x = rot_x[115:img_size - 105, 400:img_size + 180, :]

            elif i % 23 == 0:
                if i == 161:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + i + 200, img_width))
                    # a = rot_x
                    # cv.rectangle(a, (95, 50), (img_size+75, img_size-170), (0, 0, 255), (2))
                    rot_x = rot_x[50:img_size - 170, 95:img_size + 75, :]
                elif i == 207:

                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + 250, img_width))
                    # a = np.copy(rot_x)
                    # cv.rectangle(a, (200, 185), (img_size + 160, img_size - 95), (0, 0, 255), (2))
                    rot_x = rot_x[185:img_size - 95, 200:img_size + 160, :]

                else:
                    adjus_M = cv.getRotationMatrix2D((img_width / 2, img_height / 2), i, 1)
                    rot_x = cv.warpAffine(x, adjus_M, (img_height + 250, img_width))
                    # a = np.copy(rot_x)
                    # cv.rectangle(a, (390, 115), (img_size+170, img_size-105), (0, 0, 255), (2))
                    rot_x = rot_x[115:img_size - 105, 390:img_size + 170, :]
        return rot_x, None

def split_data(data_dir,augment_both=True):
    # split data and copy real image to aug dir
    img_dir = data_dir[0]
    gt_dir = data_dir[1]
    img_aug_dir= os.path.join(img_dir,'aug')
    _=make_dirs(img_aug_dir)
    if augment_both and gt_dir is not None:
        gt_aug_dir = os.path.join(gt_dir,'aug')
        _ = make_dirs(gt_aug_dir)
    elif not augment_both and gt_dir is not None:
        raise NotImplementedError('In single augmentation')

    x_list = os.listdir(os.path.join(img_dir, 'real'))
    x_list.sort()
    n = len(x_list)
    if augment_both:
        gt_list = os.listdir(os.path.join(gt_dir, 'real'))
        gt_list.sort()
        n = len(gt_list) if len(x_list) == len(gt_list) else 0

    # real folder copy to aug dir
    shutil.copytree(os.path.join(img_dir, 'real'),img_aug_dir+'/real')
    if augment_both:
        shutil.copytree(os.path.join(gt_dir, 'real'), gt_aug_dir+'/real')

    # splitting up dataset
    tmp_img = cv.imread(os.path.join(
        os.path.join(img_dir, 'real'), x_list[0]))
    img_width = tmp_img.shape[1]
    img_height = tmp_img.shape[0]

    x_p1_dir = os.path.join(img_aug_dir, 'p1')
    x_p2_dir = os.path.join(img_aug_dir, 'p2')
    _= make_dirs(x_p1_dir)
    _= make_dirs(x_p2_dir)

    if augment_both:
        gt_p1_dir = os.path.join(gt_aug_dir, 'p1')
        gt_p2_dir = os.path.join(gt_aug_dir, 'p2')
        _ = make_dirs(gt_p1_dir)
        _ = make_dirs(gt_p2_dir)

    for i in range(n):
        x_tmp = cv.imread(os.path.join(
            os.path.join(img_dir, 'real'), x_list[i]))
        # compute the new size
        h,w,c = x_tmp.shape
        if w>h:
            crop_w = 400
            crop_h = 304
            width_greater= True
        else:
            crop_h = 400
            crop_w = 304
            width_greater = False
        # if width_greater:
        x_tmp1 =x_tmp[0:crop_h, 0:crop_w,:]
        x_tmp2 = x_tmp[h-crop_h:h,w-crop_w:w,:]
        cv.imwrite(os.path.join(x_p1_dir,x_list[i]), x_tmp1)
        cv.imwrite(os.path.join(x_p2_dir,x_list[i]), x_tmp2)

        if augment_both:
            gt_tmp = cv.imread(os.path.join(
            os.path.join(gt_dir, 'real'), gt_list[i]))
            gt_tmp1 = gt_tmp[0:crop_h, 0:crop_w]
            gt_tmp2 = gt_tmp[h-crop_h:h,w-crop_w:w,:]
            cv.imwrite(os.path.join(gt_p1_dir, gt_list[i]), gt_tmp1)
            cv.imwrite(os.path.join(gt_p2_dir, gt_list[i]), gt_tmp2)
            print('saved image: ', x_list[i], gt_list[i])
        else:
            print('saved image: ', x_list[i])

    print('...splitting up augmentation done!')

    if augment_both:
        print('data saved in: ', os.listdir(gt_aug_dir), 'and in',os.listdir(img_aug_dir))
        data_dirs = [img_aug_dir, gt_aug_dir]
        return data_dirs
    else:
        print('data saved in: ', os.listdir(img_aug_dir))
        data_dirs=[img_aug_dir,None]
        return data_dirs

def rotate_data(data_dir, augment_both=True):

    X_dir = data_dir[0]
    GT_dir = data_dir[1]
    x_folders = os.listdir(X_dir)
    x_folders.sort()
    if augment_both:
        gt_folders = os.listdir(GT_dir)
        gt_folders.sort()
        if not x_folders ==gt_folders:
            raise NotImplementedError('gt and x folders not match')
    # [12, 20, 78, 90, 100, 110, 168,180, 190, 200,268, 270, 280, 90, 348]
    #         [19, 46,   57, 90, 114, 138, 161,180, 207, 230,  247  270, 285,  322, 342 ]
    # degrees = [19]#, 23*2,19*3,90,19*6,23*6,23*7,180,23*9,23*10,19*13,270,19*15,23*14,19*18]
    # [12,168,190,348, 78,100,268,280,90,180,270,20,200] []
    degrees = [12, 20, 78, 90, 100, 110, 168,180, 190, 200,268, 270, 280, 90, 348]#110, 110]

    print('Folders for working: ',x_folders)
    for folder_name in x_folders:

        x_aug_list = os.listdir(os.path.join(X_dir, folder_name))
        x_aug_list.sort()
        n = len(x_aug_list)
        if augment_both:
            gt_aug_list = os.listdir(os.path.join(GT_dir, folder_name))
            gt_aug_list.sort()
            n = len(gt_aug_list) if len(x_aug_list) == len(gt_aug_list) else None

        tmp_img = cv.imread(os.path.join(X_dir,
                                         os.path.join(folder_name, x_aug_list[1])))
        img_width = tmp_img.shape[1]
        img_height = tmp_img.shape[0]

        for i in (degrees):
            if folder_name == 'p1':
                current_X_dir = X_dir + '/p1_rot_' + str(i)
            elif folder_name == 'p2':
                current_X_dir = X_dir + '/p2_rot_' + str(i)
            elif folder_name == 'real':
                current_X_dir = X_dir + '/real_rot_' + str(i)
            else:
                print('error')
                return
            if augment_both:
                if folder_name == 'p1':
                    current_GT_dir = GT_dir + '/p1_rot_' + str(i)
                elif folder_name == 'p2':
                    current_GT_dir = GT_dir + '/p2_rot_' + str(i)
                elif folder_name == 'real':
                    current_GT_dir = GT_dir + '/real_rot_' + str(i)
                else:
                    print('error')
                    return
                _ = make_dirs(current_GT_dir)

            _=make_dirs(current_X_dir)
            for j in range(n):

                tmp_x = cv.imread(os.path.join(X_dir,
                                               os.path.join(folder_name, x_aug_list[j])))
                tmp_gt = cv.imread(os.path.join(GT_dir,
                                                os.path.join(folder_name, gt_aug_list[j]))) if augment_both else None
                rot_x, rot_gt = rotated_img_extractor(tmp_x, tmp_gt, img_width, img_height, i, two_data=augment_both)

                cv.imwrite(os.path.join(current_X_dir, x_aug_list[j]), rot_x)
                tmp_imgs = rot_x
                if augment_both and rot_gt is not None:
                    cv.imwrite(os.path.join(current_GT_dir, gt_aug_list[j]), rot_gt)
                    tmp_imgs = np.concatenate((rot_x, rot_gt), axis=1)
                print('angle> ',i, 'igm size> ',rot_x.shape)
                # cv.imshow('Rotate Data', tmp_imgs)
                # cv.waitKey(150)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
                # print('finished rotating, deegres:',i, os.path.join(X_dir,
                #                                os.path.join(folder_name, x_aug_list[j])))

            print("rotation with {} degrees fullfiled folder: {} ".format(i, folder_name))

    # cv.destroyAllWindows()

    print("... rotation done in ", folder_name)

def flip_data(data_dir, augment_both=True):

    X_dir= data_dir[0]
    GT_dir = data_dir[1]
    type_aug = '_flip'
    dir_list = os.listdir(X_dir)
    dir_list.sort()
    if augment_both:
        gt_folders = os.listdir(GT_dir)
        gt_folders.sort()
        if not dir_list ==gt_folders:
            raise NotImplementedError('gt and x folders not match')

    for i in (dir_list):
        X_list = os.listdir(os.path.join(X_dir, i))
        X_list.sort()
        save_dir_x = X_dir + '/' + str(i) + type_aug
        _=make_dirs(save_dir_x)
        n = len(X_list)
        if augment_both:
            GT_list = os.listdir(os.path.join(GT_dir, i))
            GT_list.sort()
            save_dir_gt = GT_dir + '/' + str(i) + type_aug
            _= make_dirs(save_dir_gt)
            n = len(GT_list) if len(X_list) == len(GT_list) else 0
            print("Working on the dir: ", os.path.join(X_dir, i), os.path.join(GT_dir, i))
        else:
            print("Working on the dir: ", os.path.join(X_dir, i))

        for j in range(n):
            x_tmp = cv.imread(os.path.join(X_dir, os.path.join(i, X_list[j])))
            flip_x = np.fliplr(x_tmp)
            cv.imwrite(os.path.join(save_dir_x, X_list[j]), flip_x)
            tmp_imgs =flip_x
            if augment_both:
                gt_tmp = cv.imread(os.path.join(GT_dir, os.path.join(i, GT_list[j])))
                flip_gt = np.fliplr(gt_tmp)
                cv.imwrite(os.path.join(save_dir_gt, GT_list[j]), flip_gt)
                tmp_imgs = np.concatenate((flip_x, flip_gt), axis=1)

            # cv.imshow('Flipping data',tmp_imgs)
            # cv.waitKey(150)

        print("End flipping file in {}".format(os.path.join(X_dir, i)))

    # cv.destroyAllWindows()

    print("... Flipping  data augmentation finished")

def gamma_data(data_dir,augment_both=True, in_gt=False):

    X_dir = data_dir[0]
    GT_dir=data_dir[1]

    gamma30 = '_ga30'
    gamma60 = '_ga60'
    dir_list = os.listdir(X_dir)
    dir_list.sort()
    if augment_both:
        gt_folders = os.listdir(GT_dir)
        gt_folders.sort()
        if not dir_list ==gt_folders:
            raise NotImplementedError('gt and x folders not match')
    for i in (dir_list):
        X_list = os.listdir(os.path.join(X_dir, i))
        X_list.sort()
        save_dir_x30 = X_dir + '/' + str(i) + gamma30
        save_dir_x60 = X_dir + '/' + str(i) + gamma60
        _ = make_dirs(save_dir_x30)
        _ = make_dirs(save_dir_x60)
        n =len(X_list)
        if augment_both:
            GT_list = os.listdir(os.path.join(GT_dir, i))
            GT_list.sort()
            save_dir_gt30 = GT_dir + '/' + str(i) + gamma30
            save_dir_gt60 = GT_dir + '/' + str(i) + gamma60
            _=make_dirs(save_dir_gt30)
            _=make_dirs(save_dir_gt60)
            n = len(GT_list) if len(X_list) == len(GT_list) else None
            print("Working on the dir: ", os.path.join(X_dir, i), os.path.join(GT_dir, i))
        else:
            print("Working on the dir: ", os.path.join(X_dir, i))
        for j in range(n):
            x_tmp = cv.imread(os.path.join(X_dir, os.path.join(i, X_list[j])))
            if not in_gt:
                x_tmp = image_normalization(x_tmp,0,1)
                x_tmp = gamma_correction(x_tmp, 0.4040, False)
                gam30_x = gamma_correction(x_tmp, 0.3030, True)
                gam60_x = gamma_correction(x_tmp, 0.6060, True)
                gam30_x = np.uint8(image_normalization(gam30_x))
                gam60_x = np.uint8(image_normalization(gam60_x))
            else:
                gam30_x=x_tmp
                gam60_x = x_tmp
            if augment_both:
                gt_tmp = cv.imread(os.path.join(GT_dir, os.path.join(i, GT_list[j])))
            cv.imwrite(os.path.join(save_dir_x30, X_list[j]), gam30_x)
            cv.imwrite(os.path.join(save_dir_x60, X_list[j]), gam60_x)

            tmp_imgs = np.concatenate((gam30_x, gam60_x), axis=1)
            if augment_both:
                cv.imwrite(os.path.join(save_dir_gt30, GT_list[j]), gt_tmp)
                cv.imwrite(os.path.join(save_dir_gt60, GT_list[j]), gt_tmp)
                # tmp_imgs1 = np.concatenate((gam30_x, gt_tmp), axis=1)
                # tmp_imgs2 = np.concatenate((gam60_x, gt_tmp), axis=1)
                # tmp_imgs = np.concatenate((tmp_imgs2, tmp_imgs1), axis=0)
            # cv.imshow('gramma correction',tmp_imgs)
            # cv.waitKey(150)

        print("End gamma correction, file in {}".format(os.path.join(X_dir, i)))

    # cv.destroyAllWindows()

    print("... gamma correction  data augmentation finished")

def scale_data(data_dir,augment_both=True):

    X_dir = data_dir[0]
    GT_dir=data_dir[1]

    scl1 = 0.5
    scl2 = 1.5
    scl1t = '_s05'
    scl2t = '_s15'
    dir_list = os.listdir(X_dir)
    dir_list.sort()
    if augment_both:
        gt_list = os.listdir(GT_dir)
        gt_list.sort()
        if not dir_list ==gt_list:
            raise NotImplementedError('gt and x folders not match')
    for i in (dir_list):
        X_list = os.listdir(os.path.join(X_dir, i))
        X_list.sort()
        save_dir_s1 = X_dir + '/' + str(i) + scl1t
        save_dir_s2 = X_dir + '/' + str(i) + scl2t
        _ = make_dirs(save_dir_s1)
        _ = make_dirs(save_dir_s2)
        n =len(X_list)
        if augment_both:
            GT_list = os.listdir(os.path.join(GT_dir, i))
            GT_list.sort()
            save_dir_gts1 = GT_dir + '/' + str(i) + scl1t
            save_dir_gts2 = GT_dir + '/' + str(i) + scl2t
            _=make_dirs(save_dir_gts1)
            _=make_dirs(save_dir_gts2)
            n = len(GT_list) if len(X_list) == len(GT_list) else None
            print("Working on the dir: ", os.path.join(X_dir, i), os.path.join(GT_dir, i))
        else:
            print("Working on the dir: ", os.path.join(X_dir, i))
        for j in range(n):
            x_tmp = cv.imread(os.path.join(X_dir, os.path.join(i, X_list[j])))
            x_tmp1 = scale_img(x_tmp,scl1)
            x_tmp2 = scale_img(x_tmp,scl2)

            if augment_both:
                gt_tmp = cv.imread(os.path.join(GT_dir, os.path.join(i, GT_list[j])))
                gt_tmp1 = scale_img(gt_tmp, scl1)
                gt_tmp2 = scale_img(gt_tmp, scl2)
            cv.imwrite(os.path.join(save_dir_s1, X_list[j]), x_tmp1)
            cv.imwrite(os.path.join(save_dir_s2, X_list[j]), x_tmp2)
            if augment_both:
                cv.imwrite(os.path.join(save_dir_gts1, GT_list[j]), gt_tmp1)
                cv.imwrite(os.path.join(save_dir_gts2, GT_list[j]), gt_tmp2)

            # tmp_imgs = np.concatenate((x_tmp1, gt_tmp1), axis=1)
            # cv.imshow('scaling image 0.5',tmp_imgs)
            # cv.waitKey(150)

        print("Scaling finished, file in {}".format(os.path.join(X_dir, i)))

    # cv.destroyAllWindows()

    print("... Scaling augmentation has finished")

#  main tool for dataset augmentation
def augment_brind(base_dir,augment_both, use_all_augs=True):

    print('=========== Data augmentation just for 720x1280 image size ==============')
    augment_gt = True # just for augmenting ne data type (rgb or gt)
    data_dir = base_dir


    splitting_up = False#use_all_augs #use_all_type True to augment by splitting up
    rotation = use_all_augs
    flipping = use_all_augs
    correction_gamma = use_all_augs
    image_scaling = False#use_all_augs

    img_dir = os.path.join(data_dir,'train_imgs') if data_dir is not None else 'train_imgs' # path for image augmentation
    gt_dir = os.path.join(data_dir, 'train_gt') if data_dir is not None else 'train_gt'# path for gt augmentation
    if not augment_both and augment_gt:
        img_dir = gt_dir
        gt_dir = None
        print('Augmenting  just GTs')
    elif not augment_gt:
        gt_dir=None
        print('Augmenting  just for RGB imag')
    else:
        print('Augmenting RGB image and the GT')
        # return

    dataset_dirs = [img_dir, gt_dir]
    # *********** starting data augmentation *********
    if splitting_up:
        print("Image augmentation by splitting up have started!")
        dataset_dirs = split_data(data_dir=dataset_dirs,augment_both=augment_both)
        splitting_up =False
    else:
        img_aug_dir = os.path.join(img_dir,'aug')
        if os.path.exists(img_aug_dir):
            shutil.rmtree(img_aug_dir)
        _ = make_dirs(img_aug_dir)
        gt_aug_dir = None
        if augment_both and gt_dir is not None:
            gt_aug_dir = os.path.join(gt_dir, 'aug')
            if os.path.exists(gt_aug_dir):
                shutil.rmtree(gt_aug_dir)
            _ = make_dirs(gt_aug_dir)
        dataset_dirs = [img_aug_dir, gt_aug_dir]

        shutil.copytree(os.path.join(img_dir, 'imgs'), img_aug_dir + '/real')
        if augment_both:
            shutil.copytree(os.path.join(gt_dir, 'gt'), gt_aug_dir + '/real')

    if rotation:
        print("Image augmentation by rotation has started!")
        rotate_data(data_dir=dataset_dirs,augment_both=augment_both)

    if flipping:
        print("Image augmentation by flipping has started!")
        flip_data(data_dir=dataset_dirs,augment_both=augment_both)

    if correction_gamma:
        print("Image augmentation by gamma correction has started!")
        gamma_data(data_dir=dataset_dirs, augment_both=augment_both, in_gt=augment_gt)

    if image_scaling:
        print("Data augmentation by image scaling has started!")
        scale_data(data_dir=dataset_dirs, augment_both=augment_both)