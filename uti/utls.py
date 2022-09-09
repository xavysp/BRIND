import numpy as np
import cv2 as cv
import os

from PIL import Image

def cv_imshow(img,title='image'):
    print(img.shape)
    cv.imshow(title,img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def gamma_correction(i, g,gamma=True):
    """
    0.4040 0.3030 0.6060
    :param i: image data
    :param g: gamma value
    :param gamma: if true do gamma correction if does not degamma correction
    :return:gamma corrected image if false image without gamma correction
    """
    i = np.float32(i)
    if gamma:
        img=i**g
    else:
        img=i**(1/g)
    return img

def image_normalization(img, img_min=0, img_max=255):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)
    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255
    :return: a normalized image, if max is 255 the dtype is uint8
    """
    epsilon=1e-12 # whenever an inconsistent image
    img= np.float32(img)
    img = (img-np.min(img))*(img_max-img_min)/((np.max(img)-np.min(img))+epsilon)+img_min
    return np.float32(img)

def make_dirs(paths): # make path or paths dirs
    if not os.path.exists(paths):
        os.makedirs(paths)
        print("Directories have been created: ",paths)
        return True
    else:
        print("Directories already exists: ", paths)
        return False

def edge_maker(im):
    h,w=im.shape
    map1 = np.zeros((h,w))
    map2 = np.zeros((h, w))
    for i in range(h-1):
        for j in range(w-1):
            map1[i,j]=0 if im[i,j]==im[i,j+1] else 1
    for j in range(w-1):
        for i in range(h-1):
            map2[i,j]=0 if im[i,j]==im[i+1,j] else 1

    img=map1+map2
    return img

def make_dirs(dirs_path):
    if not os.path.exists(dirs_path):
        os.makedirs(dirs_path)
        print(dirs_path," dirs created!")
    else:
        print(dirs_path," already exist!")
def fuse_2imgs(img1,img2):
    alpha=0.5
    beta = 1.-alpha
    dst = cv.addWeighted(img1,alpha,img2,beta,0.0)
    return dst

def img_fusing(base_dir=None):
    # img_dir = os.path.join(base_dir,'imgs','test','rgbr')
    img_dir = os.path.join(base_dir, 'test')
    # gt_dir = os.path.join(base_dir,'edge_maps','test','rgbr')
    gt_dir = os.path.join(base_dir,'vis_test_gt')
    list_img = os.listdir(img_dir)
    list_gt = os.listdir(gt_dir)
    save_dir = os.path.join(base_dir,'fuse_img_gt')
    os.makedirs(save_dir, exist_ok=True)

    for i, img_name in enumerate(list_img):
        tmp_img = cv.imread(os.path.join(img_dir,img_name))
        gt_name = list_gt[i]
        tmp_gt = cv.imread(os.path.join(gt_dir,gt_name))
        tmp_fused = fuse_2imgs(tmp_img,tmp_gt)
        print("in: ",os.path.join(img_dir,img_name))
        cv_imshow(img=tmp_fused,title=img_name)
        cv.imwrite(os.path.join(save_dir,img_name), tmp_fused)

    print('Done!')
def adding_2imgs(path1,path2, base_dir=None):
    img_dir = os.path.join(base_dir,'imgs', 'test', 'rgbr')
    gt1_dir = path1
    gt2_dir = path2
    list_gt = os.listdir(gt2_dir)
    save_dir = 'data/newBIPED'
    os.makedirs(save_dir,exist_ok=True)

    for i, tmp_gt_name, in enumerate(list_gt):
        old_gt = cv.imread(os.path.join(gt1_dir,tmp_gt_name),cv.IMREAD_GRAYSCALE)
        new_gt = cv.imread(os.path.join(gt2_dir,tmp_gt_name),cv.IMREAD_GRAYSCALE)
        img = cv.imread(os.path.join(img_dir,tmp_gt_name[:-3]+'jpg'))
        if old_gt is None or new_gt is None:
            print(os.path.join(gt2_dir,tmp_gt_name), 'error')
        print(tmp_gt_name, 'Then sum')
        gt = old_gt+new_gt

        # fuse1 = fuse_2imgs(img, old_gt)
        # fuse2 = fuse_2imgs(img, gt)

        # cv_imshow(title='old', img=fuse1)
        # cv_imshow(title='New', img=fuse2)
        # a = np.sum(gt[:, :, 0:gt.shape[-1]], axis=-1)
        cv.imwrite(os.path.join(save_dir,tmp_gt_name),gt)

def adding_Nimgs(dataset_dir,gt_dirs, gt_dir_list=None):
    gt_list = os.listdir(os.path.join(gt_dirs,gt_dir_list[0]))

    save_dir = os.path.join(dataset_dir,'train_edges')
    os.makedirs(save_dir,exist_ok=True)
    gt =None
    for img_name in gt_list:
        for i,gt_dir in enumerate(gt_dir_list):
            tmp_gt = cv.imread(os.path.join(gt_dirs,gt_dir, img_name),cv.IMREAD_GRAYSCALE)
            if i>0:
                gt =gt+tmp_gt
            else:
                gt=tmp_gt
        cv.imwrite(os.path.join(save_dir, img_name), gt)
        print('saved in: ',os.path.join(save_dir, img_name))


