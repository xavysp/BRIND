"""
A draft code
"""
import numpy as np
import os
import sys
import cv2
import scipy.io as sio
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw
from collections import defaultdict
from uti import utls


def rename_files(rgbn_list,rgb_list=None, rgbr_list=None, nir_list=None, op=None):
    # get files name from a folder

    #
    if op==1 or op==None:

        for i in range(len(rgbn_list)):

            n= len(rgbn_list[i])
            if n>23:

                info = rgbn_list[i][-8:-3] #-7:-3 for less than 99 -8:-3 greater than 99
                num = info[1:-1]
                is_num = is_number(num)
                if is_num:
                    num = int(num)
                    if (num>99 and num<320) and (info[0]=='-' and info[-1]=='.'): # (num>0 and num<100)

                        name = rgbn_list[i][:-7]  # -6 num<100 and -7 num>99
                        r_name = name + str(num-1)+'.raw'
                        os.rename('/opt/2kiss/rgbn_raw/'+rgbn_list[i], '/opt/2kiss/rgbn_raw/'+r_name)
                        print("'/opt/2kiss/rgbn_raw/",rgbn_list[i], "to: /opt/2kiss/rgbn_raw/",r_name, "\n")

    if (op==2) and (rgb_list != None):
        # To exchange names
        for i in range(len(rgb_list)):

            inf_v = rgb_list[i]  # -7:-3 for less than 99 -8:-3 greater than 99
            num = rgb_list[i][2:3] # 4 TE>9
            is_num = is_number(num)
            if is_num and int(num)>0:
                num = int(num)
                if num>3:  # (num>0 and num<100)

                    name = rgb_list[i][3:]  # 4 TE >9
                    r_name = 'TE' + str(3) + name
                    # os.rename('/opt/2kiss/VIS/' + rgb_list[i], '/opt/2kiss/VIS/' + r_name)
                    print("'/opt/2kiss/VIS/", rgb_list[i], "to: /opt/2kiss/VIS/", r_name, "\n")

    if (op==3) and (rgb_list != None):
        print("op 3: check if two directories have the same name with the respective files \n")
        n = np.zeros((len(rgb_list),1))
        j=0
        for i in range(len(rgb_list)):
            if rgbn_list[i]==rgb_list[i]:
                pass
            else:

                n[j,0]= i
                j +=1
        print(n)
        print(" sum: ", np.sum(n))

    if (op == 4) and (rgb_list != None and rgbn_list != None):

        # changing name to dataset
        n = np.zeros((len(rgb_list), 1))
        j = 0
        for i in range(len(rgb_list)):
            if rgbn_list[i]==rgb_list[i]:
                if i<9:
                    rgbn_name = "RGBN_00"+str(i+1)+".raw"
                    rgbn_dir = "/opt/2kiss/raw_rgbn/"
                    rgb_name = "RGB_00" + str(i+1) + ".raw"
                    rgb_dir = "/opt/2kiss/raw_rgb/"
                    os.rename(rgbn_dir + rgbn_list[i],  rgbn_dir+ rgbn_name)
                    os.rename(rgb_dir + rgb_list[i], rgb_dir + rgb_name)
                    print("name changed: ",rgbn_dir + rgbn_list[i]," ",rgbn_dir+ rgbn_name)
                    print("name changed: ",rgb_dir + rgb_list[i]," ",rgb_dir+ rgb_name)


                if i<99 and i>=9:
                    rgbn_name = "RGBN_0" + str(i+1) + ".raw"
                    rgbn_dir = "/opt/2kiss/raw_rgbn/"
                    rgb_name = "RGB_0" + str(i+1) + ".raw"
                    rgb_dir = "/opt/2kiss/raw_rgb/"
                    os.rename(rgbn_dir + rgbn_list[i],  rgbn_dir+ rgbn_name)
                    os.rename(rgb_dir + rgb_list[i], rgb_dir + rgb_name)
                    print("name changed: ",rgbn_dir + rgbn_list[i]," ",rgbn_dir+ rgbn_name)
                    print("name changed: ",rgb_dir + rgb_list[i]," ",rgb_dir+ rgb_name)

                if i>= 99:
                    rgbn_name = "RGBN_" + str(i+1) + ".raw"
                    rgbn_dir = "/opt/2kiss/raw_rgbn/"
                    rgb_name = "RGB_" + str(i+1) + ".raw"
                    rgb_dir = "/opt/2kiss/raw_rgb/"
                    os.rename(rgbn_dir + rgbn_list[i],  rgbn_dir+ rgbn_name)
                    os.rename(rgb_dir + rgb_list[i], rgb_dir + rgb_name)
                    print("name changed: ", rgbn_dir + rgbn_list[i], " ", rgbn_dir + rgbn_name)
                    print("name changed: ", rgb_dir + rgb_list[i], " ", rgb_dir + rgb_name)




            else:

                n[j,0]= i
                j +=1
        # print(n)
        # print(" sum: ", np.sum(n))

# *******************************operation 5 *********************************
    if (op == 5 and rgbr_list!=None) and (rgb_list != None and rgbn_list != None):

        # changing name to dataset (just image in jpg and png format)
        print("op 5: check if four directories have the same name with the respective files \n")
        rgbn_dir = "/opt/2kiss/rgbn/"
        rgbr_dir = "/opt/2kiss/rgbr/"
        rgb_dir = "/opt/2kiss/rgb/"
        nir_dir = "/opt/2kiss/nir/"
        n = np.zeros((len(rgb_list), 1))
        j = 0
        for i in range(len(rgb_list)):
            if rgbr_list[i]==rgb_list[i]:
                if i<9:
                    rgbn_name = "RGBN_00"+str(i+1)+".jpg"
                    rgb_name = "RGB_00" + str(i+1) + ".jpg"
                    rgbr_name = "RGB_00" + str(i + 1) + ".jpg"
                    nir_name = "NIR_00" + str(i + 1) + ".png"
                    os.rename(rgbn_dir + rgbn_list[i],  rgbn_dir+ rgbn_name)
                    os.rename(rgb_dir + rgb_list[i], rgb_dir + rgb_name)
                    os.rename(rgbr_dir + rgbr_list[i], rgbr_dir + rgbr_name)
                    os.rename(nir_dir + nir_list[i], nir_dir + nir_name)
                    print("rgbn changed: ",rgbn_dir + rgbn_list[i]," ",rgbn_dir+ rgbn_name)
                    print("rgb changed: ",rgb_dir + rgb_list[i]," ",rgb_dir+ rgb_name)
                    print("rgbr changed: ", rgbr_dir + rgbr_list[i], " ", rgbr_dir + rgbr_name)
                    print("nir changed: ", nir_dir + nir_list[i], " ", nir_dir + nir_name)


                if i<99 and i>=9:
                    rgbn_name = "RGBN_0"+str(i+1)+".jpg"
                    rgb_name = "RGB_0" + str(i+1) + ".jpg"
                    rgbr_name = "RGB_0" + str(i + 1) + ".jpg"
                    nir_name = "NIR_0" + str(i + 1) + ".png"
                    os.rename(rgbn_dir + rgbn_list[i],  rgbn_dir+ rgbn_name)
                    os.rename(rgb_dir + rgb_list[i], rgb_dir + rgb_name)
                    os.rename(rgbr_dir + rgbr_list[i], rgbr_dir + rgbr_name)
                    os.rename(nir_dir + nir_list[i], nir_dir + nir_name)
                    print("rgbn changed: ",rgbn_dir + rgbn_list[i]," ",rgbn_dir+ rgbn_name)
                    print("rgb changed: ",rgb_dir + rgb_list[i]," ",rgb_dir+ rgb_name)
                    print("rgbr changed: ", rgbr_dir + rgbr_list[i], " ", rgbr_dir + rgbr_name)
                    print("nir changed: ", nir_dir + nir_list[i], " ", nir_dir + nir_name)

                if i>= 99:
                    rgbn_name = "RGBN_"+str(i+1)+".jpg"
                    rgb_name = "RGB_" + str(i+1) + ".jpg"
                    rgbr_name = "RGB_" + str(i + 1) + ".jpg"
                    nir_name = "NIR_" + str(i + 1) + ".png"
                    os.rename(rgbn_dir + rgbn_list[i],  rgbn_dir+ rgbn_name)
                    os.rename(rgb_dir + rgb_list[i], rgb_dir + rgb_name)
                    os.rename(rgbr_dir + rgbr_list[i], rgbr_dir + rgbr_name)
                    os.rename(nir_dir + nir_list[i], nir_dir + nir_name)
                    print("rgbn changed: ",rgbn_dir + rgbn_list[i]," ",rgbn_dir+ rgbn_name)
                    print("rgb changed: ",rgb_dir + rgb_list[i]," ",rgb_dir+ rgb_name)
                    print("rgbr changed: ", rgbr_dir + rgbr_list[i], " ", rgbr_dir + rgbr_name)
                    print("nir changed: ", nir_dir + nir_list[i], " ", nir_dir + nir_name)



            else:

                n[j,0]= i
                j +=1

    if (op == 6 and rgbr_list != None) and (rgb_list != None and rgbn_list != None):

        # changing name to dataset (just image in jpg and png format)
        print("op 5: check if four directories have the same name with the respective files \n")
        # rgbn_dir = "/opt/2kiss/un_rgbn/"
        # rgbr_dir = "/opt/2kiss/un_rgbr/"
        # rgb_dir = "/opt/2kiss/un_rgb/"
        rgbn_dir =  '/home/xsoria/matlabprojects/RGBrestoration/datasetRAW/un_rgbnc/'  #'/opt/2kiss/raw_rgbn'
        rgb_dir =  '/home/xsoria/matlabprojects/RGBrestoration/datasetRAW/rgbnc/'  #'/opt/2kiss/raw_rgb'
        rgbr_dir = '/home/xsoria/matlabprojects/RGBrestoration/datasetRAW/un_rgbr/'
        n = np.zeros((len(rgb_list), 1))
        j = 0
        for i in range(len(rgb_list)):
            if rgbr_list[i] == rgb_list[i]:
                if i < 9:
                    rgbn_name = "RGBNC_00" + str(i + 1) + ".h5"
                    rgb_name = "RGBNC_00" + str(i + 1) + ".h5"
                    rgbr_name = "RGBR_00" + str(i + 1) + ".h5"
                    os.rename(rgbn_dir + rgbn_list[i], rgbn_dir + rgbn_name)
                    os.rename(rgb_dir + rgb_list[i], rgb_dir + rgb_name)
                    os.rename(rgbr_dir + rgbr_list[i], rgbr_dir + rgbr_name)
                    print("rgbn changed: ", rgbn_dir + rgbn_list[i], " ", rgbn_dir + rgbn_name)
                    print("rgb changed: ", rgb_dir + rgb_list[i], " ", rgb_dir + rgb_name)
                    print("rgbr changed: ", rgbr_dir + rgbr_list[i], " ", rgbr_dir + rgbr_name)

                if i < 99 and i >= 9:
                    rgbn_name = "RGBNC_0" + str(i + 1) + ".h5"
                    rgb_name = "RGBNC_0" + str(i + 1) + ".h5"
                    rgbr_name = "RGBR_0" + str(i + 1) + ".h5"
                    os.rename(rgbn_dir + rgbn_list[i], rgbn_dir + rgbn_name)
                    os.rename(rgb_dir + rgb_list[i], rgb_dir + rgb_name)
                    os.rename(rgbr_dir + rgbr_list[i], rgbr_dir + rgbr_name)
                    print("rgbn changed: ", rgbn_dir + rgbn_list[i], " ", rgbn_dir + rgbn_name)
                    print("rgb changed: ", rgb_dir + rgb_list[i], " ", rgb_dir + rgb_name)
                    print("rgbr changed: ", rgbr_dir + rgbr_list[i], " ", rgbr_dir + rgbr_name)

                if i >= 99:
                    rgbn_name = "RGBNC_" + str(i + 1) + ".h5"
                    rgb_name = "RGBNC_" + str(i + 1) + ".h5"
                    rgbr_name = "RGBR_" + str(i + 1) + ".h5"
                    os.rename(rgbn_dir + rgbn_list[i], rgbn_dir + rgbn_name)
                    os.rename(rgb_dir + rgb_list[i], rgb_dir + rgb_name)
                    os.rename(rgbr_dir + rgbr_list[i], rgbr_dir + rgbr_name)
                    print("rgbn changed: ", rgbn_dir + rgbn_list[i], " ", rgbn_dir + rgbn_name)
                    print("rgb changed: ", rgb_dir + rgb_list[i], " ", rgb_dir + rgb_name)
                    print("rgbr changed: ", rgbr_dir + rgbr_list[i], " ", rgbr_dir + rgbr_name)

            else:

                n[j, 0] = i
                j += 1
        # print(n)
        # print(" sum: ", np.sum(n))

def list_files(dir1):

    files_list = os.listdir(dir1)
    files_list.sort()
    print(files_list)
    print(len(files_list))

    return files_list


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass


    return False

def draw_segmentation(json_data, image_size, dataset_name=None):

    labels=  json_data['Label']
    edge_img = np.zeros((image_size[0],image_size[1]))
    edge_imgs =np.empty([image_size[0],image_size[1],1])
    # asigning dictionaries names
    lbl_names = []
    if 'Poligono' in labels:
        lbl_names.append('Poligono')
        polyg_name='Poligono'
    elif 'Poligon' in labels:
        lbl_names.append('Poligon')
        polyg_name = 'Poligon'
    elif 'Polygon' in labels:
        lbl_names.append('Polygon')
        polyg_name = 'Polygon'

    if 'polilinea' in labels:
        lbl_names.append('polilinea')
        line_name = 'polilinea'
    elif 'Polyline' in labels:
        lbl_names.append('Polyline')
        line_name = 'Polyline'
    elif 'Line' in labels:
        lbl_names.append('Line')
        line_name = 'Line'

    if len(lbl_names)==len(labels):
        n_labels = len(labels)
    else:
        print("there is/are inconsistency in labels and lbl_names size")
        print("Image name:", json_data['External ID'])
        return None

    i_lbls = 0
    indexes = []
    while (i_lbls<n_labels):
        i_sublbls = 0
        while(i_sublbls<len(labels[lbl_names[i_lbls]])):
            tmp_pts = labels[lbl_names[i_lbls]][i_sublbls] # assignation of annotations
            if not tmp_pts ==[]:
                for i in range(len(labels[lbl_names[i_lbls]][i_sublbls])):
                # for i in range(len(labels[lbl_names[i_lbls]][i_sublbls]['geometry'])):
                    x = tmp_pts[i]['y'] # x = tmp_pts['geometry'][i]['y'] #
                    # x = tmp_pts['geometry'][i]['y'] # x= tmp_pts[i]['y']
                    y = tmp_pts[i]['x'] # y = tmp_pts['geometry'][i]['x']# y = tmp_pts[i]['x']
                    # edge_img[y,x]=1
                    indexes.append(y)
                    indexes.append(x)
                if len(indexes)>2:  # when there is an error in the annotation

                    if lbl_names[i_lbls]== line_name:#'Polyline': # 'Line'

                        edge_img = draw_edges(image_size,indexes,False)
                    else:
                        edge_img = draw_edges(image_size, indexes, True)

                i_sublbls += 1

                edge_imgs = np.append(edge_imgs,np.expand_dims(edge_img,axis=-1),axis=-1)
                indexes = []
            else:
                i_sublbls += 1

        i_lbls += 1

    print("ready")
    return edge_imgs

# for drawing lines and polygons
def draw_edges(img_size,xy,is_polygon=False):

    if is_polygon:
        edge_img = Image.new('RGB', [img_size[1], img_size[0]], "black")
        a = ImageDraw.Draw(edge_img)
        a.polygon(xy, fill='black', outline='white')
        del a
        # edge_img.save('a.png',"PNG")
        edge_img = np.array(edge_img)
        edge_img = cv2.cvtColor(edge_img, cv2.COLOR_RGB2GRAY)
        edge_img = np.flipud(edge_img)
        return edge_img
    else:
        edge_img = Image.new('RGB', [img_size[1], img_size[0]], "black")
        a = ImageDraw.Draw(edge_img)
        a.line(xy, fill='white')  # outline='white'
        del a
        # edge_img.save('a.png',"PNG")
        edge_img = np.array(edge_img)
        edge_img = cv2.cvtColor(edge_img, cv2.COLOR_RGB2GRAY)
        edge_img = np.flipud(edge_img)
        return edge_img



    # tmp_labels
def making_edge_gt(dataset_dir):
    edges_dir =os.path.join(dataset_dir,'all_edge')
    save_dir = os.path.join(dataset_dir,'edge')
    file_list = os.listdir(edges_dir)
    file_list.sort()
    print(len(file_list))
    n=len(file_list)
    for i in range(0,n,6):
        img =[]
        e=0
        for j in range(n//100):
            tmp = cv2.imread(os.path.join(edges_dir,file_list[i+j]))
            # dic_gt = {'groundTruth':{}}
            if file_list[i][0:-5]== file_list[i+j][0:-5]:
                im = cv2.resize(tmp,dsize=(tmp.shape[1]//2,tmp.shape[0]//2))
                tmp =cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
                img.append(tmp)
                cv2.imshow(file_list[i+j],im)
                cv2.waitKey(250)
                cv2.destroyAllWindows()
            else:
                e+=1
        if e<1:
            print("results: ",file_list[i][0:-6])
            imgs = np.array(img)
            print(imgs.shape)
            imgs = np.sum(imgs, axis=0)
            file_name = file_list[i][0:-6]
            print(file_name)
            # cv2.namedWindow("showImg")
            cv2.imshow("showImg", np.uint8(imgs))
            cv2.waitKey(250)
            cv2.destroyAllWindows()
            print(imgs.shape)
            dic1_img = {'Boundaries': img}
            dic2_img = {'groundTruth': img}
            # cv2.imwrite(os.path.join(save_dir,file_name),np.uint8(img))
            sio.savemat(os.path.join(save_dir,file_name),dic2_img)
        else:
            print('Inconsistency error')
    print('Finished without problem')

def read_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        for box in boxes.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes

def read_mat(mat_file,is_processend=False):
    if not is_processend:

        data = sio.loadmat(mat_file)
        img_name = os.path.basename(mat_file)
        i = data['LabelMap']
        # i = utls.image_normalization(i,img_min=0,img_max=1)
        img = utls.edge_maker(i)
        # utls.cv_imshow(np.uint8(img*255))
        img_name = img_name[:-3]+'jpg'
        return img_name, img
    else:
        return sio.loadmat(mat_file)

def save_mat(mat_file,mat_path,as_dict=False):
    if as_dict:
        groundtruth = {'groundTruth':{'Boundaries':mat_file}}
        sio.savemat(mat_path, groundtruth)
        print('data saver in: ', mat_path)

    else:
        data={'Boudaries':mat_file}
        sio.savemat(mat_path, data)
        print('data saver in: ',mat_path)