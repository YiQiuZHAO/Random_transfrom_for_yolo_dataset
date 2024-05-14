# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 17:42:11 2022
@author: https://blog.csdn.net/suiyingy?type=blog
"""
from glob import glob

import cv2
import os
import json
import shutil
import numpy as np
from pathlib import Path

from labelme.utils import img_arr_to_b64

from tools import *


def xyxy2labelme(labels, w, h, id2cls, image_path, save_dir='res/'):
    save_dir = str(Path(save_dir)) + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    label_dict = {}
    label_dict['version'] = '5.0.1'
    label_dict['flags'] = {}
    label_dict['imageData'] = None
    label_dict['imagePath'] = image_path
    label_dict['imageHeight'] = h
    label_dict['imageWidth'] = w
    label_dict['shapes'] = []
    for l in labels:
        tmp = {}
        tmp['label'] = id2cls[int(l[0])]
        tmp['points'] = [[l[1], l[2]], [l[3], l[4]]]
        tmp['group_id'] = None
        tmp['shape_type'] = 'rectangle'
        tmp['flags'] = {}
        label_dict['shapes'].append(tmp)
    fn = save_dir + image_path.rsplit('.', 1)[0] + '.json'
    with open(fn, 'w') as f:
        json.dump(label_dict, f)


def yolo2labelme(yolo_image_dir, yolo_label_dir, id2cls, save_dir='res/'):
    yolo_image_dir = str(Path(yolo_image_dir)) + '/'
    yolo_label_dir = str(Path(yolo_label_dir)) + '/'
    save_dir = str(Path(save_dir)) + '/'
    image_files = os.listdir(yolo_image_dir)
    for iimgf, imgf in enumerate(image_files):
        print(iimgf + 1, '/', len(image_files), imgf)
        fn = imgf.rsplit('.', 1)[0]
        shutil.copy(yolo_image_dir + imgf, save_dir + imgf)
        image = cv2.imread(yolo_image_dir + imgf)
        h, w = image.shape[:2]
        if not os.path.exists(yolo_label_dir + fn + '.txt'):
            continue
        labels = np.loadtxt(yolo_label_dir + fn + '.txt').reshape(-1, 5)
        if len(labels) < 1:
            continue
        labels[:, 1::2] = w * labels[:, 1::2]
        labels[:, 2::2] = h * labels[:, 2::2]
        labels_xyxy = np.zeros(labels.shape)
        labels_xyxy[:, 1] = np.clip(labels[:, 1] - labels[:, 3] / 2, 0, w)
        labels_xyxy[:, 2] = np.clip(labels[:, 2] - labels[:, 4] / 2, 0, h)
        labels_xyxy[:, 3] = np.clip(labels[:, 1] + labels[:, 3] / 2, 0, w)
        labels_xyxy[:, 4] = np.clip(labels[:, 2] + labels[:, 4] / 2, 0, h)
        xyxy2labelme(labels_xyxy, w, h, id2cls, imgf, save_dir)
    print('Completed!')


# 支持中文路径
def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
    return cv_img


def labelme2yolo_single(label_file, cls2id):
    anno = json.load(open(label_file, "r", encoding="utf-8"))
    shapes = anno['shapes']
    w0, h0 = anno['imageWidth'], anno['imageHeight']
    image_path = os.path.basename(anno['imagePath'])
    labels = []
    for s in shapes:
        pts = s['points']
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        x = (x1 + x2) / 2 / w0
        y = (y1 + y2) / 2 / h0
        w = abs(x2 - x1) / w0
        h = abs(y2 - y1) / h0
        cid = cls2id[s['label']]
        labels.append([cid, x, y, w, h])
    return np.array(labels), image_path


def labelme2yolo(labelme_label_dir, cls2id, save_dir='res/'):
    labelme_label_dir = str(Path(labelme_label_dir)) + '/'
    save_dir = str(Path(save_dir)) + '/'
    yolo_label_dir = save_dir + 'labels/'
    yolo_image_dir = save_dir + 'images/'
    if not os.path.exists(yolo_image_dir):
        os.makedirs(yolo_image_dir)
    if not os.path.exists(yolo_label_dir):
        os.makedirs(yolo_label_dir)

    json_files = glob(labelme_label_dir + '*.json')
    for ijf, jf in enumerate(json_files):
        print(ijf + 1, '/', len(json_files), jf)
        filename = os.path.basename(jf).rsplit('.', 1)[0]
        labels, image_path = labelme2yolo_single(jf, cls2id)
        if len(labels) > 0:
            np.savetxt(yolo_label_dir + filename + '.txt', labels)
            shutil.copy(labelme_label_dir + image_path, yolo_image_dir + image_path)
    print('Completed!')


def yolo2labelmeseg(yolo_image_dir, yolo_label_dir, save_dir, id2cls):
    # filelist =glob(yolo_label_dir+'/*.txt')
    imglist = glob(yolo_image_dir + '/*.jpg')
    save_dir = str(Path(save_dir)) + '/'
    for i, img in enumerate(imglist):
        img_, labels, name = read_(img, task='segment')
        h, w = img_.shape[:2]

        save_dir = str(Path(save_dir)) + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        label_dict = {}
        label_dict['version'] = '5.2.0.post4'
        label_dict['flags'] = {}
        label_dict['shapes'] = []
        label_dict['imagePath'] = name + '.jpg'
        label_dict['imageData'] = img_arr_to_b64(cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)).decode('utf-8')
        # label_dict['imageData'] = None
        label_dict['imageHeight'] = h
        label_dict['imageWidth'] = w

        for l in labels:
            tmp = {}
            tmp['label'] = id2cls[int(l[0])]
            l[1::2] *= w
            l[2::2] *= h
            points = l[1:].reshape(-1, 2)
            points = points[::int(len(points)/50)] if len(points)>50 else points
            points = list(map(list, points))

            tmp['points'] = points
            tmp['group_id'] = None
            tmp['description'] = ''
            tmp['shape_type'] = 'polygon'
            tmp['flags'] = {}
            label_dict['shapes'].append(tmp)
        fn = save_dir + name + '.json'
        with open(fn, 'w') as f:
            json.dump(label_dict, f)

        # print(img)

        print(i, '/', len(imglist))
        continue


if __name__ == '__main__':
    id2cls = {0: '0'}
    cls2id = {'0': 0}

    root_dir = r'E:\00_data\05-0507\data\infused'
    save_dir = r'E:\00_data\05-0507\data\infused'
    # labelme2yolo(root_dir, cls2id, save_dir)
    # yolo2labelmeseg(r'E:\00_data\05-0507\sum', r'E:\00_data\05-0507\predict4', r'E:\00_data\05-0507\predict4', id2cls)

    # yolo_image_dir = r'E:\00_data\01_fushi_435\data\images'
    # yolo_label_dir = r'E:\00_data\01_fushi_435\data\labels'
    # save_dir = r'E:\00_data\01_fushi_435\data\images'
    # yolo2labelmeseg(root_dir, root_dir, save_dir, id2cls)
    labelme2yolo
