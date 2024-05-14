from glob import glob
from tools import *
import os.path
import cv2
import numpy as np
import random
from math import e, radians, cos, sin
from scipy.spatial import ConvexHull


def random_brightness(img: np.array):
    """
    Random brightness transformation
    :param img: original image
    :return: transformed img
    """
    scart = random.uniform(0.5, 255 / img.max())
    return (img * scart).astype('uint8')


def random_contrast(img: np.array):
    """
    Random contrast transformation
    :param img: original image
    :return: transformed img
    """
    alpha = random.uniform(16, 32)
    img = img.astype('float')
    # img_ = cv2.cvtColor(img_.astype('uint16'), cv2.COLOR_BGR2HSV).astype('float')
    for i in range(3):
        # img_ = img[:, :, i]
        img[:, :, i] *= (255 / (1 + e ** (-(img[:, :, i] - 128) / alpha)))
    img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    # img_ = cv2.cvtColor(img_.astype('uint16'), cv2.COLOR_HSV2BGR)
    # img[:, :, :3] = img_.astype('uint16')
    return img.astype('uint8')


def random_channel(img: np.array):
    """
    Random channel transformation
    :param img: original image
    :return: transformed img
    """
    c = [0, 1, 2]
    random.shuffle(c)
    return img[:, :, c]


def random_gauss(img: np.array):
    k = random.randint(3, 9)
    k = k + 1 if k % 2 == 0 else k
    return cv2.GaussianBlur(img, (k, k), 0)


def random_resize(img: np.array, keep: bool = False):
    """
    :param img: nparray
    :param keep: keep ratio
    :return: transformed img
    """
    if keep:
        fx = fy = random.uniform(0.5, 1.5)
    else:
        fx = random.uniform(0.5, 1.5)
        fy = random.uniform(0.5, 1.5)
    img = cv2.resize(img, (0, 0), fx=fx, fy=fy)
    return img


def random_translate(img: np.array, labels, task='detect'):
    """
    Random translation transformation
    :param img: original image
    :param labels: labels:nparray
    :return: transformed img,transformed labels
    """
    fx = random.uniform(-0.5, 0.5)
    fy = random.uniform(-0.5, 0.5)
    y, x = img.shape[:2]
    M = np.float_([[1, 0, fx * x], [0, 1, fy * y]])
    img = cv2.warpAffine(img, M, (x, y))
    if task == 'detect':
        labels[:, 1] += fx
        labels[:, 2] += fy
        return img, check_labels(labels, task=task, x=x, y=y)
    elif task == 'segment':
        # for i, l in enumerate(labels):
        #     dots = l[1:].reshape(2, -1)
        #     dots[0] *= fx
        #     dots[1] *= fy
        #     l[1:] = dots.reshape(1, -1)
        #     labels[i] = l
        r = []
        for i, l in enumerate(labels):
            temp = np.zeros((y, x)).astype('uint8')
            if len(l[1:]) > 0:
                dots = l[1:].reshape(-1, 1, 2)
                dots[:, :, 0] *= x
                dots[:, :, 1] *= y
                temp = cv2.drawContours(temp, [dots.astype(int)], -1, 255, -1)
                temp = cv2.warpAffine(temp, M, (x, y))
                c, t = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(c) > 0:
                    if len(c[0]) >= 3:
                        c = c[0].astype(float)
                        c[:, :, 0] = c[:, :, 0] / x
                        c[:, :, 1] = c[:, :, 1] / y
                        ds = c.reshape(1, -1)
                        l = np.insert(ds, 0, l[0])
                        r.append(l)
        return img, r


def random_flip(img: np.array, labels, task='detect'):
    """
    Random flip transformation
    :param img: original image
    :param labels: labels:nparray
    :return: transformed img,transformed labels
    """
    axis = random.randint(0, 2)
    if task == 'detect':
        if axis == 2:
            labels[:, [1, 2]] = 1 - labels[:, [1, 2]]
            img = np.flip(img)[:, :, ::-1]
        elif axis == 1:
            labels[:, 1] = 1 - labels[:, 1]
            img = np.flip(img, axis=axis)
        elif axis == 0:
            labels[:, 2] = 1 - labels[:, 2]
            img = np.flip(img, axis=axis)
    elif task == 'segment':
        if axis == 2:
            for i, l in enumerate(labels):
                l[1:] = 1 - l[1:]
                labels[i] = l
            img = np.flip(img)[:, :, ::-1]
        elif axis == 1:
            for i, l in enumerate(labels):
                l[1::2] = 1 - l[1::2]
                labels[i] = l
            img = np.flip(img, axis=axis)
        elif axis == 0:
            for i, l in enumerate(labels):
                l[2::2] = 1 - l[2::2]
                labels[i] = l
            img = np.flip(img, axis=axis)
    return img, labels


def random_rotation(img: np.array, labels, task='detect'):
    """
    Random rotation transformation
    :param img: original image
    :param labels: labels:nparray
    :return: transformed img,transformed labels
    """
    y, x = img.shape[:2]
    rcenter = (0.5 * x, 0.5 * y)
    angle = random.randint(1, 359)
    rangle = radians(angle)
    center = (int(rcenter[0]), int(rcenter[1]))
    M = cv2.getRotationMatrix2D(center, angle, 1)
    img = cv2.warpAffine(img, M, (x, y))

    if task == 'detect':
        x1 = labels[:, 1] * x
        y1 = labels[:, 2] * y
        r1 = labels[:, 3] / 2 * x
        r2 = labels[:, 4] / 2 * y

        x_ = (x1 - rcenter[0]) * cos(-rangle) - (y1 - rcenter[1]) * sin(-rangle) + rcenter[0]
        y_ = (x1 - rcenter[0]) * sin(-rangle) + (y1 - rcenter[1]) * cos(-rangle) + rcenter[1]

        ls = get_rectangle(x_, y_, r1, r2, np.asarray([rangle] * len(labels)))
        labels[:, 1:] = ls
        labels[:, [1, 3]] /= x
        labels[:, [2, 4]] /= y
        return img, check_labels(labels, task=task, x=x, y=y)
    elif task == 'segment':
        r = []
        for i, l in enumerate(labels):
            temp = np.zeros((y, x)).astype('uint8')
            if len(l[1:]) > 0:
                dots = l[1:].reshape(-1, 1, 2)
                dots[:, :, 0] *= x
                dots[:, :, 1] *= y
                temp = cv2.drawContours(temp, [dots.astype(int)], -1, 255, -1)
                temp = cv2.warpAffine(temp, M, (x, y))
                c, t = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(c) > 0:
                    if len(c[0]) >= 3:
                        c = c[0].astype(float)
                        c[:, :, 0] = c[:, :, 0] / x
                        c[:, :, 1] = c[:, :, 1] / y
                        ds = c.reshape(1, -1)
                        l = np.insert(ds, 0, l[0])
                        r.append(l)
        return img, r


def random_perspective(img: np.array, labels, a: int = 30, ax: bool = False, ay: bool = True, fov: int = 42,
                       random_axis=False, task='detect'):
    """
    Random perspective transforms
    :param img: original image
    :param labels: labels:nparray
    :param a: angle limit
    :param ax: perspective in axis x
    :param ay: perspective in axis y
    :param fov: fov of imaging
    :param random_axis: random choose axis
    :return: transformed img,transformed labels
    """
    if random_axis:
        ra = random.randint(0, 1)
        ax = True if ra == 0 else False
        ay = True if ra == 1 else False
    angle_x = random.randint(-a, a) if ax else 0
    angle_y = random.randint(-a, a) if ay else 0
    # anglez = random.randint(-60, 60) if az else 0
    img, labels = warp_img(img, labels, angle_x, angle_y, fov, task=task)
    return img, labels


def random_crop(img: np.array, labels, task='detect'):
    """
    Random crop transforms
    :param img: original image
    :param labels: labels:nparray
    :return: transformed img,transformed labels
    """
    y, x = img.shape[:2]
    l = random.uniform(0, 0.4)
    r = random.uniform(0.6, 1)
    t = random.uniform(0, 0.4)
    b = random.uniform(0.6, 1)
    labels = check_labels(labels, l, r, t, b, task=task, x=x, y=y)
    # print(l, r, t, b, task)
    lp = int(l * x)
    rp = int(r * x)
    tp = int(t * y)
    bp = int(b * y)
    img[:tp] = 0
    img[:, :lp] = 0
    img[:, rp:] = 0
    img[bp:] = 0
    return img, labels


def average_split(img, labels, ax, ay, task):
    y, x = img.shape[:2]
    px = int(x / ax)
    py = int(y / ay)
    rx = 1 / ax
    ry = 1 / ay
    imgs = []
    labelss = []
    if task == 'detect':
        for ix in range(ax):
            for iy in range(ay):
                imgs.append(img[iy * py:(iy + 1) * py, ix * px:(ix + 1) * px])
                t = labels.copy()
                t[:, 1] = (t[:, 1] - ix * rx) * ax
                t[:, 2] = (t[:, 2] - iy * ry) * ay
                t[:, 3] *= ax
                t[:, 4] *= ay
                labelss.append(check_labels(t, task=task))
    elif task == 'segment':
        for ix in range(ax):
            for iy in range(ay):
                t = labels.copy()
                imgs.append(img[iy * py:(iy + 1) * py, ix * px:(ix + 1) * px])
                r = []
                for i, l in enumerate(t):
                    temp = np.zeros((y, x)).astype('uint8')
                    dots = l[1:].reshape(-1, 1, 2).copy()
                    x_ = dots[:, :, 0] * x
                    y_ = dots[:, :, 1] * y
                    dots[:, :, 0] = x_
                    dots[:, :, 1] = y_
                    if len(dots) >= 3:
                        cv2.drawContours(temp, [dots.astype(int)], -1, 255, -1)
                        temp1 = temp[iy * py:(iy + 1) * py, ix * px:(ix + 1) * px]
                        cs, t = cv2.findContours(temp1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for c in cs:
                            if len(c) > 3:
                                c = c.astype(float)
                                c[:, :, 0] = c[:, :, 0] / px
                                c[:, :, 1] = c[:, :, 1] / py
                                ds = c.reshape(1, -1)
                                l = np.insert(ds, 0, l[0])
                                r.append(l)
                labelss.append(r)
    return imgs, labelss


def random_gray(img, rate=1):
    if rate <= 1:
        arg = random.randint(1, int(1 / rate))
        if arg == 1:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return img


def random_shadow(img):
    y, x = img.shape[:2]
    num = random.randint(1, 3)
    mask = np.ones(img.shape).astype('float')
    # color = np.random.randint(128, 255, 3)
    # color = tuple(map(int, color))
    x_ = np.random.randint(0, x, num * 10)
    y_ = np.random.randint(0, y, num * 10)
    dots = np.vstack((x_, y_)).T.reshape(num, 10, 2)
    for i in range(num):
        mask2 = np.ones(img.shape)
        rate2 = random.uniform(0.4, 0.8)
        hull = ConvexHull(dots[i])
        hull1 = hull.vertices.tolist()
        cv2.drawContours(mask2, [dots[i][hull1].reshape(-1, 1, 2)], -1, (rate2, rate2, rate2), -1)
        mask *= mask2
    img = img.astype('float') * mask
    return img.astype('uint8')


def random_spot(img):
    y, x = img.shape[:2]
    mask = np.zeros((y, x, 3)).astype('uint8')
    num = random.randint(1, 10)
    xs = np.random.randint(0, x, num * 10)
    ys = np.random.randint(0, y, num * 10)
    rs = np.random.randint(0, int(min(x, y) * 0.2), num)
    for i in range(num):
        c = (xs[i], ys[i])
        r = rs[i]
        cv2.circle(mask, c, r, (255, 255, 255), -1)
    rate = random.uniform(0.5, 0.7)
    img_ = img[:, :, :3]
    img_ = cv2.addWeighted(img_.astype('uint8'), 1, mask, rate, 1)
    img[:, :, :3] = img_.astype('uint8')
    return img


if __name__ == '__main__':
    methods = {
        'random_brightness': [True],
        'random_channel': [False],
        'random_contrast': [False],
        'random_crop': [False],
        'random_flip': [False],
        'random_rotation': [False],
        'random_translate': [False],
        'random_perspective': [False, [30, True, False, 42, False]],
        'random_resize': [False, [False]],
        'random_gauss': [False]
    }
    methods_seg = {
        'random_brightness': [True],
        'random_channel': [True],
        'random_contrast': [True],
        'random_crop_seg': [True],
        'random_flip_seg': [True],
        'random_rotation_seg': [True],
        'random_translate_seg': [True],
        'random_perspective_seg': [True, [30, True, False, 42, False]],
        'random_resize': [True, [False]],
        'random_gauss': [True]
    }
    task = 'segment'

    m = {
        'task': task,
        'methods': methods_seg
    }
    filepath = r'E:\00_data\05-0507\data\infused'
    savepath = r'E:\00_data\05-0507\data\infused\dataset'
    process(filepath, savepath, m, True)
    organize(savepath, savepath + '/dataset', [9, 1, 0])
