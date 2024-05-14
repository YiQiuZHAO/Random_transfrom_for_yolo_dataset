import random
from glob import glob
import cv2
import os.path
from glob import glob
import cv2
import numpy as np
import random
from math import e, radians, cos, sin


def get_ellipse_param(major_radius, minor_radius, angle):
    # 根据椭圆的主轴和次轴半径以及旋转角度(默认圆心在原点)，得到椭圆参数方程的参数，
    # 椭圆参数方程为：
    #     A * x^2 + B * x * y + C * y^2 + F = 0
    a, b = major_radius, minor_radius
    sin_theta = np.sin(-angle)
    cos_theta = np.cos(-angle)
    A = a ** 2 * sin_theta ** 2 + b ** 2 * cos_theta ** 2
    B = 2 * (a ** 2 - b ** 2) * sin_theta * cos_theta
    C = a ** 2 * cos_theta ** 2 + b ** 2 * sin_theta ** 2
    F = -a ** 2 * b ** 2
    return A, B, C, F


def read_lines(line: str):
    if len(line) > 1:
        line = line.replace('\n', '')
        l = line.split(' ')
        if '' in l: l.remove('')
        return np.asarray(list(map(eval, l)))


def read_(filepath, task):
    rootpath, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    imgfile = '/'.join([rootpath, name + ext])
    txtfile = filepath.replace(ext,'.txt')
    if os.path.exists(txtfile):
        # img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint16), 1)
        img = cv2.imread(filepath,cv2.IMREAD_UNCHANGED)
        labels = open(txtfile, 'r').readlines().copy()
        if task == 'detect':
            labels = np.asarray(list(map(read_lines, labels)))
        elif task == 'segment':
            labels = list(map(read_lines, labels))
        return img, labels, name
    else:
        return None, None, None


# def read_seg(filepath):
#     rootpath, filename = os.path.split(filepath)
#     name, ext = os.path.splitext(filename)
#     imgfile = '/'.join([rootpath, name + ext])
#     txtfile = '/'.join([os.path.abspath(os.path.join(rootpath, '..')), 'labels', name + '.txt'])
#     if os.path.exists(txtfile):
#         img = cv2.imdecode(np.fromfile(imgfile, dtype=np.uint8), 1)
#         labels = open(txtfile, 'r').readlines().copy()
#         labels = list(map(lambda a: np.asarray(list(map(eval, a[:-1].split(' ')))), labels))
#         return img, labels, name
#     else:
#         return None, None, None

def write_lines_seg(line):
    dots = line[1:].reshape(-1, 2)
    if len(line) >= 100:
        step = len(line) // 100
        dots = dots[::step]
        # line[1:] = dots.reshape(1,-1)
    return np.insert(dots.reshape(1, -1), 0, line[0])


def write_(img, labels, savepath, name, m=None, task='detect'):
    name = '-'.join([m, name]) if m != None else name
    imgpath = '/'.join([savepath, 'images'])
    labelpath = '/'.join([savepath, 'labels'])
    if not os.path.isdir(imgpath):
        os.makedirs(imgpath)
    if not os.path.isdir(labelpath):
        os.makedirs(labelpath)
    if img.shape[2] == 3:
        imgfile = '/'.join([imgpath, name + '.jpg'])
    else:
        imgfile = '/'.join([imgpath, name + '.tif'])
    labelfile = '/'.join([labelpath, name + '.txt'])
    # cv2.imencode('.tif', img)[1].tofile(imgfile)
    cv2.imwrite(imgfile, img)
    if len(labels) > 0:
        lines = []
        f = open(labelfile, 'w')
        for line in labels:
            # if len(line) > 1000:
            #     line = line[1::4]
            # if len(line) > 500:
            #     line = line[1::2]
            if task == 'detect':
                l = ' '.join([str(int(line[0]))] + str(line[1:])[1:-1].replace('\n', '').split()) + '\n'
            elif task == 'segment':
                line = write_lines_seg(line)
                l = ' '.join([str(int(line[0]))] + list(map(str, line[1:]))) + '\n'
            else:
                continue
            lines.append(l)
        f.writelines(lines)
        f.close()
    if len(labels) == 0:
        f = open(labelfile, 'w')
        f.close()


def copy_file(l, savepath):
    from shutil import copyfile
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    if not os.path.isdir(savepath + '/images'):
        os.makedirs(savepath + '/images')
    if not os.path.isdir(savepath + '/labels'):
        os.makedirs(savepath + '/labels')
    for file in l:
        filepath, filename = os.path.split(file)
        name, ext = os.path.splitext(filename)
        txtfile = '/'.join([os.path.abspath(os.path.join(filepath, '..')), 'labels', name + '.txt'])
        if os.path.exists(txtfile):
            copyfile(file, savepath + '/images/' + filename)
            copyfile(txtfile, savepath + '/labels/' + name + '.txt')


def calculate_rectangle(A, B, C, F):
    # 根据椭圆参数方程的参数，得到椭圆的外接矩形top-left和right-bottom坐标。
    # 椭圆上下外接点的纵坐标值
    y = np.sqrt(4 * A * F / (B ** 2 - 4 * A * C))
    y1, y2 = -np.abs(y), np.abs(y)
    # 椭圆左右外接点的横坐标值
    x = np.sqrt(4 * C * F / (B ** 2 - 4 * C * A))
    x1, x2 = -np.abs(x), np.abs(x)
    return [x1, y1], [x2, y2]


def get_rectangle(center_x, center_y, major_radius, minor_radius, angle):
    # 按照数据集接口返回矩形框
    A, B, C, F = get_ellipse_param(major_radius, minor_radius, angle)
    p1, p2 = calculate_rectangle(A, B, C, F)
    return np.asarray([center_x, center_y, abs(p1[0]) * 2, abs(p1[1]) * 2]).T


def labels2box(labels, x=1, y=1, task='detect'):
    if task == 'detect':
        l = labels[:, 1:].copy()
        r = np.zeros((len(l), 4))
        labels[:, 1] = np.int_((l[:, 0] - l[:, 2] / 2) * x)
        labels[:, 2] = np.int_((l[:, 1] - l[:, 3] / 2) * y)
        labels[:, 3] = np.int_((l[:, 0] + l[:, 2] / 2) * x)
        labels[:, 4] = np.int_((l[:, 1] + l[:, 3] / 2) * y)
        return labels
    elif task == 'segment':
        for i, l in enumerate(labels):
            l[1::2] *= x
            l[2::2] *= y
            labels[i][1:] = l[1:]
        return labels


def box2labels(labels, x=1, y=1, task='detect'):
    if task == 'detect':
        r = np.zeros((len(labels), 4))
        r[:, 0] = (labels[:, 0] + labels[:, 2]) / (2 * x)
        r[:, 1] = (labels[:, 1] + labels[:, 3]) / (2 * y)
        r[:, 2] = np.abs(labels[:, 2] - labels[:, 0]) / x
        r[:, 3] = np.abs(labels[:, 3] - labels[:, 1]) / y
        return r
    elif task == 'segment':
        for i, l in enumerate(labels):
            dots = l[1:].reshape(2, -1)
            # l[1::2] = l[1::2] / x
            # l[2::2] = l[2::2] / y
            dots[0] /= x
            dots[1] /= y

            labels[i] = np.insert(dots.reshape(1, -1), 0, l[0])

        return labels


def check_labels(labels, ll=0, rl=1, tl=0, bl=1, task='detect', x=1, y=1):
    if task == 'detect':
        result = []
        for i, box in enumerate(labels):
            x_, y_, w_, h_ = box[1:]
            l = x_ - w_ / 2
            t = y_ - h_ / 2
            r = x_ + w_ / 2
            b = y_ + h_ / 2
            if (l <= ll and r <= ll) or (t <= tl and b <= tl) or (l >= rl and r >= rl) or (t >= bl and b >= bl):
                continue
            else:
                l = ll if l <= ll else l
                r = rl if r >= rl else r
                t = tl if t <= tl else t
                b = bl if b >= bl else b
                if abs(r - l) >= 0.005 and abs(b - t) >= 0.005:
                    box2 = np.asarray([(l + r) / 2, (t + b) / 2, abs(r - l), abs(b - t)])
                    box[1:] = box2
                    result.append(box)
        return np.asarray(result)
    elif task == 'segment':
        # ll *= x
        # rl *= x
        # tl *= y
        # bl *= y
        lp = int(ll * x)
        rp = int(rl * x)
        tp = int(tl * y)
        bp = int(bl * y)
        r = []
        for i, l in enumerate(labels):
            # dots = l[1:].reshape(-1, 2)
            temp = np.zeros((y, x)).astype('uint8')
            # lr = []
            # print(len(l))
            if len(l[1:]) > 0:
                dots = l[1:].reshape(-1, 1, 2)
                # print(dots)
                dots[:, :, 0] *= x
                dots[:, :, 1] *= y
                temp = cv2.drawContours(temp, [dots.astype(int)], -1, 255, -1)
                # show_imgs(temp)

                temp[:tp] = 0
                temp[:, :lp] = 0
                temp[:, rp:] = 0
                temp[bp:] = 0

                c, t = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(c) > 0:
                    if len(c[0]) >= 3:
                        c = c[0].astype(float)
                        c[:, :, 0] = c[:, :, 0] / x
                        c[:, :, 1] = c[:, :, 1] / y
                        ds = c.reshape(1, -1)
                        # print(c, ds)
                        # ds[0::2] /= x
                        # ds[1::2] /= y
                        l = np.insert(ds, 0, l[0])
                        r.append(l)

        return r


def warp_points(points, c, anglex=0, angley=0, fov=42, x=1, y=1):
    points = np.concatenate([points, np.asarray([0, 0] * len(points)).reshape(len(points), 2)], 1)
    z = np.sqrt(y ** 2 + x ** 2) / 2 / np.tan(radians(fov / 2))
    points[:, 0] -= c[0]
    points[:, 1] -= c[1]
    rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(radians(anglex)), -np.sin(radians(anglex)), 0],
                   [0, -np.sin(radians(anglex)), np.cos(radians(anglex)), 0, ],
                   [0, 0, 0, 1]], np.float32)
    ry = np.array([[np.cos(radians(angley)), 0, np.sin(radians(angley)), 0],
                   [0, 1, 0, 0],
                   [-np.sin(radians(angley)), 0, np.cos(radians(angley)), 0, ],
                   [0, 0, 0, 1]], np.float32)
    # rz = np.array([[np.cos(radians(anglez)), np.sin(radians(anglez)), 0, 0],
    #                [-np.sin(radians(anglez)), np.cos(radians(anglez)), 0, 0],
    #                [0, 0, 1, 0],
    #                [0, 0, 0, 1]], np.float32)
    # r = rx.dot(ry).dot(rz)
    r = rx.dot(ry)
    pw = np.asmatrix(points) * np.asmatrix(r)
    pw[:, 0] = pw[:, 0] * z / (z - pw[:, 2]) + c[0]
    pw[:, 1] = pw[:, 1] * z / (z - pw[:, 2]) + c[1]
    return np.asarray(pw)


def warp_img(img, labels, anglex=0, angley=0, fov=42, task='detect'):
    y, x = img.shape[0:2]
    center = (int(x / 2), int(y / 2))
    org = np.array([[0, 0],
                    [y, 0],
                    [0, x],
                    [y, x]], np.float32)
    dst = warp_points(org, center, anglex, angley, fov, x, y)[:, :2].astype(np.float32)
    warpR = cv2.getPerspectiveTransform(org, dst)
    img = cv2.warpPerspective(img, warpR, (x, y))

    if task == 'detect':

        xe = labels[:, 1] * x
        ye = labels[:, 2] * y
        r1e = labels[:, 3] / 2 * x
        r2e = labels[:, 4] / 2 * y
        r = []
        for i in range(len(labels)):
            xi, yi, r1i, r2i = int(xe[i]), int(ye[i]), int(r1e[i]), int(r2e[i])
            temp = np.zeros((y, x)).astype('uint8')
            temp = cv2.ellipse(temp, (xi, yi), (r1i, r2i), 0, 0, 360, 128, -1)
            temp = cv2.warpPerspective(temp, warpR, (x, y))
            indexs = np.where(temp != 0)

            if len(indexs[0]) > 3:
                box = np.zeros((4))
                box[1] = (indexs[0].max() + indexs[0].min()) / 2 / y
                box[0] = (indexs[1].max() + indexs[1].min()) / 2 / x
                box[3] = (indexs[0].max() - indexs[0].min()) / y
                box[2] = (indexs[1].max() - indexs[1].min()) / x
                l = np.insert(box, 0, labels[i, 0])
                r.append(l)
        return img, np.asarray(r)
    elif task == 'segment':
        r = []
        for i, l in enumerate(labels):
            temp = np.zeros((y, x)).astype('uint8')
            if len(l[1:]) > 0:
                dots = l[1:].reshape(-1, 1, 2)
                dots[:, :, 0] *= x
                dots[:, :, 1] *= y
                temp = cv2.drawContours(temp, [dots.astype(int)], -1, 255, -1)
                temp = cv2.warpPerspective(temp, warpR, (x, y))
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


# def warp_img_seg(img, labels, anglex=0, angley=0, fov=42):
#     y, x = img.shape[0:2]
#     center = (int(x / 2), int(y / 2))
#     org = np.array([[0, 0],
#                     [y, 0],
#                     [0, x],
#                     [y, x]], np.float32)
#     dst = warp_points(org, center, anglex, angley, fov, x, y)[:, :2].astype(np.float32)
#     warpR = cv2.getPerspectiveTransform(org, dst)
#     img = cv2.warpPerspective(img, warpR, (x, y))
#
#     labels = label2box_seg(labels, x, y)
#     for i, l in enumerate(labels):
#         dots = l[1:].reshape(-1, 2)
#         dots = warp_points(dots, center, anglex, angley, fov, x, y)
#         l[1:] = dots.reshape(1, -1)
#         labels[i] = l
#     labels = box2labels_seg(labels, x, y)
#
#     return img, check_labels(labels)


def show_imgs(imgs=None, names=[], key=0, des=True):
    if type(imgs) == list:
        for i, img in enumerate(imgs):
            name = names[i] if len(imgs) == len(names) else 'Show-' + str(i + 1)
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.imshow(name, img)
        cv2.waitKey(key)
    elif type(imgs) != type(None):
        name = names if names != [] else 'Show'
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, imgs)
        cv2.waitKey(key)
    else:
        cv2.waitKey(1)
    if des:
        cv2.destroyAllWindows()


# colors = [0]*80
# for i in range(len(colors)):
#     colors[i] = str((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))+'\n'
# f = open('colors.txt','w')
# f.writelines(colors)
# f.close()
f = open('colors.txt', 'r')
colors = f.readlines()
f.close()
colors = list(map(lambda n: eval(n[:-1]), colors))


def draw_labels(img, labels, task='detect'):
    global colors
    if len(labels) > 0:
        y, x = img.shape[:2]
        if task == 'detect':
            # colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))]
            boxs = labels2box(labels, x, y)[:, 1:]
            for i, b in enumerate(boxs):
                # if labels[i, 0]+1 > len(colors):
                #     colors += [0] * (labels[i, 0]+1 - len(colors))
                #     colors[labels[i, 0]] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                p1 = b[:2].astype('uint16')
                p2 = b[2:].astype('uint16')
                img = cv2.rectangle(img, p1, p2, colors[int(labels[i, 0])], 3)
        elif task == 'segment':
            labels = labels2box(labels, x, y, task='segment')
            # colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))]
            for i, l in enumerate(labels):
                # if l[0]+1 > len(colors):
                #     colors += [0] * (int(l[0]) - len(colors)+1)
                #     colors[int(l[0])] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                dots = l[1:].reshape(-1, 1, 2).astype(int)
                # print(l[0])
                img = cv2.drawContours(img, [dots], -1, colors[int(l[0])], 3)
    return img


# def draw_labels_seg(img, labels):
#     if len(labels) > 0:
#         y, x = img.shape[:2]
#         labels = label2box_seg(labels, x, y)
#         for i, l in enumerate(labels):
#             dots = l[1:].reshape(-1, 2)
#             img = cv2.drawContours(img, dots, -1, (0, 255, 0), 3)
#     return img


def stats_labels(path: str):
    files = glob(path + '/*.txt')
    labels = []
    for file in files:
        f = open(file, 'r')
        lines = f.readlines()
        if len(lines) > 0:
            for l in lines:
                ls = l.split(' ')
                label = eval(ls[0])
                if label + 1 > len(labels):
                    for i in range(label + 1 - len(labels)):
                        labels.append(0)
                labels[eval(ls[0])] += 1
        f.close()
    print(labels)
    return labels


def stats_only_label(path: str, label=None):
    files = glob(path + '/*.txt')
    if label != -1:
        sames = []
        hasin = []
        for file in files:
            f = open(file, 'r')
            lines = f.readlines()
            label = lines[0].split(' ')[0] if label == None else str(label)
            same = True
            has = False
            for l in lines:
                if l[0].split(' ')[0] != label:
                    same = False
                if l[0].split(' ')[0] == label:
                    has = True
            if same: sames.append(file)
            if has: hasin.append(file)
            f.close()
        return sames, hasin
    else:
        empty = []
        for file in files:
            f = open(file, 'r')
            lines = f.readlines()
            if len(''.join(lines)) <= 5:
                empty.append(file)
            f.close()
        return empty


def splitimg(img, labels, w, h, task):
    y, x = img.shape[:2]
    px = int(x / w)
    py = int(y / h)
    imgs = []
    labelss = []
    for xi in range(w):
        for yi in range(h):
            imgpart = img[yi * py:(yi + 1) * py, xi * px:(xi + 1) * px]
            imgs.append(imgpart)
            if task == 'datect':
                labelt = labels.copy()
                labelt[:, 1] = (labelt[:, 1] - xi / w) * w
                labelt[:, 2] = (labelt[:, 2] - yi / h) * h
                labelt[:, 3] *= w
                labelt[:, 4] *= h
                labelss.append(check_labels(labelt))
            elif task == 'segment':
                labelt = labels.copy()
                for i, l in enumerate(labelt):
                    dots = l[1:].reshape(2, -1)
                    dots[0] = (dots[0] - xi / w) * w
                    dots[1] = (dots[1] - yi / h) * h
                    l[1:] = dots.reshape[1, -1]
                labelss.append(check_labels(labelt, task=task))
    return imgs, labelss


def delete_empty(filepath):
    imgs = glob(filepath + '/*.jpg')
    for file in imgs:
        rootpath, filename = os.path.split(file)
        name, ext = os.path.splitext(filename)
        # imgfile = '/'.join([rootpath, name + ext])
        txtfile = '/'.join([os.path.abspath(os.path.join(rootpath, '..')), 'labels', name + '.txt'])
        if not os.path.exists(txtfile):
            os.remove(file)


def delet_imgs(imgs):
    for file in imgs:
        os.remove(file)


if __name__ == '__main__':
    # img = cv2.imread('test.jpg')
    # labels = open('test.txt', 'r').readlines().copy()
    # labels = np.asarray(list(map(lambda a: np.asarray(list(map(eval, a[:-1].split(' ')))), labels)))
    # img2, labels2 = warpImg(img, labels.copy(), 60, 60, 60)
    # boxs2 = labels2box(labels2[:, 1:], x, y)
    # for i, b in enumerate(boxs2):
    #     p1 = b[:2].astype('uint16')
    #     p2 = b[2:].astype('uint16')
    #     img2 = cv2.rectangle(img2, p1, p2, (0, 0, 255), 3)
    #
    # boxs = labels2box(labels, x, y)
    # for i, b in enumerate(boxs):
    #     p1 = b[:2].astype('uint16')
    #     p2 = b[2:].astype('uint16')
    #     img = cv2.rectangle(img, p1, p2, (0, 0, 255), 3)
    #
    # for i,d in enumerate(dst):
    #     c = (int(d[0,0]),int(d[0,1]))
    #     cv2.circle(img2,c,5,[0,255,0],-1)
    #
    # cv2.namedWindow('1', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('2', cv2.WINDOW_NORMAL)
    # cv2.imshow('1', img)
    # cv2.imshow('2', img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # filepath = 'D:/data/SEG/images/preview0000000000.jpg'
    filepath = 'D:/data/Data/images/IMG_220139-1.jpg'
    # task = 'segment'
    task = 'detect'
    img, labels, name = read_(filepath, task)

    # img2, labels2 = random_perspective(img.copy(), labels.copy(), ax=True,ay=True, task=task)
    # img3 = draw_labels(img2.copy(), labels2.copy(), task)
    # show_imgs(img3)
    # imgs,labelss = average_split(img,labels,2,2,task)
    # for i in range(len(imgs)):
    #     imgs[i] = draw_labels(imgs[i],labelss[i],task)
    # show_imgs(imgs)
