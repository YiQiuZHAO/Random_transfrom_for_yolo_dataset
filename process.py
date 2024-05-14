import os.path
from tools import *
from randomtrans import *


def process_dataset(filepath, savepath, methods: list, task):
    """
    process_dataset with random transformation
    :param filepath:
    :param savepath:
    :param methods:
    :return: None
    """
    imgs = glob(filepath + '/*.jpg')+glob(filepath + '/*.tif')
    partnum = len(imgs) / len(methods)
    random.shuffle(imgs)
    assignment_list = []
    for i in range(len(methods)):
        assignment_list.append(imgs[i * int(partnum):(i + 1) * int(partnum)])
    for i, m in enumerate(methods):
        process_imgs(assignment_list[i], m, savepath, task)


def split_dataset(filepath, savepath, ax, ay, task):
    if type(filepath) == str:
        assignment_list = glob(filepath + '/*.jpg')
    elif type(filepath) == list:
        assignment_list = filepath
    else:
        return
    for i, file in enumerate(assignment_list):
        print(len(assignment_list), i)
        img, labels, name = read_(file, task=task)
        if type(img) == np.ndarray:
            imgs, labelss = average_split(img, labels, ax, ay, task)
            for i in range(ax * ay):
                name = name + ''.join(['0'] * (len(str(ax * ay)) - len(str(i))) + [str(i)])

                write_(imgs[i], labelss[i], savepath, name, 'split', task)


def organize(filepath, savepath, rat: list = [8, 1, 1], rate=1):
    """
    Organize dataset
    :param filepath:
    :param savepath:
    :param rat: rate of train, val, test set. example:[8,1,1]
    :param rate: sample rate
    :return: None
    """
    print('Organizing.......')
    train, val, test = rat
    if type(filepath) == str:
        imgs = glob(filepath + '/*.jpg')+glob(filepath + '/*.tif')
    elif type(filepath) == list:
        imgs = filepath
    else:
        return
    # imgs = glob(filepath + '/*.jpg')
    partnum = int(len(imgs) / sum(rat))
    random.shuffle(imgs)
    if 0 < rate < 1:
        imgs = random.sample(imgs, int(rate * len(imgs)))
    elif rate == 1:
        pass
    else:
        print("Rate error")
        return

    assignment_list = []
    assignment_list.append(imgs[0:train * partnum])
    assignment_list.append(imgs[train * partnum:(train + val) * partnum])
    assignment_list.append(imgs[(train + val) * partnum:(train + val + test) * partnum])

    train_path = '/'.join([savepath, 'train'])
    val_path = '/'.join([savepath, 'val'])
    test_path = '/'.join([savepath, 'test'])

    copy_file(assignment_list[0], train_path)
    copy_file(assignment_list[1], val_path)
    copy_file(assignment_list[2], test_path)

    f = open(savepath + '/data.yaml', 'w')
    f.write('path: '+savepath + '\n')
    f.write('train: '+'/train/images' + '\n')
    f.write('val: ' + '/val/images' + '\n')
    f.write('test: ' + '/test/images' + '\n')
    f.write('nc: \n')
    f.write('names: \n  0: 0')
    f.close()

    f = open(savepath + '/train.txt','w')
    f.writelines(list(map(lambda a:os.path.split(a)[1]+'\n',assignment_list[0])))
    f.close()
    f = open(savepath + '/val.txt','w')
    f.writelines(list(map(lambda a:os.path.split(a)[1]+'\n',assignment_list[1])))
    f.close()
    f = open(savepath + '/test.txt','w')
    f.writelines(list(map(lambda a:os.path.split(a)[1]+'\n',assignment_list[2])))
    f.close()
    print('Finish.')


def process_imgs(assignment_list, m, savepath, task):
    for file in assignment_list:
        img, labels, name = read_(file, task=task)
        if type(img) != type(None):

            mm = eval(m[0])

            if mm in {random_brightness, random_channel, random_contrast, random_gauss, random_shadow, random_spot}:
                img = mm(img)

            if mm in {random_gray, random_resize}:
                arg = m[1][1]
                img = mm(img, *arg)

            if mm in {random_crop, random_flip, random_rotation, random_translate}:
                img, labels = mm(img, labels, task=task)

            if mm in {random_perspective}:
                arg = m[1][1]
                img, labels = mm(img, labels, *arg, task=task)

            write_(img, labels, savepath, name, m[0].split('_')[1], task=task)

            # img = draw_labels(img, labels, task=task)
            # show_imgs(img, key=30, des=False)
    cv2.destroyAllWindows()


def process(filepath, savepath, method_list: dict, single):
    """
    process in two mod
    :param filepath:
    :param savepath:
    :param method_list:
    :param single:
    :return:
    """
    methods = method_list['methods']
    task = method_list['task']
    ms = []
    for key in methods:
        if methods[key][0]:
            ms.append([key, methods[key]])
    if not len(methods) > 0:
        print("No method selected!")
        return
    if single:
        process_dataset(filepath, savepath, ms, task=task)
        print("Finish.")
    else:
        for i, meth in enumerate(ms):
            process_dataset(filepath, savepath, [meth], task=task)
            print(meth, "done.", i + 1, "of", len(ms))
        print("Finish.")


# class method_list():
#     def __init__(self, filepath, savepath, task='detect'):
#         super().__init__()
#         self.filepath = filepath
#         self.savepath = savepath
#         self.task = task
#         self.random_brightness = [False]
#         self.random_channel = [False]
#         self.random_contrast = [False]
#         self.random_resize = [False, [False]]
#         self.random_gauss = [False]
# 
#         self.random_crop = [False]
#         self.random_flip = [False]
#         self.random_rotation = [False]
#         self.random_translate = [False]
#         self.random_perspective = [False, [30, True, False, 42, False]]
# 
#         self.ms = 0
#         self.update_ms()
# 
#     def update_ms(self):
#         self.ms = {
#             'random_brightness': self.random_brightness,
#             'random_channel': self.random_channel,
#             'random_contrast': self.random_contrast,
#             'random_resize': self.random_resize,
#             'random_gauss': self.random_gauss,
#             'random_crop': self.random_crop,
#             'random_flip': self.random_flip,
#             'random_rotation': self.random_rotation,
#             'random_translate': self.random_translate,
#             'random_perspective': self.random_perspective
#         }
# 
#     def get_ms(self, single=True):
#         mss = []
#         self.update_ms()
#         for key in self.ms:
#             if self.ms[key][0]:
#                 mss.append(key)
#         if not len(mss) > 0:
#             print("No method selected!")
#             return
#         else:
#             l = {'task': self.task,
#                  'methods': self.ms[mss]}
#             return l


def drawAndSave(filepath, savepath, task='detect'):
    files = glob(filepath + '/*.jpg')
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    for file in files:
        img, labels, name = read_(file, task)
        if type(img) == np.ndarray:
            img = draw_labels(img, labels, task)
            imgsave = savepath + '/' + name + '.jpg'
            cv2.imencode('.jpg', img)[1].tofile(imgsave)


def drawAndShow(filepath, task='detect'):
    files = glob(filepath + '/*.jpg')
    des = False
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    for i, file in enumerate(files):
        img, labels, name = read_(file, task)
        if type(img) == np.ndarray:
            img = draw_labels(img, labels, task)
            if i == len(files) - 1:
                des = True
            show_imgs(img, key=50, des=des)


if __name__ == '__main__':

    filepath = r'E:\00_data\05-0507\data\infused\images'
    savepath = r'E:\00_data\05-0507\data\infused\dataset'
    # task = 'detect'
    # filepath = './test/seg/images'
    # savepath = './test/seg/save'
    task = 'segment'
    drawpath = './test/draw'
    methods = {
        'random_brightness': [True],
        'random_channel': [False],
        'random_contrast': [True],
        'random_crop': [True],
        'random_flip': [True],
        'random_rotation': [True],
        'random_translate': [True],
        'random_perspective': [True, [60, True, True, 42, False]],
        'random_resize': [True, [False]],
        'random_gauss': [True],
        'random_gray': [False, [1]],
        'random_shadow': [True],
        'random_spot': [True]
    }

    m = {
        'task': task,
        'methods': methods
    }
    process(filepath, savepath, m, False)
    # drawAndSave(savepath + '/images',drawpath,task)
    # split_dataset(filepath, savepath, 1, 2, task)
    # savepath = r'F:\train\dataset'
    organize(savepath+'/images',savepath+'/dataset',[8,2,0])
