# Random_transfrom_for_yolo_dataset
* yolo数据集随机变换增强工具
* 图像与标签同时变换

```formattools：yolo格式与labelme格式互换
randomtrans：随机变换函数
process：批处理工具
tools：功能函数
```
# 支持变换类型：
具体参数见注释
```
'random_brightness': 亮度随机变换
'random_channel': 随机通道交换
'random_contrast': 随机对比度变换
'random_crop': 随机裁剪
'random_flip': 随机翻转
'random_rotation': 随机旋转
'random_translate': 随机平移
'random_perspective': 随机透视变换
'random_resize': 随机缩放
'random_gauss': 随极高斯模糊
'random_gray': 随机灰度化
'random_shadow': 随机阴影
'random_spot': 随机亮斑
```
# 支持变换数据集
目标检测、实例分割

# 使用
## 准备：
  将图像与标签（txt）保存至同一文件夹下
## 处理：
  使用process中的process函数进行随即变换
  
  ```函数接收参数filepath: str, savepath: str, method_list: dict, single: bool
    :param filepath: 数据文件目录
    :param savepath: 保存目录
    :param method_list: 随即变换类型及参数
    :param single: True，每张图像仅使用一种变换，输出数据集数量与输入相同； False，每张图像使用多种变换，输出数据数量为输入×变换类型数
  ```
  输出组织
  save_path:
    images:
    labels:
## 数据集组织：
  使用process中的organize函数，随机划分训练集、验证集、测试集
  
  ```函数接收参数filepath: str, savepath: str, rat: list, rate: int
    :param filepath: 数据文件目录
    :param savepath: 保存目录
    :param rat: 训练集、验证集、测试集划分比例，例:[8,1,1]
    :param rate: 数据集随机抽取比例，用以随机缩减数据集
  ```
    
  输出组织：
    ```savepath:
      dataset：
        train：#训练集
          images:
          labels:
        val：#验证集
          images:
          labels:
        test：#测试集
          images:
          labels:
        data.yaml #yolo数据集文件
        train.txt #训练集文件名
        val.txt #验证集文件名
        test.txt #测试集文件名
    ```
  
