import os

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore import Tensor
import mindspore

def create_dataset(path,batch_size=32, train=True):
    train_path = os.path.join(path, 'train')
    test_path = os.path.join(path, 'test')
    if not os.path.exists(train_path):
        raise ValueError(f"The train directory {train_path} does not exist!")
    if not os.path.exists(test_path):
        raise ValueError(f"The test directory {test_path} does not exist!")
    # 1. 定义数据集（从MNIST数据集中加载训练或测试数据）
    dataset = ds.MnistDataset(train_path if train else test_path)

    # 2. 定义数据增强和处理所需参数
    resize_height, resize_width = 28, 28  # 修改图片尺寸
    normalize_mean = [0.5]  # 使用列表或元组定义均值
    normalize_std = [0.5]  # 使用列表或元组定义标准差

    channel_num = 1  # 输入通道数

    # 3. 生成数据增强操作
    transform_ops = [
        vision.Resize((resize_height, resize_width)),  # 修改图片尺寸
        vision.Normalize(mean=normalize_mean, std=normalize_std),  # 归一化处理
        vision.HWC2CHW(),  # 修改图像频道数
    ]

    # 4. 使用map映射函数，将数据操作应用到数据集
    dataset = dataset.map(operations=transforms.Compose(transform_ops), input_columns="image")

    # 5. 修改标签的数据类型
    dataset = dataset.map(operations=transforms.TypeCast(mindspore.int32), input_columns="label")

    # shuffle操作
    dataset = dataset.shuffle(buffer_size=10000)  # 设定随机打乱的缓冲区大小

    # 设置batch_size，drop_remainder为True
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

    return dataset



if __name__ == '__main__':

    date_path='..\\MNIST_DATA'
    # 使用create_dataset函数创建训练和测试数据集
    train_dataset = create_dataset(path=date_path, batch_size=32, train=True)
    test_dataset = create_dataset(path=date_path, batch_size=32, train=False)

    # 遍历数据集中的数据
    for data in train_dataset.create_dict_iterator():
        images = data["image"]  # 获取图像数据
        labels = data["label"]  # 获取标签数据
        print(f"images shape: {images.shape} \n label shape: {labels.shape}")


        # print(images.shape, labels.shape)  # 输出形状
        break  # 仅输出一个batch的数据
