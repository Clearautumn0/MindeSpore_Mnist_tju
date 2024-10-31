import os
import mindspore as ms
import mindspore.context as context
#transforms.c_transforms用于通用型数据增强，vision.c_transforms用于图像类数据增强
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV

#nn模块用于定义网络，model模块用于编译模型，callback模块用于设定监督指标
from mindspore import nn
from mindspore.train import Model # type: ignore
from mindspore.train.callback import LossMonitor # type: ignore
#设定运行模式为图模式，运行硬件为昇腾芯片
context.set_context(mode=context.GRAPH_MODE, device_target='CPU') # Ascend, CPU, GPU

DATA_PATH = '..\\MNIST_DATA'  # 训练数据路径
BATCH_SIZE = 32                              # 批量大小
LEARNING_RATE = 0.01                        # 学习率
EPOCHS = 5                                   # 训练轮数





