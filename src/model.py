import mindspore
import mindspore.nn as nn
import numpy as np
from mindspore import Tensor
import mindspore.ops as ops

class LeNet(nn.Cell):
    def __init__(self):
        super(LeNet, self).__init__()

        # 定义网络各层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, pad_mode='pad')  # 第一卷积层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样层（池化层）
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, pad_mode='pad')  # 第二卷积层

        self.fc1 = nn.Dense(in_channels=16 * 5 * 5, out_channels=120)  # 第一全连接层
        self.fc2 = nn.Dense(in_channels=120, out_channels=84)  # 第二全连接层
        self.fc3 = nn.Dense(in_channels=84, out_channels=10)  # 第三全连接层（输出层）

    def eval(self):
        self.is_training = False  # 当调用eval时，切换为评估模式
    def construct(self, x):
        # 定义前向传播过程
        x = self.conv1(x)  # 经过第一卷积层
        x = ops.relu(x)  # 激活函数
        x = self.pool(x)  # 经过第一个下采样层

        x = self.conv2(x)  # 经过第二卷积层
        x = ops.relu(x)  # 激活函数
        x = self.pool(x)  # 经过第二个下采样层

        x = x.view(x.shape[0], -1)  # 展平为全连接层的输入
        x = self.fc1(x)  # 经过第一全连接层
        x = ops.relu(x)  # 激活函数
        x = self.fc2(x)  # 经过第二全连接层
        x = ops.relu(x)  # 激活函数
        x = self.fc3(x)  # 经过第三全连接层（输出层）

        return x





if __name__ == '__main__':
    # 实例化LeNet模型
    lenet_model = LeNet()

    # 测试模型前向推理是否正常
    input_tensor = Tensor(np.random.rand(32, 1, 28, 28), mindspore.float32)  # 随机生成输入数据
    output = lenet_model(input_tensor)  # 前向推理
    print(output.shape)  # 输出shape，应为 (32, 10)
