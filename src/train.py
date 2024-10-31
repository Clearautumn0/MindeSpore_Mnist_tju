from tqdm import tqdm
from mindspore import nn, CheckpointConfig, ModelCheckpoint, LossMonitor, Accuracy

from src.config import BATCH_SIZE
from src.data_preprocessing import create_dataset
from src.evaluate import evaluate_model
from src.model import LeNet
from mindspore.train import Model
from mindspore.train.callback import LossMonitor # type: ignore
def train(data_path, learning_rate=0.01, epochs=10):

    # 初始化
    lenet_model = LeNet()

    # 定义损失函数
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # 定义优化器
    optimizer = nn.Momentum(lenet_model.trainable_params(), learning_rate=learning_rate, momentum=0.9)
    # 定义评估指标
    metrics = {'accuracy': Accuracy()}
    # 创建模型
    model = Model(lenet_model, loss_fn=loss_fn, optimizer=optimizer, metrics=metrics)

    train_dataset = create_dataset(path=data_path, batch_size=BATCH_SIZE, train=True)
    test_dataset = create_dataset(path=data_path, batch_size=BATCH_SIZE, train=False)

    # 模型检查点配置
    config = CheckpointConfig(save_checkpoint_steps=100, keep_checkpoint_max=5)
    checkpoint_cb = ModelCheckpoint(prefix="lenet", directory="./checkpoints", config=config)

    # 使用封装的状态模型



    model.train(epoch=epochs, train_dataset=train_dataset, callbacks=[LossMonitor(), checkpoint_cb])

    # 每个 epoch 后进行验证
    evaluate_model(model, test_dataset)
