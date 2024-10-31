
import mindspore.nn as nn

import numpy as np



def evaluate_model(model, val_dataset):
    # 使用验证数据集进行验证
    # 注意在此处调用eval时不需要再传递参数
    model.eval(val_dataset)# 设置模型为评估模式
    total_loss = 0
    correct = 0
    total = 0

    for data in val_dataset.create_dict_iterator():
        images = data["image"]
        labels = data["label"]
        logits = model.predict(images)
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')(logits, labels)
        total_loss += loss.asnumpy()

        # 计算准确率
        predicted = np.argmax(logits.asnumpy(), axis=1)
        correct += np.sum(predicted == labels.asnumpy())
        total += labels.shape[0]

    avg_loss = total_loss / total  # 平均损失
    accuracy = correct / total * 100.0  # 准确率
    print(f'Validation Loss: {avg_loss}, Validation Accuracy: {accuracy:.2f}%')