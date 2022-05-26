# -*-coding: utf-8 -*-

import argparse
import os
import numpy as np
import torch
import tensorboardX as tensorboard
from datetime import datetime
from easydict import EasyDict
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from audio.dataloader.audio_dataset import AudioDataset
from audio.utils.utility import print_arguments
from audio.utils import file_utils
from audio.models import mobilenet_v2, resnet


class Train(object):
    """Training  Pipeline"""

    def __init__(self, cfg):
        cfg = EasyDict(cfg.__dict__)
        self.device = "cuda:{}".format(cfg.gpu_id) if torch.cuda.is_available() else "cpu"
        self.num_epoch = cfg.num_epoch
        self.net_type = cfg.net_type
        self.work_dir = os.path.join(cfg.work_dir, self.net_type)
        self.model_dir = os.path.join(self.work_dir, "model")
        self.log_dir = os.path.join(self.work_dir, "log")
        file_utils.create_dir(self.model_dir)
        file_utils.create_dir(self.log_dir)

        self.tensorboard = tensorboard.SummaryWriter(log_dir=self.log_dir)
        self.train_loader, self.test_loader = self.build_dataset(cfg)
        # 获取模型
        self.model = self.build_model(cfg)
        # 获取优化方法，分别设定学习率和权重衰减
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=cfg.learning_rate, weight_decay=5e-4)
        # 获取学习率衰减函数，milestones中的每个元素代表哪几个epoch调整学习率， gamma为学习率调整倍数
        self.scheduler = MultiStepLR(self.optimizer, milestones=[50, 80], gamma=0.1)
        # 获取损失函数，这里采用交叉熵损失函数
        self.losses = torch.nn.CrossEntropyLoss()

    def build_dataset(self, cfg):
        """构建训练数据和测试数据"""
        input_shape = eval(cfg.input_shape)
        # 加载训练数据
        train_dataset = AudioDataset(cfg.train_data,
                                     class_name=cfg.class_name,
                                     data_dir=cfg.data_dir,
                                     mode='train',
                                     spec_len=input_shape[3])
        train_loader = DataLoader(dataset=train_dataset, 
                                  batch_size=cfg.batch_size, 
                                  shuffle=True,
                                  num_workers=cfg.num_workers)
        cfg.class_name = train_dataset.class_name
        cfg.class_dict = train_dataset.class_dict
        cfg.num_classes = len(cfg.class_name)

        # 加载测试数据
        test_dataset = AudioDataset(cfg.test_data,
                                    class_name=cfg.class_name,
                                    data_dir=cfg.data_dir,
                                    mode='test',
                                    spec_len=input_shape[3])
        test_loader = DataLoader(dataset=test_dataset,              
                                 batch_size=cfg.batch_size, 
                                 shuffle=False,
                                 num_workers=cfg.num_workers)

        print("train nums:{}".format(len(train_dataset)))
        print("test  nums:{}".format(len(test_dataset)))
        return train_loader, test_loader

    def build_model(self, cfg):
        """构建模型"""
        if cfg.net_type == "mbv2":
            model = mobilenet_v2.mobilenet_v2(num_classes=cfg.num_classes)
        elif cfg.net_type == "resnet34":
            model = resnet.resnet34(num_classes=args.num_classes)
        elif cfg.net_type == "resnet18":
            model = resnet.resnet18(num_classes=args.num_classes)
        else:
            raise Exception("Error:{}".format(cfg.net_type))
        model.to(self.device)
        return model

    def epoch_test(self, epoch):
        """模型测试"""
        loss_sum = []
        accuracies = []
        self.model.eval() # model.eval()的作用是在测试时不启用 Batch Normalization 和 Dropout。在测试时，model.eval()是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变；对于Dropout，model.eval()是利用到了所有网络连接，即不进行随机舍弃神经元。
        
        with torch.no_grad(): # with torch.no_grad()主要是用于停止autograd模块的工作，以起到加速和节省显存的作用
            for step, (inputs, labels) in enumerate(tqdm(self.test_loader)):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).long()
                output = self.model(inputs)
                # 计算损失值
                loss = self.losses(output, labels)
                # 计算准确率
                output = torch.nn.functional.softmax(output, dim=1)
                # 把output中的tensor数据取出来转成numpy类型放在cpu上
                output = output.data.cpu().numpy()
                # 取出每一行中最大值的索引
                output = np.argmax(output, axis=1)
                labels = labels.data.cpu().numpy()
                acc = np.mean((output == labels).astype(int))
                accuracies.append(acc)
                loss_sum.append(loss)
        acc = sum(accuracies) / len(accuracies)
        loss = sum(loss_sum) / len(loss_sum)
        print("Test epoch:{:3.3f},Acc:{:3.3f},loss:{:3.3f}".format(epoch, acc, loss))
        print('=' * 70)
        return acc, loss

    def epoch_train(self, epoch):
        """模型训练"""
        loss_sum = []
        accuracies = []
        self.model.train()
        for step, (inputs, labels) in enumerate(tqdm(self.train_loader)):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).long()
            output = self.model(inputs)
            # 计算损失值
            loss = self.losses(output, labels)
            # 梯度归零
            self.optimizer.zero_grad()
            # 反向传播计算得到每个参数的梯度值
            loss.backward()
            # 通过梯度下降执行一步参数更新
            self.optimizer.step()

            # 计算准确率
            output = torch.nn.functional.softmax(output, dim=1)
            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            labels = labels.data.cpu().numpy()
            acc = np.mean((output == labels).astype(int))
            accuracies.append(acc)
            loss_sum.append(loss)
            if step % 50 == 0:
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                print('[%s] Train epoch %d, batch: %d/%d, loss: %f, accuracy: %f，lr:%f' % (
                    datetime.now(), epoch, step, len(self.train_loader), sum(loss_sum) / len(loss_sum),
                    sum(accuracies) / len(accuracies), lr))
        acc = sum(accuracies) / len(accuracies)
        loss = sum(loss_sum) / len(loss_sum)
        print("Train epoch:{:3.3f},Acc:{:3.3f},loss:{:3.3f}".format(epoch, acc, loss))
        print('=' * 70)
        return acc, loss

    def run(self):
        # 开始训练
        for epoch in range(self.num_epoch):
            train_acc, train_loss = self.epoch_train(epoch)
            test_acc, test_loss = self.epoch_test(epoch)
            self.tensorboard.add_scalar("train_acc", train_acc, epoch)
            self.tensorboard.add_scalar("train_loss", train_loss, epoch)
            self.tensorboard.add_scalar("test_acc", test_acc, epoch)
            self.tensorboard.add_scalar("test_loss", test_loss, epoch)
            self.scheduler.step()
            self.save_model(epoch, test_acc)

    def save_model(self, epoch, acc):
        """保持模型"""
        model_path = os.path.join(self.model_dir, 'model_{:0=3d}_{:.3f}.pth'.format(epoch, acc))
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        torch.jit.save(torch.jit.script(self.model), model_path)


def get_parser():
    data_dir = "/home/dm/data3/dataset/UrbanSound8K/audio"
    train_data = 'data/UrbanSound8K/train.txt'
    test_data = 'data/UrbanSound8K/test.txt'
    class_name = 'data/UrbanSound8K/class_name.txt'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch_size', type=int, default=32, help='训练的批量大小')
    parser.add_argument('--num_workers', type=int, default=8, help='读取数据的线程数量')
    parser.add_argument('--num_epoch', type=int, default=100, help='训练的轮数')
    parser.add_argument('--class_name', type=str, default=class_name, help='类别文件')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='初始学习率的大小')
    parser.add_argument('--input_shape', type=str, default='(None, 1, 128, 128)', help='数据输入的形状')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--net_type', type=str, default="mbv2", help='backbone')
    parser.add_argument('--data_dir', type=str, default=data_dir, help='数据路径')
    parser.add_argument('--train_data', type=str, default=train_data, help='训练数据的数据列表路径')
    parser.add_argument('--test_data', type=str, default=test_data, help='测试数据的数据列表路径')
    parser.add_argument('--work_dir', type=str, default='work_space/', help='模型保存的路径')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print_arguments(args)
    t = Train(args)
    t.run()
