#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from utils.dp_mechanism import cal_sensitivity, cal_sensitivity_MA, Laplace, Gaussian_Simple, Gaussian_MA
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdateDP(object):
    def __init__(self, args, dataset=None, idxs=None):
        # 初始化本地更新配置
        self.args = args  # 保存训练参数
        self.loss_func = nn.CrossEntropyLoss()  # 设置损失函数为交叉熵，适用于分类任务
        # 从参与的全部索引中随机选择一部分进行训练，比例由 dp_sample 决定
        self.idxs_sample = np.random.choice(list(idxs), int(self.args.dp_sample * len(idxs)), replace=False)
        # 创建 DataLoader，用于加载训练数据
        self.ldr_train = DataLoader(DatasetSplit(dataset, self.idxs_sample), batch_size=len(self.idxs_sample),
                                    shuffle=True)
        self.idxs = idxs  # 保存所有可能的训练数据索引
        self.times = self.args.epochs * self.args.frac  # 计算总的训练次数
        self.lr = args.lr  # 设置学习率
        self.noise_scale = self.calculate_noise_scale()  # 计算噪声规模，依据选择的DP机制不同而不同


    def calculate_noise_scale(self):
        # 根据不同的差分隐私机制计算噪声规模
        if self.args.dp_mechanism == 'Laplace':
            epsilon_single_query = self.args.dp_epsilon / self.times
            return Laplace(epsilon=epsilon_single_query)
        elif self.args.dp_mechanism == 'Gaussian':
            epsilon_single_query = self.args.dp_epsilon / self.times
            delta_single_query = self.args.dp_delta / self.times
            return Gaussian_Simple(epsilon=epsilon_single_query, delta=delta_single_query)
        elif self.args.dp_mechanism == 'MA':
            return Gaussian_MA(epsilon=self.args.dp_epsilon, delta=self.args.dp_delta, q=self.args.dp_sample, epoch=self.times)

    def train(self, net, iter, epochs):
        # 训练模型
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr)  # 使用SGD优化器
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay)  # 设置学习率衰减
        loss_client = 0

        for images, labels in self.ldr_train:
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            net.zero_grad()
            log_probs = net(images)
            loss = self.loss_func(log_probs, labels)
            loss.backward()
            if self.args.dp_mechanism != 'no_dp':
                self.clip_gradients(net)  # 应用梯度裁剪
            optimizer.step()
            scheduler.step()
            if self.args.dp_mechanism != 'no_dp':
                self.noise_scale = self.calculate_noise_scale()
                self.add_noise(net)  # 向参数添加噪声
                # print("Noise scale: {:.5f},Epsilon: {:.5f},Delta: {:.5f},s".format(self.noise_scale, self.args.dp_epsilon, self.args.dp_delta))
            loss_client = loss.item()
        self.lr = scheduler.get_last_lr()[0]  # 更新学习率
        return net.state_dict(), loss_client

    def clip_gradients(self, net):
        # 梯度裁剪，用于控制梯度的敏感度
        if self.args.dp_mechanism == 'Laplace':
            # Laplace use 1 norm
            self.per_sample_clip(net, self.args.dp_clip, norm=1)
        elif self.args.dp_mechanism == 'Gaussian' or self.args.dp_mechanism == 'MA':
            # Gaussian use 2 norm
            self.per_sample_clip(net, self.args.dp_clip, norm=2)

    def per_sample_clip(self, net, clipping, norm):
        # 对每个样本的梯度进行裁剪，以限制梯度的范数，防止过大的梯度泄露过多的个人信息
        grad_samples = [x.grad_sample for x in net.parameters()]  # 提取每个参数的样本梯度
        per_param_norms = [
            g.reshape(len(g), -1).norm(norm, dim=-1) for g in grad_samples  # 计算每个样本梯度的范数
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(norm, dim=1)  # 跨参数计算每个样本的总范数
        per_sample_clip_factor = (
            torch.div(clipping, (per_sample_norms + 1e-6))  # 计算裁剪因子，防止除以零
        ).clamp(max=1.0)  # 将裁剪因子限制在1以内
        for grad in grad_samples:
            factor = per_sample_clip_factor.reshape(per_sample_clip_factor.shape + (1,) * (grad.dim() - 1))
            grad.detach().mul_(factor.to(grad.device))  # 应用裁剪因子到梯度
        # 平均裁剪后的梯度并替换原始梯度
        for param in net.parameters():
            param.grad = param.grad_sample.detach().mean(dim=0)

    def add_noise(self, net):
        # 根据差分隐私机制和设置的敏感度计算并添加噪声
        sensitivity = cal_sensitivity(self.lr, self.args.dp_clip, len(self.idxs_sample))  # 计算敏感度
        state_dict = net.state_dict()  # 获取模型的状态字典
        if self.args.dp_mechanism == 'Laplace':
            # 如果使用拉普拉斯机制
            for k, v in state_dict.items():
                state_dict[k] += torch.from_numpy(np.random.laplace(loc=0, scale=sensitivity * self.noise_scale,
                                                                    size=v.shape)).to(self.args.device)
        elif self.args.dp_mechanism == 'Gaussian':
            # 如果使用高斯机制
            for k, v in state_dict.items():
                state_dict[k] += torch.from_numpy(np.random.normal(loc=0, scale=sensitivity * self.noise_scale,
                                                                   size=v.shape)).to(self.args.device)
        elif self.args.dp_mechanism == 'MA':
            # 如果使用高斯机制的变体，如Momentum Accountant
            sensitivity = cal_sensitivity_MA(self.args.lr, self.args.dp_clip, len(self.idxs_sample))
            for k, v in state_dict.items():
                state_dict[k] += torch.from_numpy(np.random.normal(loc=0, scale=sensitivity * self.noise_scale,
                                                                   size=v.shape)).to(self.args.device)
        net.load_state_dict(state_dict)  # 将添加了噪声的参数加载回模型


class LocalUpdateDPSerial(LocalUpdateDP):
    def __init__(self, args, dataset=None, idxs=None):
        # 调用父类构造函数进行初始化
        super().__init__(args, dataset, idxs)

    def train(self, net, iter, epochs):
        net.train()  # 设置模型为训练模式
        # 初始化优化器和学习率调度器
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=self.args.momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay)

        losses = 0  # 初始化损失统计

        # 动态调整参数
        print("epsilon: ", self.args.dp_epsilon)
        print("delta: ", self.args.dp_delta)

        # 遍历数据加载器中的数据
        for images, labels in self.ldr_train:
            net.zero_grad()  # 清除历史梯度
            index = int(len(images) / self.args.serial_bs)  # 计算批次的划分数量
            total_grads = [torch.zeros(size=param.shape).to(self.args.device) for param in net.parameters()]

            # 对每个批次进行独立的前向和反向传播
            for i in range(0, index + 1):
                net.zero_grad()  # 每个小批次开始时清除梯度
                start = i * self.args.serial_bs
                end = (i+1) * self.args.serial_bs if (i+1) * self.args.serial_bs < len(images) else len(images)
                if start == end:
                    break
                image_serial_batch, labels_serial_batch = images[start:end].to(self.args.device), labels[start:end].to(self.args.device)
                log_probs = net(image_serial_batch)
                loss = self.loss_func(log_probs, labels_serial_batch)
                loss.backward()
                if self.args.dp_mechanism != 'no_dp':
                    self.clip_gradients(net)  # 应用梯度裁剪
                grads = [param.grad.detach().clone() for param in net.parameters()]
                for idx, grad in enumerate(grads):
                    total_grads[idx] += torch.mul(torch.div((end - start), len(images)), grad)
                losses += loss.item() * (end - start)
            # 更新总梯度
            for i, param in enumerate(net.parameters()):
                param.grad = total_grads[i]
            optimizer.step()  # 更新模型参数
            scheduler.step()  # 更新学习率
            if self.args.dp_mechanism != 'no_dp':
                self.add_noise(net)  # 在参数中添加噪声
            self.lr = scheduler.get_last_lr()[0]
        # 返回更新后的模型状态和平均损失
        return net.state_dict(), losses / len(self.idxs_sample)
