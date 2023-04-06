# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class CollaborativeFilteringModel(nn.Module):
    #num_users表示用户数，num_items表示电影数，num_factors表示隐向量维度(神经网络内容，这里不赘述
    def __init__(self, num_users, num_items, num_factors):
        super().__init__()
        #用户和电影的嵌入，转为num_factors维向量
        self.user_factors = nn.Embedding(num_users, num_factors)
        self.item_factors = nn.Embedding(num_items, num_factors)
        #使用全链接层对用户和电影组合，输出单一评分值，此处预测的是用户对于未评分影片的评分
        self.fc = nn.Linear(num_factors, 1)

    #前向传播，神经网络是多层的，数据按着一定运算方式计算，每一层结果都会作为下一层的输入值
    def forward(self, user, item):
        user_embedding = self.user_factors(user)
        item_embedding = self.item_factors(item)
        interaction = user_embedding * item_embedding
        rating = self.fc(interaction)
        return rating
