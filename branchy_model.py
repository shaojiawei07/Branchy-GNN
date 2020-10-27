import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import DGCNN
from collections import OrderedDict


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature

def awgn_channel(x, noise_factor):
	x = F.normalize(x, p=2, dim = 1)
	#print(noise_factor)#print(torch.norm(x,dim = 1))
	return x + torch.randn_like(x) * noise_factor

class DGCNN_exit1(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_exit1, self).__init__()
        self.args = args
        self.DGCNN = DGCNN(args)
        dict_tmp = torch.load('./pretrained/model.1024.t7')
        new_state_dict = OrderedDict()
        #print(dict_tmp)
        for name, tensor in dict_tmp.items():
            #print(name)
            name = name[7:]
            new_state_dict[name] = tensor

        self.DGCNN.load_state_dict(new_state_dict)
        self.k = 20

        for para in self.DGCNN.parameters():
            para.requires_grad = False

        self.exit1_conv = nn.Sequential(nn.Conv1d(64, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   )
        self.exit1_fc2 = nn.Sequential(nn.Linear(512,1536),
                                   nn.BatchNorm1d(1536),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   )
        self.exit1_predict = nn.Sequential(nn.Linear(1536,512),
                                   nn.BatchNorm1d(512),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Dropout(0.5),
                                   nn.Linear(512,256),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Dropout(0.5),
                                   nn.Linear(256,128),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Dropout(0.5),
                                   nn.Linear(128,40),
                                   nn.BatchNorm1d(40),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   )


    def forward(self, x, noise_factor = 0.1):

        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k) # [batch_size, dim=3 * 2, point_num, k]
        x = self.DGCNN.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0] # [batch_size, dim = 64, point_num]
        x = x1 # do not need to concate

        #exit 1 
        x = self.exit1_conv(x)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # (batch_size, dimension)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1) # (batch_size, dimension)
        x = torch.cat((x1, x2), 1)
        x = self.exit1_fc2(x)

        #awgn channel model
        #x = awgn_channel(x,0.1) # 20dB
        x = awgn_channel(x,self.args.channel_noise)

        x = self.exit1_predict(x)
        return x

class DGCNN_exit2(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_exit2, self).__init__()
        self.args = args
        self.DGCNN = DGCNN(args)
        dict_tmp = torch.load('./pretrained/model.1024.t7')
        new_state_dict = OrderedDict()
        #print(dict_tmp)
        for name, tensor in dict_tmp.items():
            #print(name)
            name = name[7:]
            new_state_dict[name] = tensor

        self.DGCNN.load_state_dict(new_state_dict)
        self.k = 20

        for para in self.DGCNN.parameters():
            para.requires_grad = False

        self.exit2_conv = nn.Sequential(nn.Conv1d(128, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   )
        self.exit2_fc2 = nn.Sequential(nn.Linear(512,1024),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   )
        self.exit2_predict = nn.Sequential(nn.Linear(1024,512),
                                   nn.BatchNorm1d(512),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Dropout(0.5),
                                   nn.Linear(512,256),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Dropout(0.5),
                                   nn.Linear(256,128),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Dropout(0.5),
                                   nn.Linear(128,40),
                                   nn.BatchNorm1d(40),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   )


    def forward(self, x, noise_factor = 0.1):

        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k) # [batch_size, dim=3 * 2, point_num, k]
        x = self.DGCNN.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0] # [batch_size, dim = 64, point_num]
        x = get_graph_feature(x1, k=self.k)
        x = self.DGCNN.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0] # [batch_size, dim = 64, point_num]

        #exit 2    
        x = torch.cat((x1, x2), dim=1) # [batch_size, dim =128, point_num]
        x = self.exit2_conv(x)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # (batch_size, dimension)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1) # (batch_size, dimension)
        x = torch.cat((x1, x2), 1)
        x = self.exit2_fc2(x)

        #awgn channel model
        #x = awgn_channel(x,0.1) # 20dB
        x = awgn_channel(x,self.args.channel_noise)

        x = self.exit2_predict(x)
        return x

class DGCNN_exit3(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_exit3, self).__init__()
        self.args = args
        self.DGCNN = DGCNN(args)
        dict_tmp = torch.load('./pretrained/model.1024.t7')
        new_state_dict = OrderedDict()
        #print(dict_tmp)
        for name, tensor in dict_tmp.items():
            #print(name)
            name = name[7:]
            new_state_dict[name] = tensor
            #print(name)
        #self.DGCNN.load_state_dict(torch.load('./pretrained/model.1024.t7'))
        self.DGCNN.load_state_dict(new_state_dict)
        self.k = 20

        for para in self.DGCNN.parameters():
            para.requires_grad = False

        self.exit3_conv = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   )
        self.exit3_fc2 = nn.Sequential(nn.Linear(512,512),
                                   nn.BatchNorm1d(512),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   )
        self.exit3_predict = nn.Sequential(nn.Linear(512,512),
                                   nn.BatchNorm1d(512),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Dropout(0.5),
                                   nn.Linear(512,256),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Dropout(0.5),
                                   nn.Linear(256,128),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Dropout(0.5),
                                   nn.Linear(128,40),
                                   nn.BatchNorm1d(40),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   )


    def forward(self, x, noise_factor = 0.1):

        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k) # [batch_size, dim=3 * 2, point_num, k]
        x = self.DGCNN.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0] # [batch_size, dim = 64, point_num]
        x = get_graph_feature(x1, k=self.k)
        x = self.DGCNN.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0] # [batch_size, dim = 64, point_num]
        x = get_graph_feature(x2, k=self.k)
        x = self.DGCNN.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0] # [batch_size, dim = 128, point_num]

        #exit 3    
        x = torch.cat((x1, x2, x3), dim=1) # [batch_size, dim =256, point_num]
        x = self.exit3_conv(x)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # (batch_size, dimension)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1) # (batch_size, dimension)
        x = torch.cat((x1, x2), 1)
        x = self.exit3_fc2(x)

        #awgn channel model
        #x = awgn_channel(x,0.1) # 20dB
        x = awgn_channel(x,self.args.channel_noise)




        x = self.exit3_predict(x)
        return x

class DGCNN_exit4(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_exit4, self).__init__()
        self.args = args
        self.DGCNN = DGCNN(args)
        dict_tmp = torch.load('./pretrained/model.1024.t7')
        new_state_dict = OrderedDict()
        #print(dict_tmp)
        for name, tensor in dict_tmp.items():
            #print(name)
            name = name[7:]
            new_state_dict[name] = tensor
            #print(name)
        #self.DGCNN.load_state_dict(torch.load('./pretrained/model.1024.t7'))
        self.DGCNN.load_state_dict(new_state_dict)
        self.k = 20

        for para in self.DGCNN.parameters():
            para.requires_grad = False


        self.exit4_fc2 = nn.Sequential(nn.Linear(2048,128),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   )
        self.exit4_predict = nn.Sequential(nn.Linear(128,512),
                                   nn.BatchNorm1d(512),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Dropout(0.5),
                                   nn.Linear(512,256),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Dropout(0.5),
                                   nn.Linear(256,128),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Dropout(0.5),
                                   nn.Linear(128,40),
                                   nn.BatchNorm1d(40),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   )


    def forward(self, x, noise_factor = 0.1):

        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k) # [batch_size, dim=3 * 2, point_num, k]
        x = self.DGCNN.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0] # [batch_size, dim = 64, point_num]
        x = get_graph_feature(x1, k=self.k)
        x = self.DGCNN.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0] # [batch_size, dim = 64, point_num]
        x = get_graph_feature(x2, k=self.k)
        x = self.DGCNN.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0] # [batch_size, dim = 128, point_num]
        x = get_graph_feature(x3, k=self.k)
        x = self.DGCNN.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.DGCNN.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = self.exit4_fc2(x)


        #awgn channel model
        #x = awgn_channel(x,0.1) # 20dB
        x = awgn_channel(x,self.args.channel_noise)

        x = self.exit4_predict(x)
        return x


