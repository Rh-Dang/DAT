from __future__ import division
import sys
sys.path.append('/data_sdd/datadrh/HOZ/models')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils.model_util import norm_col_init, weights_init

from model_io import ModelOutput

import scipy.sparse as sp
import numpy as np
import scipy.io as scio
import os
import json
import copy
import math
from sklearn.cluster import KMeans

object_list = ['AlarmClock', 'Book', 'Bowl', 'CellPhone', 'Chair', 'CoffeeMachine', 'DeskLamp', 'FloorLamp',
'Fridge', 'GarbageCan', 'Kettle', 'Laptop', 'LightSwitch', 'Microwave', 'Pan', 'Plate', 'Pot',
'RemoteControl', 'Sink', 'StoveBurner', 'Television', 'Toaster',]

class Target_Aware_Self_Attention_Layer(nn.Module):     #多头注意力

    def __init__(self,
                 hidden_dim,
                 C_in=None,
                 num_heads=1,                   #头数：需要能够整除特征维度
                 dropout_rate=0.0,
                 length_out = 0):
        super(Target_Aware_Self_Attention_Layer, self).__init__()
        self.length_out = length_out
        self.hidden_dim = hidden_dim
        self.norm_mixer = nn.LayerNorm(C_in)
        self.linear_V = nn.Linear(C_in, num_heads * hidden_dim)
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout_rate)
        self.V_dynamic_0 = nn.Sequential(
            nn.Linear(hidden_dim, self.length_out),
            nn.ReLU(),
        )
        self.V_dynamic_1 = nn.Sequential(
            nn.Conv1d(hidden_dim, self.length_out, kernel_size=3, bias=True,dilation=1),
            nn.ReLU(),
        )
        self.V_dynamic_2 = nn.Sequential(
            nn.Conv1d(hidden_dim, self.length_out, kernel_size=3, bias=True,dilation=2),
            nn.ReLU(),
        )

        self.head_communicate = nn.Linear(num_heads, 1)

    def forward(self, V):
        """
        :param Q: A 3d tensor with shape of [N, T_q, C_q]   他人的特征表示
        :param K: A 3d tensor with shape of [N, T_k, C_k]   自己的特征表示
        :param V: A 3d tensor with shape of [N, T_v, C_v]   需要attention的特征表示
        :return:
        """
        num_heads = self.num_heads
        N = 1                                           #batch
        length_in = V.shape[0]                          #n
        V = V.unsqueeze(0)                              #拓展一维batch   (1, n, dim)
        
        V = self.norm_mixer(V)
        
        V_l = nn.ReLU()(self.linear_V(V))        #(1, n, num_heads * hidden_dim)
        
        V_split = V_l.split(split_size=self.hidden_dim, dim=2)     
  
        V_ = torch.cat(V_split, dim=0)  # (num_heads, n, hidden_dim) 
        
        # print(V_.shape)       # (3,27,32)
        # print(V_[1,length_in - 1, :].shape)#
        # print(V_[1,:,:].shape)
        # print(V_[1,0, :].shape)
        V_0 = V_[0,:,:]
        V_1 = torch.cat((V_[1,length_in - 1, :].unsqueeze(0),V_[1,:,:],V_[1,0, :].unsqueeze(0)), dim =0)   #(n+2, hidden_dim)
        V_2 = torch.cat((V_[2,length_in - 2, :].unsqueeze(0),V_[2,length_in - 1, :].unsqueeze(0),V_[2,:,:],V_[2,0, :].unsqueeze(0),V_[2,1, :].unsqueeze(0)), dim =0)  #(n+4, hidden_dim)

        shorten_kernel_0 = self.V_dynamic_0(V_0.unsqueeze(0)).permute(0,2,1)   #(1, k, n)
        shorten_kernel_1 = self.V_dynamic_1(V_1.unsqueeze(0).permute(0,2,1))    #(1, k, n)
        shorten_kernel_2 = self.V_dynamic_2(V_2.unsqueeze(0).permute(0,2,1))    #(1, k, n)

        shorten_kernel = torch.cat((shorten_kernel_0, shorten_kernel_1, shorten_kernel_2), dim = 0)  #(3, k, n)
     
        outputs = torch.bmm(shorten_kernel, V_)  # (num_heads, k, n)  *  (num_heads, n, hidden_dim) = (num_heads, k, hidden_dim)

        # V_origin = torch.sum(V_origin, dim = 1)
        # a = torch.nonzero(V_origin).squeeze()
        # a = torch.max(a).cpu().detach()

        # outputs_out = np.array(shorten_kernel.cpu().detach())
        # for i in range(self.num_heads):
        #     if V.shape[-1] == 16:
        #         np.savetxt('visualization_files/navgraph/navgraph_navigation/pyramid/pyramid_wotrans_head4/concentrated-attention_step' + str(a) +'layer1'+'head'+str(i)+'.txt', outputs_out[i])
        #     if V.shape[-1] == 32:
        #         np.savetxt('visualization_files/navgraph/navgraph_navigation/pyramid/pyramid_wotrans_head4/concentrated-attention_step' + str(a) +'layer2'+'head'+str(i)+'.txt', outputs_out[i])
        
        # Restore shape
        outputs = outputs.transpose(0,2).transpose(0,1)    #(k, hidden_dim, num_heads)
        outputs = self.head_communicate(outputs).transpose(0,2).transpose(1,2)   #(1, k, hidden_dim)

        # outputs = outputs.split(N, dim=0)  # (N, T_q, C)  是一个tuple
        # outputs = torch.cat(outputs, dim=2)

        # #Residual connection
        # outputs = outputs + V_l

        # Normalize

        outputs = outputs.squeeze(0)   #将batch维度消去

        return outputs   

class Target_Aware_Self_Attention_Block(nn.Module):
    def __init__(self, 
                C_out,
                C_in,
                num_heads = 1,
                dropout = 0,
                length_out = 0):
        super(Target_Aware_Self_Attention_Block, self).__init__()
        self.length_out = length_out
        self.attention = Target_Aware_Self_Attention_Layer(C_out, C_in, num_heads, dropout, self.length_out)
        self.target_attention = nn.Sequential(
            nn.Linear(64, C_out),
            nn.ReLU(),
            nn.Linear(C_out, C_out),
            nn.ReLU(),
        )

    # def kmeans(self, x, ncluster, niter=10):
    #     '''
    #     x : torch.tensor(data_num,data_dim)
    #     ncluster : The number of clustering for data_num
    #     niter : Number of iterations for kmeans
    #     '''
    #     N, D = x.size()
    #     print(N)
    #     print(ncluster)
    #     c = x[torch.randperm(N)[:ncluster]] # init clusters at random
    #     for i in range(niter):
    #         # assign all pixels to the closest codebook element
    #         # .argmin(1) : 按列取最小值的下标,下面这行的意思是将x.size(0)个数据点归类到random选出的ncluster类
    #         a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1)
    #         # move each codebook element to be the mean of the pixels that assigned to it
    #         # 计算每一类的迭代中心，然后重新把第一轮随机选出的聚类中心移到这一类的中心处
    #         c = torch.stack([x[a==k].mean(0) for k in range(ncluster)])
    #         # re-assign any poorly positioned codebook elements
    #         nanix = torch.any(torch.isnan(c), dim=1)
    #         ndead = nanix.sum().item()
    #         # print('done step %d/%d, re-initialized %d dead clusters' % (i+1, niter, ndead))
    #         c[nanix] = x[torch.randperm(N)[:ndead]] # re-init dead clusters
    #     return c

    def forward(self, input, target):
        
        input = self.attention(input)   #先self - attention  
        # input_kmeans = input.detach().cpu().numpy()
        # kmeans = KMeans(n_clusters=self.length_out, random_state=0, n_jobs=-1).fit(input_kmeans)
        # centers = torch.tensor(kmeans.cluster_centers_).to(input.device)

        target_attention = self.target_attention(target)
        output = input * target_attention
        return output


class Navigation_Graph_Embedding(nn.Module):
    def __init__(self,
                hidden_dim,
                C_in,
                num_heads = 1,
                dropout = 0,
                nav_length = 27):
        super(Navigation_Graph_Embedding, self).__init__()
        self.graph = torch.zeros(nav_length, C_in)
        self.self_attention_1 = Target_Aware_Self_Attention_Block(hidden_dim, C_in, num_heads, dropout, length_out = int(nav_length/9))
        # self.self_attention_2 = Target_Aware_Self_Attention_Block(hidden_dim, hidden_dim, num_heads, dropout, length_out = int(nav_length/9))
        self.outlinear = nn.Sequential(
            nn.Linear(3*hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.layer1_pool = nn.AdaptiveAvgPool1d(1)
        self.layer2_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, input, target):    #input (n, 16)
        
        graph = self.graph.to(target.device)
        graph[:input.shape[0], :] = input    #将input嵌入到一个固定长度的0向量中
        graph = self.self_attention_1(graph, target)   #(3, hidden_dim)
        output_layer1 = self.layer1_pool(graph.t().unsqueeze(0)).squeeze(0).t()
        # graph = self.self_attention_2(graph, target)    #(3, hidden_dim)
        # output_layer2 = self.layer2_pool(graph.t().unsqueeze(0)).squeeze(0).t()
        graph = graph.split(split_size=1, dim=0)
        output = torch.cat(graph, dim=1)         #（1, 3*hidden_dim)
        output = self.outlinear(output)
        output = output + output_layer1

        return output   #(1,hidden_dim)

class Multihead_Attention(nn.Module):     #多头注意力
    """
    multihead_attention
    """

    def __init__(self,
                 hidden_dim,
                 C_q=None,
                 C_k=None,
                 num_heads=1,                   #头数：需要能够整除特征维度
                 dropout_rate=0.0):
        super(Multihead_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        C_q = C_q if C_q else hidden_dim
        C_k = C_k if C_k else hidden_dim
        self.linear_Q = nn.Linear(C_q, hidden_dim)   
        self.linear_K = nn.Linear(C_k, hidden_dim)
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear_out = nn.Linear(num_heads, 1)

    def forward(self,
                Q, K):
        """
        :param Q: A 3d tensor with shape of [T_q, C_q]   他人的特征表示
        :param K: A 3d tensor with shape of [T_k, C_k]   自己的特征表示
        :param V: A 3d tensor with shape of [T_v, C_v]   需要attention的特征表示
        :return:
        """
        num_heads = self.num_heads
        N = 1                                           #batch
        Q = Q.unsqueeze(dim = 0)             #将batch维度扩展出来
        K = K.unsqueeze(dim = 0)

        # Linear projections
        Q_l = nn.ReLU()(self.linear_Q(Q))                         #先将特征映射到同一维度
        K_l = nn.ReLU()(self.linear_K(K))

        # Split and concat
        Q_split = Q_l.split(split_size=self.hidden_dim // num_heads, dim=2)  #将不同维度的特征图分配到不同的注意力头
        K_split = K_l.split(split_size=self.hidden_dim // num_heads, dim=2)

        Q_ = torch.cat(Q_split, dim=0)  # (h*N, T_q, C/h)                    #将特征图分开之后加到batch中，想到那个batch是并行的，所以加到batch上也就是实现了并行
        K_ = torch.cat(K_split, dim=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = torch.bmm(Q_, K_.transpose(2, 1))    #(h*N, T_q(1), T_k)

        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)   # /根号dk的操作

        # Dropouts   
        outputs = self.dropout(outputs) 
        outputs = outputs.split(N, dim=0)
        outputs = torch.cat(outputs, dim=1)  #(1, num_heads, num_point)
        outputs = outputs.transpose(1,2)     #(1, num_point, num_heads)
        outputs = self.linear_out(outputs)   ##(1, num_point, 1)
        outputs = nn.Softmax(dim=1)(outputs).squeeze(dim=0)

        # # Residual connection
        # outputs = outputs + Q_l

        # # Normalize
        # outputs = self.norm(outputs)  # (N, T_q, C)

        return outputs   #输出的是  注意力向量 (22,1)


class Pyramid_Wotrans_Head4_Ms_Headms_Nomlp_One(torch.nn.Module):
    def __init__(self, args):
        action_space = args.action_space
        self.num_cate = args.num_category
        resnet_embedding_sz = args.hidden_state_sz
        hidden_state_sz = args.hidden_state_sz
        super(Pyramid_Wotrans_Head4_Ms_Headms_Nomlp_One, self).__init__()

        self.image_size = 300
        self.conv1 = nn.Conv2d(resnet_embedding_sz, 64, 1)

        self.action_at_a = nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), requires_grad=False)
        self.action_at_b = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]), requires_grad=False)
        self.action_at_scale = nn.Parameter(torch.tensor(0.58), requires_grad=False) #平衡但单独输出的停止动作影响参数

        self.graph_detection_feature = nn.Sequential(
            nn.Linear(262, 128),
            nn.ReLU(),
            nn.Linear(128, 49),
        )

        self.embed_action = nn.Linear(action_space, 10)

        self.nav_embedding_dim = 32
        pointwise_in_channels = 64 + self.num_cate + 10 + self.nav_embedding_dim

        self.pointwise = nn.Conv2d(pointwise_in_channels, 64, 1, 1)

        self.lstm_input_sz = 7 * 7 * 64

        self.hidden_state_sz = hidden_state_sz
        self.lstm = nn.LSTM(self.lstm_input_sz, hidden_state_sz, 2)
        num_outputs = action_space
        self.critic_linear_1 = nn.Linear(hidden_state_sz, 64)
        self.critic_linear_2 = nn.Linear(64, 1)
        self.actor_linear = nn.Linear(hidden_state_sz, num_outputs)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.conv1.weight.data.mul_(relu_gain)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)

        self.critic_linear_1.weight.data = norm_col_init(
            self.critic_linear_1.weight.data, 1.0
        )
        self.critic_linear_1.bias.data.fill_(0)
        self.critic_linear_2.weight.data = norm_col_init(
            self.critic_linear_2.weight.data, 1.0
        )
        self.critic_linear_2.bias.data.fill_(0)

        self.lstm.bias_ih_l0.data.fill_(0)
        self.lstm.bias_ih_l1.data.fill_(0)
        self.lstm.bias_hh_l0.data.fill_(0)
        self.lstm.bias_hh_l1.data.fill_(0)
        self.dropout_rate = 0.35
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.info_embedding = nn.Linear(5,49)

        
        ###定义一个可学习数组，每一列表示在寻找这类物品时每个物品的重要程度（即与寻找物品的相关程度）
        self.target_object_attention = torch.nn.Parameter(torch.FloatTensor(self.num_cate, self.num_cate), requires_grad=True)
        self.target_object_attention.data.fill_(1/22)   #初始化的时候是归一化的

        #四个场景中对于物体注意力的调整
        self.scene_object_attention = torch.nn.Parameter(torch.FloatTensor(4, self.num_cate, self.num_cate), requires_grad=True)
        self.scene_object_attention.data.fill_(1/22)   #初始化的时候是归一化的

        #两种注意力的占比
        self.attention_weight = torch.nn.Parameter(torch.FloatTensor(2), requires_grad=True)
        self.attention_weight.data.fill_(1/2)


        #用于将图片特征变成对于物体的注意力
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))    #(64,7,7) -> (64,1,1)
        # self.image_to_attent = nn.Linear(64,22)

        #image_attention_object分支中多头注意力
        self.muti_head_attention = Multihead_Attention(hidden_dim = 512, C_q = resnet_embedding_sz + 64, C_k = 262, num_heads = 8, dropout_rate = 0.3)
        self.conf_threshod = 0.6
        #一定要保证key_size（hidden_dim/num_heads)不能太小，因为太小的key_size学习能力就太弱了

        self.num_cate_embed = nn.Sequential(
            nn.Linear(self.num_cate, 32),  #对于物体类别标号的编码
            nn.ReLU(),
            nn.Linear(32, 64),  
            nn.ReLU(),
        )

        # #是否检测到target
        # self.detect_target = 0   #0表示没有检测到
        self.nav_dim = args.nav_dim  
        self.nav_length = args.nav_length

        self.linear_graph = nn.Sequential(
            nn.Linear(self.nav_dim + 2, 16),  
            nn.ReLU(),
        )

        self.coord_dim = 6
    
        self.norm  = nn.LayerNorm(pointwise_in_channels)
        self.target_attent_coord = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.navgraph_embedding = Navigation_Graph_Embedding(hidden_dim = self.nav_embedding_dim, C_in = 16, num_heads = 3, dropout = 0, nav_length = self.nav_length)


    def one_hot(self, spa):  #spa是要编码的序列长度  tem是被赋予的

        y = torch.arange(spa).unsqueeze(-1)   #unsqueeze(-1)应该是在最后加一维，即将横向排列的变成竖向排列
        y_onehot = torch.FloatTensor(spa, spa)  #创建一个n*n维的矩阵

        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)    #将第一维（列） 以y的编码序号 分散到y_onehot中，并且都填充1

        return y_onehot   ## (22,22)

    def get_coord(self, coord, start_coord):   #没有以其实位置为中心
        
        if start_coord['horizon'] == -10:   #因为初始化时所有都幅值为-10
            start_coord['x'] = coord.x
            start_coord['y'] = coord.y
            start_coord['rotation'] = coord.rotation * math.pi/180
            start_coord['horizon'] = coord.horizon * math.pi/180

        x = coord.x - start_coord['x']
        y = coord.y - start_coord['y']
        rotate = coord.rotation * math.pi/180 - start_coord['rotation']
        horizon = coord.horizon * math.pi/180  #因为以自我为中心俯仰角是可识别的
      
        coord = torch.tensor([x, y, rotate, horizon])
        return coord, start_coord
    
    def coord_centring(self, navgraph, coord):
        navgraph_copy = torch.zeros(navgraph.shape[0], self.nav_dim + 2).to(navgraph.device)
        nav_coord = navgraph[:,5: 7]
        rotation = navgraph[:,7]
        horizon = navgraph[:,8]   

        navgraph_copy[:,: 5] = navgraph[:,: 5]         #bbox at
        navgraph_copy[:,5: 7] = nav_coord - coord[:2]  
        navgraph_copy[:,7] = torch.sin(rotation - coord[2])
        navgraph_copy[:,8] = torch.cos(rotation - coord[2])
        navgraph_copy[:,9] = torch.sin(horizon - coord[3])
        navgraph_copy[:,10] = torch.cos(horizon - coord[3])

        return navgraph_copy


    def embedding(self, state, target, action_embedding_input, target_object, nav_graph, coord, start_coord):

        #################################计算stop action prob(独立的用是否检测到物体判断是否应该停止)#########################
        at_v = torch.mul(target['scores'].unsqueeze(dim=1), target['indicator']) #将detection confidence和target object相乘
        at = torch.mul(torch.max(at_v), self.action_at_scale)  #将目标物体被检测到的置信度取出来，乘上0.6（单独训练的停止动作对总动作输出的影响）
        action_at = torch.mul(at, self.action_at_a) + self.action_at_b  #变成动作类别输出的形式，便于与actor_out相乘
        #################################################################################################################
        
        target_object = target['indicator']                    #目标物体的独热编码
        
        action_embedding = F.relu(self.embed_action(action_embedding_input)) 
        action_reshaped = action_embedding.view(1, 10, 1, 1).repeat(1, 1, 7, 7)  #对上一帧动作的编码

        image_embedding = F.relu(self.conv1(state))  #将resnet18编码后的512维张量-64维

        x = self.dropout(image_embedding)

        ################################计算object embedding(利用物体级别的图和视觉特征输出物体级别的特征表示)#######################
        target_appear = target['features']
        target_conf = target['scores'].unsqueeze(dim=1)
        target_bbox = target['bboxes'] / self.image_size

        target = torch.cat((target_appear, target_bbox, target_conf, target_object), dim=1)  #视觉特征256| 检测框4 | 置信度1 | 目标物体1 = 262

        target_object_attention = F.softmax(self.target_object_attention, 0)        #对基础物体权重进行归一化
       
        
        attention_weight = F.softmax(self.attention_weight, 0)    #将三个部分占比的权重归一化
       
    
        object_attention = target_object_attention * attention_weight[0]
        
        ##利用muti_head attention计算图像对于物体的注意力
        
        object_select = torch.sign(target_conf - 0.6)  #(22,1)
        object_select[object_select > 0] = 0                       
        object_select[object_select < 0] = - object_select[object_select < 0]   #(1,22)       #选取的物体置为0，不选取的置为1
        object_select_appear = object_select.squeeze().expand(262, 22).bool()           #不选取的物体列全部为True
        target_mutiHead = target.masked_fill(object_select_appear.t(),0)             #将不选取的物体全部填0

        image_object_attention = self.avgpool(state).squeeze(dim = 2).squeeze(dim = 0).t()   #(1,512)  
        spa = self.one_hot(self.num_cate).to(target.device)      #多卡训练的时候不能用.cuda() 要用.to(device) （22,22）
        num_cate_index = torch.mm(spa.t(), target_object).t()
        num_cate_index = self.num_cate_embed(num_cate_index)   #64维的编码后物体标号数据  （22,64）
        image_object_attention = torch.cat((image_object_attention, num_cate_index), dim = 1)  #(1,512+64=576)
        image_object_attention = self.muti_head_attention(image_object_attention, target_mutiHead)

        target_attention= torch.mm(object_attention, target_object)   #将目标物体的物体重要性向量取出
        target_attention = target_attention + image_object_attention * attention_weight[1]  #最终的权重向量依然是归一化的
        
        target = F.relu(self.graph_detection_feature(target))    #518维-128维-49维  N*49
        target = target * target_attention                        #对物体特征进行注意力
        target_embedding = target.reshape(1, self.num_cate, 7, 7)    #变成 1*N*7*7
        target_embedding = self.dropout(target_embedding)  
        ##############################################################################################################################
        #############################################  nav_graph操作 ################################################

        coord, start_coord = self.get_coord(coord, start_coord)
        coord = coord.to(target.device)      #6
        
        target_bbox = torch.mm(target_bbox.t(), target_object).t()      #(1,4)
        #之后还要添加比如位置等信息
        nav_node = torch.zeros(1,self.nav_dim).to(target.device)    #需要改初始化
        nav_node[:,:4] = target_bbox   # dim 0 -3 bbox 
        nav_node[:,4] = at             # dim 4  conf 
        nav_node[:,5: 5 + 4] = coord  # dim 5 - 6 coord  

        if at/self.action_at_scale > 0.4:   #只保存检测到目标物体的帧，就会导致在前期没有检测到物体时navgraph是等于0的
            if nav_graph.sum() == 0:   #如果是走的第一个点   (n, 22, dim)
                nav_graph = nav_node
            else:
                if nav_graph.shape[0] < self.nav_length:    #保证nav_graph的长度不超过27步
                    nav_graph = torch.cat((nav_graph, nav_node), dim = 0)
                else:
                    nav_graph = torch.cat((nav_graph[1:, :], nav_node), dim = 0)


        if nav_graph.sum() == 0:     #graph中没有东西的时候填0
            nav_graph_mean = torch.zeros(1 ,self.nav_embedding_dim, 7, 7).to(target.device)    # (1, 32)
        else:
            nav_graph_center = self.coord_centring(nav_graph, coord)  #将坐标变成以自身为中心的 (n, 11)
            
            nav_graph_embeded = self.linear_graph(nav_graph_center)  #  (n, 11) -> (n, 16)

            nav_graph_embeded = self.navgraph_embedding(nav_graph_embeded, num_cate_index)   #(n, 16) -> (1, 32)
                
            nav_graph_mean = nav_graph_embeded.view(1, self.nav_embedding_dim, 1, 1).repeat(1, 1, 7, 7)  #  (1, 32, 7, 7)
            nav_graph_mean = self.dropout(nav_graph_mean)
        #coord_embedding = coord.view(1,  self.coord_dim, 1, 1).repeat(1, 1, 7, 7)
        x = torch.cat((x, target_embedding, action_reshaped, nav_graph_mean), dim=1)
        # x: 原始图片编码 64*7*7                          64是每一个像素的通道特征    7*7是图片大小
        # target_embedding：物体层编码  22*7*7            22是每一个物体类别          7*7是每一个物体类别的特征向量
        # action_reshaped：上一帧动作的编码 10*7*7        10是动作类别编码向量        7*7是单纯的重复
        #######################################感觉这种特征的堆叠还是缺乏逻辑性##############################
        x = self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = F.relu(self.pointwise(x))
        x = self.dropout(x)
        out = x.view(x.size(0), -1)    #直接将1*122*7*7的张量 变成  1*5978的向量

        return out, image_embedding, action_at, nav_graph, start_coord

    def a3clstm(self, embedding, prev_hidden_h, prev_hidden_c):

        embedding = embedding.reshape([1, 1, self.lstm_input_sz])      #变成1*1*(64*7*7)
        output, (hx, cx) = self.lstm(embedding, (prev_hidden_h, prev_hidden_c))  #lstm编码  64*7*7-512(h和C的维度) 
        #lstm的num_layers=2:表示两个并行的lstm序列并行进行信息传递，最后的输出cat一起/相加
        x = output.reshape([1, self.hidden_state_sz])    #1*512

        actor_out = self.actor_linear(x)   #512 - 6(6种动作)
        critic_out = self.critic_linear_1(x)   #512-64 
        critic_out = self.critic_linear_2(critic_out)   #64-1

        return actor_out, critic_out, (hx, cx)

    def forward(self, model_input, model_options):
        coord = model_input.coord
        start_coord = model_input.start_coord
        nav_graph = model_input.nav_graph                                            
        target_object = model_input.target_object  #物体类别的独热编码=target['indicator']

        state = model_input.state  #resnet18编码后的512维张量 512*7*7
        (hx, cx) = model_input.hidden   #hx是上一时刻输出的状态 cx是上一时刻传过来的隐状态

        target = model_input.target_class_embedding  #包含1.目标类别的独热编码target['indicator']  2.检测到的物体的信息target['info']
        action_probs = model_input.action_probs     #上一帧的动作

        x, image_embedding , action_at,  nav_graph, start_coord= self.embedding(state, target, action_probs, target_object, nav_graph, coord, start_coord)
        actor_out, critic_out, (hx, cx) = self.a3clstm(x, hx, cx)
        actor_out = torch.mul(actor_out, action_at)   #将actor_out（输出的动作概率）与action_at（停止动作的概率）相乘  
        return ModelOutput(
            value=critic_out,            
            logit=actor_out,             #这个epoide做的动作
            hidden=(hx, cx),             #lstm的信息传递
            embedding=image_embedding,   #当前图像的编码
            nav_graph = nav_graph,
            start_coord = start_coord
        )
