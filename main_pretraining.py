import os
import datetime
import random
import ctypes
import setproctitle
import time
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tensorboardX import SummaryWriter

from runners.pretrain_engine import train_one_epoch, evaluate
from utils import command_parser
from utils.misc_util import Logger
from utils.class_finder import model_class
from utils.pretrain_util import PreVisTranfsDataset, init_distributed_mode
import utils.misc as utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7" 
os.environ["OMP_NUM_THREADS"] = "1"

def main():
    setproctitle.setproctitle("Training")
    args = command_parser.parse_arguments()
    #init_distributed_mode(args)                   #在分布式训练中用于分配显卡

    args.data_dir = '/data_sdd/datadrh/HOZ/data/AI2Thor_VisTrans_Pretrain_Data/'  #预训练需要的数据

    print(args)

    # records related  #建立log文件
    start_time_str = time.strftime(
        '%Y-%m-%d_%H-%M-%S', time.localtime(time.time())
    )
    log_dir = os.path.join(args.log_dir, '{}_{}_{}'.format(args.title, args.phase, start_time_str))  
    #args.title: 模型名称  args.phase: train or test  

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

 
    log_file = os.path.join(log_dir, 'pretrain.txt')   #在log文件夹中建立储存日志的文件
    sys.stdout = Logger(log_file, sys.stdout)
    sys.stderr = Logger(log_file, sys.stderr)

    # tb_log_dir = os.path.join(args.work_dir, 'runs', '{}_{}_{}'.format(args.title, args.phase, start_time_str))
    # log_writer = SummaryWriter(log_dir=tb_log_dir)

    if not os.path.exists(args.save_model_dir):    #创建储存模型的文件夹
        os.makedirs(args.save_model_dir)

    # start training preparation steps
    if args.remarks is not None:                #一些较为详细的评论
        print(args.remarks)
    print('Training started from: {}'.format(
        time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())))
    )

    device = torch.device('cuda:'+str(args.gpu_ids[0]))
    print ('using GPU:',args.gpu_ids[0])

    model_creator = model_class(args.model)     #建立模型
    print('chosing model:',args.model)

    model = model_creator(args)
    print('creating model')

    model.to(device)
    print('model on:',args.gpu_ids[0])
    


    criterion = torch.nn.CrossEntropyLoss()     #损失函数
    criterion.cuda()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)   #输出参数量

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.pretrained_lr,
                                  weight_decay=args.weight_decay)  
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop) #学习率下降策略

    dataset_train = PreVisTranfsDataset(args, 'train')  #将输入数据与ground truth数据建立起来
    dataset_val = PreVisTranfsDataset(args, 'val')
    dataset_test = PreVisTranfsDataset(args, 'test')

    sampler_train = torch.utils.data.RandomSampler(dataset_train)   #将数据随机化
    # sampler_train = torch.utils.data.WeightedRandomSampler([1, 1, 1, 1, 1, 1], len(dataset_train))
    sampler_val = torch.utils.data.RandomSampler(dataset_val)
    sampler_test = torch.utils.data.RandomSampler(dataset_test)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)  #组织成batch（128）形式

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, num_workers=args.workers)
    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                 drop_last=False, num_workers=args.workers)

    if args.continue_training is not None:                                              #用之前训练好的模型继续训练
        checkpoint = torch.load(args.continue_training, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:                                                                 #评估模型
        epoch = args.start_epoch
        evaluate(model, criterion, data_loader_test, device, epoch, args.record_act_map)   #record_act_map用来记载动作地图，即每次单独测试都会将动作地图记录下来
        return 0

    print('Start training')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):   #开始循环训练
        if args.distributed:                           #如果是分布式运算，需要set_epoch方法
            sampler_train.set_epoch(epoch)  
        train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm,  #clip_max_norm梯度裁剪：限制梯度爆炸
            print_freq=args.print_freq)
        lr_scheduler.step()

        checkpoint_paths = [os.path.join(args.save_model_dir, 'checkpoint.pth')]
        if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.epoch_save == 0:
            print('Evaluating on Test dataset!')
            evaluate(model, criterion, data_loader_test, device, epoch)
            checkpoint_paths.append(os.path.join(args.save_model_dir, f'checkpoint{epoch:04}.pth'))

        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

        print('Evaluating on Val dataset!')
        evaluate(model, criterion, data_loader_val, device, epoch)   #每一个epoch都需要在validation上测试

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    main()
