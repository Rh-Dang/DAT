from __future__ import print_function, division

import os
import random
import ctypes
import setproctitle
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter   

from utils import command_parser

from utils.class_finder import model_class, agent_class, optimizer_class
from utils.model_util import ScalarMeanTracker
from utils.data_utils import check_data, loading_scene_list
from main_eval import main_eval   #main_eval是训练的时候测试
from full_eval import full_eval   #full_eval是训练后测试

from runners import a3c_train, a3c_val


os.environ["OMP_NUM_THREADS"] = "1"


def main():
    setproctitle.setproctitle("Train/Test Manager")   #将进程的名字命名为Train/Test Manager
    args = command_parser.parse_arguments()    #将python 运行程序的时候后面可以带的参数（以及默认值）导入

    print('Training started from: {}'.format(
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))       #打印出训练开始的时间
    )

    args.learned_loss = False
    args.num_steps = 50                                   
    target = a3c_val if args.eval else a3c_train     #如果参数是args.eval==1,那么target=a3c_val

    scenes = loading_scene_list(args)    #将测试/训练的场景取出来

    create_shared_model = model_class(args.model)    #选择一种模型   create_shared_model= 从这三个中选择一个'BaseModel', 'HOZ', 'MetaMemoryHOZ'
    init_agent = agent_class(args.agent_type)     #选择是NavigationAgent or RandomAgent
    optimizer_type = optimizer_class(args.optimizer)     #选择一个优化器类型
   
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)   #保证每次得到的随机数都是固定的
    random.seed(args.seed)

    if args.eval:              #如果是测试就直接执行main_eval.py
        args.test_or_val = 'test'
        main_eval(args, create_shared_model, init_agent)
        return

    start_time = time.time()
    local_start_time_str = time.strftime(          #开始训练的时间
        '%Y_%m_%d_%H_%M_%S', time.localtime(start_time)
    )

    tb_log_dir = args.log_dir + '/' + args.title + '_' + args.phase + '_' + local_start_time_str    #日志文件存放的位置和文件名
    log_writer = SummaryWriter(log_dir=tb_log_dir)   #用tensorbordX将结果进行可视化

    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method("spawn" , force = True)  #通过spawn方法启动进程

    shared_model = create_shared_model(args)

    train_total_ep = 0
    n_frames = 0

    if args.pretrained_trans is not None:                                          
        saved_state = torch.load(
            args.pretrained_trans, map_location=lambda storage, loc: storage
        )
        model_dict = shared_model.state_dict()
        pretrained_dict = {k: v for k, v in saved_state['model'].items() if             
                           (k in model_dict and v.shape == model_dict[k].shape)}  #将预训练模型中与强化学习重叠的部分取出来
        model_dict.update(pretrained_dict)                                        #将参数替换成预训练的参数
        shared_model.load_state_dict(model_dict)                                  #将部分替换后的参数加载到模型上


    if args.continue_training is not None:     #继续训练之前没训练完的模型
        orgin_state = shared_model.state_dict()
        saved_state = torch.load(
            args.continue_training, map_location=lambda storage, loc: storage
        )
        orgin_state.update(saved_state)
        shared_model.load_state_dict(orgin_state)
        train_total_ep = int(args.continue_training.split('_')[-7])   #说明还是按照原来训练的代数继续往下训练
        n_frames = int(args.continue_training.split('_')[-8])

    if args.fine_tuning is not None:          #对预训练好的模型进行fine_tune
        saved_state = torch.load(
            args.fine_tuning, map_location=lambda storage, loc: storage
        )
        model_dict = shared_model.state_dict()
        pretrained_dict = {k: v for k, v in saved_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(pretrained_dict)
        shared_model.load_state_dict(model_dict)

    if args.update_meta_network:
        for layer, parameters in shared_model.named_parameters():
            if not layer.startswith('meta'):
                parameters.requires_grad = False


    shared_model.share_memory()
    if args.pretrained_trans is not None:
        optimizer = optimizer_type(
            [
                {'params': [v for k, v in shared_model.named_parameters() if
                            v.requires_grad and (k in pretrained_dict)],
                 'lr': args.pretrained_low_lr},                                          #这里已经预训练好的参数需要更小的学习率
                {'params': [v for k, v in shared_model.named_parameters() if
                            v.requires_grad and (k not in pretrained_dict)],
                 'lr': args.lr},
            ]
        )
    else:
        optimizer = optimizer_type(
            [v for k, v in shared_model.named_parameters() if v.requires_grad], lr=args.lr
        )



    optimizer = optimizer_type(
        [v for k, v in shared_model.named_parameters() if v.requires_grad], lr=args.lr
    )
    optimizer.share_memory()
    print(shared_model)

    processes = []

    end_flag = mp.Value(ctypes.c_bool, False)   #对结束标志初始化
    train_res_queue = mp.Queue()           #用于进程之间的通信

    
    for rank in range(0, args.workers):    #多线程进行训练
        p = mp.Process(
            target=target,     #target是定义训练过程的py文件
            args=(
                rank,
                args,
                create_shared_model,
                shared_model,
                init_agent,
                optimizer,
                train_res_queue,
                end_flag,
                scenes,
            ),
        )
        p.start()
        setproctitle.setproctitle('python')
        processes.append(p)
        time.sleep(0.1)
    

    print("Train agents created.")

    train_thin = args.train_thin
    train_scalars = ScalarMeanTracker()

    try:
        while train_total_ep < args.max_ep:   

            train_result = train_res_queue.get()
            train_scalars.add_scalars(train_result)
            train_total_ep += 1
            n_frames += train_result['ep_length']

            if (train_total_ep % train_thin) == 0:   #每1000个step输出一些训练结果
                log_writer.add_scalar('n_frames', n_frames, train_total_ep)
                tracked_means = train_scalars.pop_and_reset()
                for k in tracked_means:
                    log_writer.add_scalar(
                        k + '/train', tracked_means[k], train_total_ep
                    )

            if (train_total_ep % args.ep_save_freq) == 0:      

                print('{}: {}'.format(train_total_ep, n_frames))   #输出训练step+机器人走的总步数
                if not os.path.exists(args.save_model_dir):
                    os.makedirs(args.save_model_dir)
                state_to_save = shared_model.state_dict()
                save_path = os.path.join(
                    args.save_model_dir,
                    '{0}_{1}_{2}_{3}.dat'.format(
                        args.title, n_frames, train_total_ep, local_start_time_str
                    ),
                )
                torch.save(state_to_save, save_path)

    finally:
        log_writer.close()
        end_flag.value = True
        for p in processes:
            time.sleep(0.1)
            p.join()

    if args.test_after_train:
        full_eval()


if __name__ == "__main__":
    main()
