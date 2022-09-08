import torch
import numpy as np
import h5py

from utils.model_util import gpuify, toFloatTensor
from models.model_io import ModelInput

from .agent import ThorAgent


class NavigationAgent(ThorAgent):
    """ A navigation agent who learns with pretrained embeddings. """

    def __init__(self, create_model, args, rank, scenes, targets, gpu_id):
        max_episode_length = args.max_episode_length   #最大的导航步长
        hidden_state_sz = args.hidden_state_sz   
        self.action_space = args.action_space   
        from utils.class_finder import episode_class    

        episode_constructor = episode_class(args.episode_type)   #从episodes文件夹中选取一个episode
        episode = episode_constructor(args, gpu_id, args.strict_done)   

        super(NavigationAgent, self).__init__(
            create_model(args), args, rank, scenes, targets, episode, max_episode_length, gpu_id
        )
        self.hidden_state_sz = hidden_state_sz
        self.keep_ori_obs = args.keep_ori_obs

        self.glove = {}
        if 'SP' in self.model_name:
            with h5py.File('/home/dhm/Code/vn/glove_map300d.hdf5', 'r') as rf:
                for i in rf:
                    self.glove[i] = rf[i][:]
        
        self.num_cate = args.num_category
        self.nav_dim = args.nav_dim
        self.coord_memory_dim = args.coord_memory_dim

    def eval_at_state(self, model_options):
        model_input = ModelInput()

        # model inputs
        if self.episode.current_frame is None:
            model_input.state = self.state()
        else:
            model_input.state = self.episode.current_frame

        model_input.hidden = self.hidden

        model_input = self.process_detr_input(model_input)
        # current_detection_feature = self.episode.current_detection_feature()
        # current_detection_feature = current_detection_feature[self.targets_index, :]  #前512维度是视觉特征，后5维度是检测框和label
        # target_embedding_array = np.zeros((len(self.targets), 1))
        # target_embedding_array[self.targets.index(self.episode.target_object)] = 1

        # self.episode.detection_results.append(
        #     list(current_detection_feature[self.targets.index(self.episode.target_object), 512:]))
        # # if self.episode.current_cls_masks() is None:
        # current_cls_masks = np.zeros((22,7,7))
        # # else:
        # #     current_cls_masks = self.episode.current_cls_masks()[()]

        # target_embedding = {'appear': current_detection_feature[:, :512],  
        #                     'info': current_detection_feature[:, 512:],
        #                     'indicator': target_embedding_array,
        #                     'masks':current_cls_masks}
        # target_embedding['appear'] = toFloatTensor(target_embedding['appear'], self.gpu_id)
        # target_embedding['info'] = toFloatTensor(target_embedding['info'], self.gpu_id)
        # target_embedding['indicator'] = toFloatTensor(target_embedding['indicator'], self.gpu_id)
        # target_embedding['masks'] = toFloatTensor(target_embedding['masks'], self.gpu_id)
        # model_input.target_class_embedding = target_embedding

        model_input.start_coord = self.start_coord
        model_input.coord = self.episode.environment.controller.state    #坐标 + 朝向 + 俯仰角
        model_input.nav_graph = self.nav_graph
        model_input.coord_memory = self.coord_memory
        model_input.action_probs = self.last_action_probs
        model_input.actions = self.actions

        model_input.scene = self.episode.environment.scene_name
        model_input.target_object = self.episode.target_object

        if 'Memory' in self.model_name:
            state_length = self.hidden_state_sz

            if len(self.episode.state_reps) == 0:
                model_input.states_rep = torch.zeros(1, state_length)
            else:
                model_input.states_rep = torch.stack(self.episode.state_reps)

            dim_obs = 512
            if len(self.episode.obs_reps) == 0:
                model_input.obs_reps = torch.zeros(1, dim_obs)
            else:
                model_input.obs_reps = torch.stack(self.episode.obs_reps)

            if len(self.episode.state_memory) == 0:
                model_input.states_memory = torch.zeros(1, state_length)
            else:
                model_input.states_memory = torch.stack(self.episode.state_memory)

            if len(self.episode.action_memory) == 0:
                model_input.action_memory = torch.zeros(1, 6)
            else:
                model_input.action_memory = torch.stack(self.episode.action_memory)

            model_input.states_rep = toFloatTensor(model_input.states_rep, self.gpu_id)
            model_input.states_memory = toFloatTensor(model_input.states_memory, self.gpu_id)
            model_input.action_memory = toFloatTensor(model_input.action_memory, self.gpu_id)
            model_input.obs_reps = toFloatTensor(model_input.obs_reps, self.gpu_id)

        return model_input, self.model.forward(model_input, model_options)

    def process_detr_input(self, model_input):
        # process detection features from DETR detector
        current_detection_feature = self.episode.current_detection_feature() #取出数据
        # print('current_detection_feature:')
        # print(current_detection_feature.shape)
        # np.savetxt('visualization_files/current_detection_feature .txt',current_detection_feature )
        # np.savetxt('visualization_files/current_detection_feature_label.txt',current_detection_feature[:, 257])
        zero_detect_feats = np.zeros([22, 264]) #0初始化

        
        for cate_id in range(len(self.targets) + 1):   #同类别检测结果抑制，即只取置信度最高的同类物体
            cate_index = current_detection_feature[:, 257] == cate_id   #cate_index是所有cate_id类别的物体标号
            if cate_index.sum() > 0:                                   #如果有检测到这个物体
                index = current_detection_feature[cate_index, 256].argmax(0)    #取cate_id类中置信度最大的物体标号
                zero_detect_feats[cate_id - 1, :] = current_detection_feature[cate_index, :][index]  
                
        current_detection_feature = zero_detect_feats
        # np.savetxt('visualization_files/zero_detect_feats.txt',current_detection_feature)
        # np.savetxt('visualization_files/zero_detect_feats_label.txt',current_detection_feature[:, 257] )

        # print('current_detection_feature2:')
        # print(current_detection_feature.shape)

        #self.targets就是一个包含所有物体名称的数组
        detection_inputs = {
            'features': current_detection_feature[:, :256],
            'scores': current_detection_feature[:, 256],
            'labels': current_detection_feature[:, 257], 
            'bboxes': current_detection_feature[:, 260:],
            'target': self.targets.index(self.episode.target_object),
        }

        # generate target indicator array based on detection results labels
        target_embedding_array = np.zeros((detection_inputs['features'].shape[0], 1))
        target_embedding_array[
            detection_inputs['labels'][:] == (self.targets.index(self.episode.target_object) + 1)] = 1
        detection_inputs['indicator'] = target_embedding_array #如果检测到的是目标物体，就在indicator位置1

        detection_inputs = self.dict_toFloatTensor(detection_inputs)

        model_input.target_class_embedding = detection_inputs 

        return model_input

    def preprocess_frame(self, frame):
        """ Preprocess the current frame for input into the model. """
        state = torch.Tensor(frame)
        return gpuify(state, self.gpu_id)

    def reset_hidden(self):
        with torch.cuda.device(self.gpu_id):
            self.hidden = (
                torch.zeros(2, 1, self.hidden_state_sz).cuda(),
                torch.zeros(2, 1, self.hidden_state_sz).cuda(),
            )

        self.last_action_probs = gpuify(
            torch.zeros((1, self.action_space)), self.gpu_id
        )

        self.nav_graph = gpuify(
            torch.zeros((1, self.nav_dim)), self.gpu_id
        )
        self.coord_memory = gpuify(
            torch.zeros((1, self.coord_memory_dim)), self.gpu_id
        )
        self.start_coord = {'x': -10, 'y': -10, 'rotation': -10, 'horizon': -10}

        #self.model.reset()

    def repackage_hidden(self):
        if self.hidden != None:                             #如果不用LSTM就用不上这个了
            self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        self.last_action_probs = self.last_action_probs.detach()

    def state(self):
        return self.preprocess_frame(self.episode.state_for_agent())

    def exit(self):
        pass

    def dict_toFloatTensor(self, dict_input):

        for key in dict_input:
            dict_input[key] = toFloatTensor(dict_input[key], self.gpu_id)

        return dict_input
