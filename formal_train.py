import json, os, torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import random_split
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from utils.ListDataset import ListDataset
from torch.utils.data import DataLoader
from generator.generator import Generator
from discriminator.discriminator import Discriminator
from utils.utils import get_logger
from utils.evaluate_funcs import edit_distance, hausdorff_metric, dtw_metric
from utils.search import Searcher
import torch.nn.functional as F
import Settings
dataset_name = Settings.dataset_name
road_num = Settings.road_num
time_size = Settings.time_size
loc_pad = Settings.road_num
time_pad = Settings.time_size
data_feature = Settings.data_feature
gen_config = Settings.gen_config
dis_config = Settings.dis_config
device = Settings.device
clip = Settings.clip
learning_rate = Settings.learning_rate

# 模型存储位置
save_folder = '/data/like/TrajGen/model/{}/save/'.format(dataset_name)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
generator_model_path= os.path.join(save_folder, 'adversarial_generator.pt')
discriminator_model_path = os.path.join(save_folder, 'adversarial_discriminator.pt')
# 其他参数
weight_decay = 0.0001
dis_train_rate = 0.8
total_epoch = 5
dis_sample_num = 1024
gen_sample_num = 1024
batch_size = 256

def collate_fn(indices):
    """
    自定义 DataLoader 收集函数
    Args:
        indices: 一个 batch 的数据

    Returns:

    """
    trace_loc = []
    trace_tim = []
    label = []
    for i in indices:
        trace_loc.append(torch.tensor(i[0]))
        trace_tim.append(torch.tensor(i[1]))
        label.append(i[2])
    trace_loc = pad_sequence(trace_loc, batch_first=True, padding_value=loc_pad)
    trace_tim = pad_sequence(trace_tim, batch_first=True, padding_value=time_pad)
    label = torch.tensor(label)
    trace_mask = ~(trace_loc == loc_pad)
    return [trace_loc.to(device), trace_tim.to(device), label.to(device), trace_mask.to(device)]

# 返回判别器的训练数据
def generate_discriminator_data(pos, searcher, gen_model):
    """
    Args:
        pos (pandas.Dataframe): 正样本轨迹的 df 形式数据，包含三列 trace_loc trace_tim, trace_label
        gen_model (Generator): 生成器，用于生成样本

    Returns:
        train_dataloader (DataLoader): 返回组织好的训练数据
        eval_dataloader (DataLoader)
    """
    data = []
    for index, row in tqdm(pos.iterrows(), total=pos.shape[0], desc='generate discriminator data'):
        trace_loc = list(map(int, row['trj_list'].replace("[","").replace("]","").split(',')))
        trace_tim = [int(i) for i in row['time_list'].replace("[","").replace("]","").split(',')]
        data.append([trace_loc, trace_tim, 1])
        neg_trace_loc, neg_trace_tim = searcher.random_search(gen_model=generator, trace_loc=[trace_loc[0]],trace_tim=[trace_tim[0]], des_id=trace_loc[-1])
        data.append([neg_trace_loc, neg_trace_tim, 0])
    dataset = ListDataset(data)
    # split train and eval dataloaders
    train_num = int(len(dataset) * dis_train_rate)
    eval_num = len(dataset) - train_num
    train_dataset, eval_dataset = random_split(dataset, [train_num, eval_num])
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn), \
        DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        
def train_discriminator(trace, generator, discriminator, seacher, max_epoch, total_trace):
    """
    根据抽样的真实轨迹与生成的轨迹预训练判别器
    """
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator.train(False)
    for epoch in range(max_epoch):
        # 每一轮随机抽取 dis_sample_num 条真实轨迹作为正样本，对应生成的 dis_sample_num 条轨迹作为负样本
        pos_sample_index = np.random.randint(0, total_trace, size=dis_sample_num)
        pos_sample = trace.iloc[pos_sample_index]
        train_loader, eval_loader = generate_discriminator_data(pos_sample, searcher, generator)
        train_total_loss = 0
        
        discriminator.train(True)
        for batch in tqdm(train_loader, desc='train discriminator'):
            # batch[0]   batch[1]   batch[2]  batch[0]
            # trace_loc trace_tim   label    trace_mask
            dis_optimizer.zero_grad()
            # score (tensor): 轨迹是否为真的分数. (batch_size, 2) 0 维度表示为假的概率，1 表示为真的概率。
            score = discriminator.forward(trace_loc=batch[0], trace_time=batch[1],trace_mask=batch[3])
            loss = discriminator.loss_func(score, batch[2])
            params_before = torch.cat([p.flatten() for p in discriminator.parameters()])
            loss.backward()
            train_total_loss += loss.item()
            dis_optimizer.step()
            params_after = torch.cat([p.flatten() for p in discriminator.parameters()])
            delta = (params_after - params_before).abs().mean()    
            # print(f"\nParameter change mean: {delta.item()}")
        # 验证
        discriminator.train(False)
        eval_hit = 0
        eval_total_cnt = len(eval_loader.dataset)
        for batch in tqdm(eval_loader, desc='eval discriminator'):
            # score = tensor([[ 0.0607,  0.0183],[ 0.0675,  0.0392]], device='cuda:0', grad_fn=<AddmmBackward0>)
            score = discriminator.forward(trace_loc=batch[0], trace_time=batch[1],trace_mask=batch[3])
            truth = batch[2]
            # val=tensor([[0.0607],0.0675]]), device='cuda:0', grad_fn=<TopkBackward0>)
            # index = tensor([[0],[0]]), device='cuda:0')
            val, index = torch.topk(score, 1, dim=1)
            for i, p in enumerate(index):
                if truth[i] in p:
                    eval_hit += 1
        avg_ac = eval_hit / eval_total_cnt
        logger.info('train discriminator epoch {}: loss {:.6f}, top1 ac {:.6f}'.format(epoch, train_total_loss, avg_ac))
        discriminator.train(True)

def train_generator(trace, generator, discriminator, searcher,  total_trace):
    # 1.1 init optimizer
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=weight_decay)
    gen_optimizer.zero_grad()
    generator.train(True)
    # 1.2 init evaluation distance
    total_edit_distance = 0
    total_hausdorff = 0
    total_dtw = 0
    total_cnt = 0
    total_loss = 0
    # 1.3 init training samples: each round selct true samples and generated samples with size gen_sample_num
    pos_sample = trace.iloc[np.random.randint(0, total_trace, size=gen_sample_num)]
    for index, row in tqdm(pos_sample.iterrows(), total=pos_sample.shape[0], desc='train generator'):
        trace_loc = list(map(int, row['trj_list'].replace("[","").replace("]","").split(',')))
        trace_tim = [int(i) for i in row['time_list'].replace("[","").replace("]","").split(',')]
        # 根据这个轨迹的 OD 生成轨迹
        neg_trace_loc, neg_trace_tim = searcher.random_search(gen_model=generator, trace_loc=[trace_loc[0]],trace_tim=[trace_tim[0]], des_id=trace_loc[-1])
        # print(trace_loc, len(trace_tim))
        # print(neg_trace_loc, len(neg_trace_tim))
        # print("\n")
        data = []
        data.append([neg_trace_loc, neg_trace_tim, 0])
        batch = collate_fn(data)
        # 计算模型对每一步的候选集概率预测值与所选择下一跳在候选集中的下标
        seq_len = len(neg_trace_loc)
        if seq_len<=1:
            continue
        des_center_gps = rid_gps[str(trace_loc[-1])]
        # candidate_prob_list存储所有步的candidate 概率
        candidate_prob_list = []
        # gen_candidate 存储所有生成的下一个点 在领结表中的 索引
        gen_candidate = []
        for i in range(1, seq_len):
            input_trace_loc = neg_trace_loc[:i]
            input_trace_tim = neg_trace_tim[:i]
            now_rid = input_trace_loc[-1]
            candidate_set = adjacent_list[str(now_rid)]
            # 构建模型输入 trace_loc_tensor.shape=[1, seq_len]
            trace_loc_tensor = torch.LongTensor(input_trace_loc).to(device).unsqueeze(0)
            trace_tim_tensor = torch.LongTensor(input_trace_tim).to(device).unsqueeze(0)
            candidate_set_tensor = torch.LongTensor(candidate_set).to(device).unsqueeze(0)
            candidate_prob = generator.predict(trace_loc=trace_loc_tensor, trace_time=trace_tim_tensor, candidate_set=candidate_set_tensor)
            candidate_prob_list.append(candidate_prob.squeeze(0))
            # 获取所选择的下一跳在候选集中 的下标
            gen_candidate.append(candidate_set.index(neg_trace_loc[i]))
        # 计算 loss
        gen_candidate = torch.tensor(gen_candidate).to(device) # (seq_len)
        candidate_prob_list = pad_sequence(candidate_prob_list, batch_first=True)  # (seq_len, candidate_size)
        # 选出 gen_candidate 对应的 prob tensor([0.5,0.4,0.3,1.0])
        select_prob = torch.gather(candidate_prob_list, dim=1, index=gen_candidate.unsqueeze(1)).squeeze(1)  # (seq_len)
        # print("select_prob", select_prob)
        # 计算reward
        # score (tensor): 轨迹是否为真的分数. (batch_size, 2) 0 维度表示为假的概率，1 表示为真的概率。
        score = discriminator(trace_loc=batch[0], trace_time=batch[1],trace_mask=batch[3])
        # 计算 softmax 并保留梯度 (batch_size, 2) 概率归一化
        score_softmax = F.softmax(score, dim=1)
        # 第二列除以第一列
        reward = score_softmax[:, 1] / (score_softmax[:, 0] + 1e-9)  # (batch_size,)
        # print(score, score_softmax, score_softmax[:, 1], score_softmax[:, 0],reward)
        loss = -torch.log(select_prob*reward + 1e-9).mean()
        # 计算梯度
        torch.nn.utils.clip_grad_norm_(generator.parameters(), clip)
        # params_before = torch.cat([p.flatten() for p in generator.parameters()])
        total_loss += loss
        loss.backward()
        gen_optimizer.step()
        # params_after = torch.cat([p.flatten() for p in generator.parameters()])
        # delta = (params_after - params_before).abs().mean()    
        # print(f"\nParameter change mean: {delta.item()}") 
        # 计算评估指标
        total_edit_distance += edit_distance(neg_trace_loc, trace_loc)
        generate_gps_list = []
        for road_id in neg_trace_loc:
            now_gps = rid_gps[str(road_id)]
            generate_gps_list.append([now_gps[1], now_gps[0]])
        true_gps_list = []
        for road_id in trace_loc:
            now_gps = rid_gps[str(road_id)]
            true_gps_list.append([now_gps[1], now_gps[0]])
        true_gps_list = np.array(true_gps_list)
        generate_gps_list = np.array(generate_gps_list)
        total_hausdorff += hausdorff_metric(true_gps_list, generate_gps_list)
        total_dtw += dtw_metric(true_gps_list, generate_gps_list)
        total_cnt += 1
    logger.info('evaluate generator:')
    logger.info('avg EDT {}, hausdorff {}, dtw {}, loss {:.6f}'.format(total_edit_distance / total_cnt,
                                                                  total_hausdorff / total_cnt,
                                                                  total_dtw / total_cnt, total_loss/total_cnt))

if __name__ == '__main__':
    logger = get_logger("train_gan.py")
    logger.info('loading true trajectory.')
    # 1. 加载数据
    # 1.1 真实轨迹数据
    trace = pd.read_csv('/data/like/TrajGen/data/{}/train.csv'.format(dataset_name), nrows=10000000)
    logger.info(f'loaded true trajectory number: {trace.shape[0]}')
    # 1.2 路网邻接表
    with open('/data/like/TrajGen/data/{}/adjacent_list.json'.format(dataset_name), 'r') as f:
        adjacent_list = json.load(f)
    logger.info(f'loaded adjacent_list: {len(adjacent_list)}')
    # 1.3 路网 GPS
    with open('/data/like/TrajGen/data/{}/rid_gps.json'.format(dataset_name), 'r') as f:
        rid_gps = json.load(f)
    logger.info(f'loaded rid_gps: {len(rid_gps)}')
    
    # 2. 初始化模型
    # 2.1 加载预训练生成器
    pretrain_gan_file= "/data/like/TrajGen/model/{}/save/function_g_fc.pt".format(dataset_name)
    generator = Generator(config=gen_config, data_feature=data_feature).to(device)
    logger.info('load pretrain generator from ' + pretrain_gan_file)
    generatorv1_state = torch.load(pretrain_gan_file, map_location=device, weights_only=True)
    generator.function_g.load_state_dict(generatorv1_state)
    # 2.2 预先训练判别器
    logger.info('pretrain discriminator ...')
    discriminator = Discriminator(config=dis_config, data_feature=data_feature).to(device)
    # 2.3 加载搜索器
    searcher = Searcher(device, adjacent_list, rid_gps, dataset_name)
    train_discriminator(trace, generator, discriminator, searcher, 3, trace.shape[0])
    # 3. 开始对抗训练
    logger.info(f'constrtive training, total epoch: {total_epoch}')
    for epoch in range(total_epoch):
        print("\n")
        logger.info('start train generator at epoch {}'.format(epoch))
        train_generator(trace, generator, discriminator, searcher, trace.shape[0])
        if epoch%5 == 0:
            logger.info('start train discriminator at epoch {}'.format(epoch))
            train_discriminator(trace, generator, discriminator, searcher, 1, trace.shape[0])
            # 保存本次训练的生成器与判别器
            torch.save(generator.state_dict(), generator_model_path)
            torch.save(discriminator.state_dict(), discriminator_model_path)
