from generator.function_g_fc import FunctionGFC
import pandas as pd
from utils.ListDataset import ListDataset
from torch.utils.data import random_split, DataLoader
import torch, os, argparse
import numpy as np
from utils.utils import get_logger
from tqdm import tqdm
import Settings
dataset_name = Settings.dataset_name
road_num = Settings.road_num
time_size = Settings.time_size
loc_pad = Settings.road_num
time_pad = Settings.time_size
data_feature = Settings.data_feature
device = Settings.device
clip = Settings.clip
learning_rate = Settings.learning_rate

def collate_fn(indices):
    batch_trace_loc = []
    batch_trace_time = []
    batch_des = []
    batch_candidate_set = []
    batch_target = []
    trace_loc_len = []
    candidate_set_len = []
    for item in indices:
        # print(item[0])
        trace_loc = [int(i) for i in item[0].split(',')]
        trace_time = [int(i) for i in item[1].split(',')]
        batch_des.append(item[2])
        candidate_set = [int(i) for i in item[3].split(',')]
        batch_trace_loc.append(trace_loc)
        batch_trace_time.append(trace_time)
        batch_candidate_set.append(candidate_set)
        batch_target.append(item[4])
        trace_loc_len.append(len(trace_loc))
        candidate_set_len.append(len(candidate_set))
    # 补齐
    max_trace_len = max(trace_loc_len)
    max_candidate_size = max(candidate_set_len)
    for i in range(len(batch_trace_loc)):
        pad_len = max_trace_len - len(batch_trace_loc[i])
        batch_trace_loc[i] += [loc_pad] * pad_len
        batch_trace_time[i] += [time_pad] * pad_len
        # 对于候选集，选择非下一跳的点进行补齐
        while len(batch_candidate_set[i]) < max_candidate_size:
            # 因为我们已经干掉了 candidate_set len 为 1 的点了
            assert len(batch_candidate_set[i]) != 1, 'candidate set is 1!'
            pad_index = np.random.randint(len(batch_candidate_set[i]))
            if pad_index != batch_target[i]:
                batch_candidate_set[i].append(batch_candidate_set[i][pad_index])
    batch_des = [int(id) for id in batch_des]
    batch_target = [int(id) for id in batch_target]
    return [torch.LongTensor(batch_trace_loc).to(device), torch.LongTensor(batch_trace_time).to(device), torch.LongTensor(batch_des).to(device),torch.LongTensor(batch_candidate_set).to(device), torch.LongTensor(batch_target).to(device)]

# 训练相关参数
max_epoch = 5
batch_size = 2048
weight_decay = 0.00001
lr_patience = 2
lr_decay_ratio = 0.1
save_folder = '/data/like/TrajGen/model/{}/save/'.format(dataset_name)
save_file_name = 'function_g_fc.pt'
temp_folder = '/data/like/TrajGen/model/{}/temp/'.format(dataset_name)
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
early_stop_lr = 1e-6
train = True
# 生成器 config
gen_config = {
    "road_emb_size": 256,  
    "time_emb_size": 50,
    "hidden_size": 256,
    "dropout_p": 0.6,
    "lstm_layer_num": 2,
    "pretrain_road_rep": None,
    "device": device
}

logger = get_logger(name='FunctionGFC')
logger.info('read data')
# 1. 读取训练输入数据
train_data = pd.read_csv('/data/like/TrajGen/data/{}/pre_train.csv'.format(dataset_name), low_memory=False)
eval_data = pd.read_csv('/data/like/TrajGen/data/{}/pre_eval.csv'.format(dataset_name), low_memory=False)
test_data = pd.read_csv('/data/like/TrajGen/data/{}/pre_test.csv'.format(dataset_name), low_memory=False)
train_data = train_data.values.tolist()
eval_data = eval_data.values.tolist()
test_data = test_data.values.tolist()
train_num = len(train_data)
eval_num = len(eval_data)
test_num = len(test_data)
total_data = train_num + eval_num + test_num
logger.info('total input record is {}. train set: {}, val set {}, test set {}'.format(total_data, train_num,
                                                                         eval_num, test_num))
#  封装一个 Python 列表，使其可以被 PyTorch 的数据加载器（DataLoader）
train_dataset = ListDataset(train_data)
eval_dataset = ListDataset(eval_data)
test_dataset = ListDataset(test_data)
# trace_loc, trace_time, des, candidate_set, target
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# 2. 加载模型
gen_model = FunctionGFC(gen_config, data_feature).to(device)
logger.info('init function g fc')
logger.info(gen_model)
optimizer = torch.optim.Adam(gen_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=lr_patience, factor=lr_decay_ratio)
# 3. 开始训练
if train:
    metrics = []
    for epoch in range(max_epoch):
        print("\n")
        logger.info('start train epoch {}'.format(epoch))
        gen_model.train(True)
        train_loss = 0
        for trace_loc, trace_time, des, candidate_set, target in train_loader:
            optimizer.zero_grad()
            trace_mask = ~(trace_loc == loc_pad)
            loss = gen_model.calculate_loss(trace_loc=trace_loc, trace_time=trace_time, candidate_set=candidate_set,
                                            target=target, trace_mask=trace_mask)
            loss.backward()
            train_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(gen_model.parameters(), clip)
            optimizer.step()
        # val
        val_hit = 0
        gen_model.train(False)
        for trace_loc, trace_time, des, candidate_set, target in val_loader:
            trace_mask = ~(trace_loc == loc_pad)
            score = gen_model.predict(trace_loc=trace_loc, trace_time=trace_time,  candidate_set=candidate_set, trace_mask=trace_mask)
            target = target.tolist()
            val, index = torch.topk(score, 1, dim=1)
            for i, p in enumerate(index):
                if target[i] in p:
                    val_hit += 1
        val_ac = val_hit / eval_num
        metrics.append(val_ac)
        lr_scheduler.step(val_ac)
        # store temp model
        torch.save(gen_model.state_dict(), os.path.join(temp_folder, 'function_g_{}.pt'.format(epoch)))
        lr = optimizer.param_groups[0]['lr']
        logger.info('==> Train Epoch {}: Train Loss {:.6f}, val AC {:.6f}, lr {}'.format(epoch, train_loss, val_ac, lr))
        if lr < early_stop_lr:
            logger.info('early stop')
            break
    # load best epoch
    best_epoch = np.argmax(metrics)
    load_temp_file = 'function_g_{}.pt'.format(best_epoch)
    logger.info('load best from {}'.format(best_epoch))
    gen_model.load_state_dict(torch.load(os.path.join(temp_folder, load_temp_file),weights_only=True))
else:
    gen_model.load_state_dict(torch.load(os.path.join(save_folder, save_file_name), map_location=device, weights_only=True))
# 4. 测试集模型评估
test_hit = 0
gen_model.train(False)
for trace_loc, trace_time, des, candidate_set, target in tqdm(test_loader, desc='test model'):
    trace_mask = ~(trace_loc == loc_pad)
    score = gen_model.predict(trace_loc=trace_loc, trace_time=trace_time, candidate_set=candidate_set, trace_mask=trace_mask)
    target = target.tolist()
    val, index = torch.topk(score, 1, dim=1)
    for i, p in enumerate(index):
        if target[i] in p:
            test_hit += 1
test_ac = test_hit / test_num
logger.info('==> Test Result: ac {:.6f}'.format(test_ac))
# 保存模型, 删除 temp 文件
torch.save(gen_model.state_dict(), os.path.join(save_folder, save_file_name))
for rt, dirs, files in os.walk(temp_folder):
    for name in files:
        remove_path = os.path.join(rt, name)
        os.remove(remove_path)
