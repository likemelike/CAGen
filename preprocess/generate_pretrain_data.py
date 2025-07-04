import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime
from shapely.geometry import LineString
import numpy as np
import argparse
import os
import Settings
dataset_name = Settings.dataset_name

# 这段代码主要用于轨迹数据预处理，将原始 GPS 轨迹数据编码为深度学习训练数据，方便后续用于轨迹生成模型的训练。它的核心功能包括：
	# 加载轨迹数据（支持北京、波尔图的数据集）
	# 读取路网信息（邻接表 & GPS 坐标）
	# 轨迹编码 & 生成训练数据
	# 写入训练、验证、测试集

def encode_trace(trace, fp, adjacent_list, rid_gps):
    """
    编码轨迹 将轨迹转换为神经网络可用的训练样本
    Args:
        trace: 一条轨迹记录
        fp: 写入编码结果的文件
    """
    trj_list = [int(i) for i in trace['trj_list'].replace("[","").replace("]","").split(',')]
    time_list = [int(i) for i in trace['time_list'].replace("[","").replace("]","").split(',')]
    des = trj_list[-1] # 目标点
    des_gps = rid_gps[str(des)] # 目标点 GPS
    # 遍历轨迹，构造输入-输出样本
    for i in range(1, len(trj_list)):
        cur_loc = trj_list[:i]
        cur_time = time_list[:i]
        cur_rid = cur_loc[-1]
        if str(cur_rid) not in adjacent_list or trj_list[i] not in adjacent_list[str(cur_rid)]:
            # 发生了断路
            return
        candidate_set = adjacent_list[str(cur_rid)]
        if len(candidate_set) > 1:
            # 对于有多个候选点的才有学习的价值
            target = trj_list[i]
            target_index = 0
            # 开始写入编码结果
            cur_loc_str = ",".join([str(i) for i in cur_loc])
            cur_time_str = ",".join([str(i) for i in cur_time])
            candidate_set_str = ",".join([str(i) for i in candidate_set])
            fp.write("\"{}\",\"{}\",{},\"{}\",{}\n".format(cur_loc_str, cur_time_str, des, candidate_set_str, target_index))


if __name__ == '__main__':
    # 1. 读取 顶点编号的轨迹数据集
    train_data = pd.read_csv('/data/like/TrajGen/data/{}/train.csv'.format(dataset_name))
    test_data = pd.read_csv('/data/like/TrajGen/data/{}/test.csv'.format(dataset_name))
    # 2. 读取路网邻接表    字典 source: [negibor1, neigbor2,...]
    with open('/data/like/TrajGen/data/{}/adjacent_list.json'.format(dataset_name), 'r') as f:
        adjacent_list = json.load(f)
    # 3. 读取路网信息表 字典 vertex_id: [lon, lat]
    with open('/data/like/TrajGen/data/{}/rid_gps.json'.format(dataset_name), 'r') as f:
        rid_gps = json.load(f)
    # 4. 设置参数
    train_rate = 0.9
    train_num = int(train_data.shape[0] * train_rate)
    # 5. 写出预训练用数据
    #        trace_loc,       trace_time,   des,     candidate_set,   target
    # "20253,36561,26413", "483,483,484",  30344,   "1408,1409,1410",    1
    train_output = open('/data/like/TrajGen/data/{}/pre_train.csv'.format(dataset_name), 'w')
    eval_output = open('/data/like/TrajGen/data/{}/pre_eval.csv'.format(dataset_name), 'w')
    test_output = open('/data/like/TrajGen/data/{}/pre_test.csv'.format(dataset_name), 'w')
    train_output.write('trace_loc,trace_time,des,candidate_set,target\n')
    eval_output.write('trace_loc,trace_time,des,candidate_set,target\n')
    test_output.write('trace_loc,trace_time,des,candidate_set,target\n')
    for index, row in tqdm(train_data.iterrows(), total=train_data.shape[0], desc='encode train traj'):
        if index <= train_num:
            encode_trace(row, train_output,  adjacent_list, rid_gps)
        else:
            encode_trace(row, eval_output,  adjacent_list, rid_gps)
    for index, row in tqdm(test_data.iterrows(), total=test_data.shape[0], desc='encode test traj'):
        encode_trace(row, test_output,  adjacent_list, rid_gps)
    train_output.close()
    eval_output.close()
    test_output.close()
