import pandas as pd
import numpy as np
import json
import Settings
dataset_name = Settings.dataset_name


def testAdjacentInfo():
    with open('/data/like/TrajGen/data/{}/adjacent_list.json'.format(dataset_name), 'r') as f:
        adjacent_list = json.load(f)
    # 测试距离矩阵最大最小值
    save_path = f"/data/like/TrajGen/data/{dataset_name}/distance.npy"
    distance_matrix = np.load(save_path)
    print(f"Distance Matrix: {distance_matrix.shape}")
    adjacent_dists = []
    for key in adjacent_list:
        for neighbor in adjacent_list[key]:
            adjacent_dists.append(distance_matrix[int(key),neighbor]) 
    return np.min(adjacent_dists), np.max(adjacent_dists), np.mean(adjacent_dists)

def testEncodedTrajectory():
    true_traj = pd.read_csv('/data/like/TrajGen/data/{}/train.csv'.format(dataset_name))
    max_id_value = 0
    max_time_value = 0
    for index, trace in true_traj.iterrows():
        trj_list = [int(i) for i in trace['trj_list'][1:-1].split(',')]
        max_id_value = max(max_id_value, np.max(trj_list))
        time_list = [int(i) for i in trace['time_list'][1:-1].split(',')]
        max_time_value = max(max_time_value, np.max(time_list))
    return max_id_value, max_time_value


min_dist, max_dist, mean_dist = testAdjacentInfo()
print(f"min_dist:{min_dist}  max_dist:{max_dist}  mean_dist:{mean_dist}")
max_id_value, max_time_value = testEncodedTrajectory()
print(f"max_id_value: {max_id_value} max_time_value: {max_time_value}")







