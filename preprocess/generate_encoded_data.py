# -*- coding: utf-8 -*-
import os,  argparse, json, csv, time
from tqdm import tqdm
from scipy.spatial import KDTree
from prefixspan import PrefixSpan
import numpy as np
import Settings
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.utils import get_logger
dataset_name = Settings.dataset_name

# The minimal length of a trajectory
min_len = 10 
# The maximum moving distance between two adjacent location (km)
max_distance = 3
# The ratio of train dataset
train_ratio = 0.8
logger = get_logger("data loader.py")
# one meter == 0.00001
scale = 0.0005
time_size = 60*24
hot_freq = 10 if dataset_name == "Geolife" else 50
max_trjs_num = 1000000000
# The min_support and min_length of a co-movement pattern
# min_support=30, min_length=3 by default

# Haversine calculates the distance (km) between two lon/lat locations
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    # Haversine 
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c  

def find_frequent_subsequences(file_path, output_path, min_support=50, min_length=3):
    df = pd.read_csv(file_path)
    sequences = df['trj_list'].dropna()
    sequences = sequences.apply(lambda x: list(map(int, str(x).strip('[]').split(',')))).tolist()
    logger.info(f"Loading trajectory: {len(sequences)}")
    # 创建 PrefixSpan 对象, 挖掘频繁子序列
    ps = PrefixSpan(sequences)
    frequent_patterns = ps.frequent(min_support)
    filtered_patterns = [(pattern, support) for support, pattern in frequent_patterns if len(pattern) >= min_length]
    logger.info(f"Found {len(filtered_patterns)} co-movement patterns")
    sequence_dict = {}
    for seq, weight in filtered_patterns:
        key = (seq[0], seq[-1])  # 以 (起点, 终点) 作为 key
        if key not in sequence_dict:
            sequence_dict[key] = []  # 初始化为空列表
        sequence_dict[key].append([seq, weight])  # 追加 (序列, 权重)
    np.save(output_file, sequence_dict, allow_pickle=True)
    logger.info(f"Write {len(sequence_dict)} frequent seqeunces dict to path: {output_path}")

def split_dataset(input_csv, train_ratio, output_train, output_test):
    # 读取数据集
    df = pd.read_csv(input_csv)
    # 按比例划分数据集
    train_df, val_df = train_test_split(df, train_size=train_ratio, random_state=42)
    # 保存训练集和验证集
    train_df.to_csv(output_train, index=False)
    val_df.to_csv(output_test, index=False)
    logger.info(f"Split Data Finished, Train: {len(train_df)} Test: {len(val_df)}")
    logger.info(f"Train is store in {output_train}, Test is stored in {output_test}")

def setRegionArgs(dataset_name, scale, time_size):
    """
    parameter settings of space partition and time partition
    """
    parser = argparse.ArgumentParser(description="Region.py")
    if dataset_name == "Geolife":
        lons_range, lats_range = [116.25, 116.55], [39.83, 40.03]
    elif dataset_name == 'Porto':
        lons_range, lats_range = [-8.735, -8.156], [40.953, 41.307]
    else:
        logger.info("No specific dataset found!!")
        exit(0)
    # space partition under specific scale
    maxx, maxy = (lons_range[1]-lons_range[0])//scale, (lats_range[1]-lats_range[0])//scale
    parser.add_argument("-lons", default= lons_range, help="range of longitude")
    parser.add_argument("-lats", default= lats_range, help="range of latitude")
    parser.add_argument("-numx", type=int, default = maxx)
    parser.add_argument("-numy", type=int, default = maxy)
    parser.add_argument("-space_cell_size", type=int, default=maxx*maxy, help="cell numbers in 2D coordinates")
    # time partition under specific number of time slices
    parser.add_argument("-time_span", type = int, default= 86400 // time_size)
    args = parser.parse_args()
    return args

class Region:
    """
    划分空间格子, 将轨迹映射到热度词编码
    对于每一个原始的（x,y,t)轨迹点，将其转化为对应的时空格子编码
    """
    def __init__(self, dataset_name, scale, time_size, max_trjs_num=10000):
        self.read_path = os.path.join("/data/like/clean/", dataset_name + ".xyt")
        self.max_trjs_num = max_trjs_num
        self.args = setRegionArgs(dataset_name, scale, time_size)
        self.word2hotcell = {}
        self.hotcell2word = {}
        print("Parameters：", self.args)

    def create_vocal(self):
        id_counter = {}
        read_lines = 0
        # 1. store all trjs in the form of spaceID sequences
        trjs = []
        with open(self.read_path, 'r') as file:
            for line in file:
                line = line.strip()
                read_lines += 1
                if line:
                    trj = []
                    points = line.split(';')
                    for i, point in enumerate(points):
                        try:
                            lon, lat, t = map(float, point.split(','))
                            space_id = self.lonlat2spaceId(lon, lat)
                            assert t<=86400
                            t = int(t) // self.args.time_span
                            trj.append([space_id, t])
                        except ValueError:
                            logger.info(f"Omit: {point}")
                if read_lines % 200000 == 0:
                    logger.info(read_lines)
                if read_lines > self.max_trjs_num:
                    break
                # 对每条轨迹，删除重复的space_id点
                seen = {}  # 用字典存储唯一的 space_id
                result = []
                for space_id, t in trj:
                    if space_id not in seen:  # 避免重复
                        seen[space_id] = t
                        result.append([space_id, t])
                if len(result)>=min_len:
                    trjs.append(result)
        # 2. count the number of each space cell to be hit
        for trj in tqdm(trjs, "hot word count"):
            for space_id, t in trj:
                id_counter[space_id] = id_counter.get(space_id, 0) + 1
        # 3. record the mapping from hot cell id to the word id
        sort_cnt = sorted(id_counter.items(), key=lambda count: count[1], reverse=True)
        hot_ids = [int(sort_cnt[i][0]) for i in range(len(sort_cnt)) if sort_cnt[i][1] >= hot_freq]
        self.hotcell2word = {hot_ids[ii]: ii for ii in range(len(hot_ids))}
        self.word2hotcell = {ii: hot_ids[ii] for ii in range(len(hot_ids))}
        # 4. create a KDTree for fast NN computation
        hot_cell_lonlats = []
        for hot_id in hot_ids:
            hot_cell_lonlats.append(self.spaceId2lonlat(hot_id))
        self.tree = KDTree(hot_cell_lonlats)
        return trjs, self.hotcell2word, self.word2hotcell
    '''
    ****************************************************************************************************************************************************
    一系列的转化函数
    ****************************************************************************************************************************************************
    '''
    def lonlat2xyoffset(self, lon, lat):
        '''经纬度转换为米为单位, 映射到平面图上 (116.3, 40.0)->(4,8)'''
        xoffset = round((lon - self.args.lons[0]) / scale)
        yoffset = round((lat - self.args.lats[0]) / scale)
        return int(xoffset), int(yoffset)

    def xyoffset2lonlat(self, xoffset, yoffset):
        ''' 米单位转换为经纬度  (4,8)-> (116.3, 40.0)'''
        lon = self.args.lons[0]+xoffset*scale
        lat = self.args.lats[0]+yoffset*scale
        return lon,lat
    
    def offset2spaceId(self, xoffset, yoffset):
        ''' (xoffset,yoffset) -> space_id  (4,8)->116'''
        return int(yoffset * self.args.numx + xoffset)

    def lonlat2spaceId(self, lon, lat):
        ''' lonlat--> space_id  116.3,40->116'''
        xoffset, yoffset = self.lonlat2xyoffset(lon, lat)
        space_id = self.offset2spaceId(xoffset, yoffset)
        return int(space_id)

    def spaceId2lonlat(self, space_id):
        '''space_id -->lonlat 116->116.3,40'''
        yoffset = int(space_id // self.args.numx)
        xoffset = int(space_id % self.args.numx)
        lon,lat = self.xyoffset2lonlat(xoffset,yoffset)
        return lon,lat

    def trj2words(self,trj):
        words = []
        ts = []
        for space_id, t in trj:
            map_id = self.hotcell2word.get(space_id, -1)
            # use the nearest space cell id to approximate
            if  map_id == -1:
                lon, lat = self.spaceId2lonlat(space_id)
                # use a NN to replace a non-hot location
                dist, map_id = self.tree.query([lon, lat])
            map_id = int(map_id)
            if len(words)>0:
                cur_lon, cur_lat = self.spaceId2lonlat(r.word2hotcell.get(map_id))
                last_lon, last_lat = r.spaceId2lonlat(r.word2hotcell.get(words[-1]))
                if map_id in words or haversine(last_lon, last_lat,cur_lon, cur_lat) > max_distance:
                    continue
            words.append(map_id)
            ts.append(t)
        return words, ts



if __name__ == "__main__":
    
    r = Region(dataset_name, scale, time_size, max_trjs_num)
    logger.info('0-Getting hot word dict...')
    trjs, hotcell2word, word2hotcell = r.create_vocal()
    logger.info('The number of hot words：{}'.format(len(hotcell2word)))
    
    logger.info("1-Getting adjacent table and storing encoded trajectories..")
    trj_data = []
    time_data = []
    adjacent_list = {}  
    for trj in tqdm(trjs,desc="Encoding trajectory"):
        trj_words, time_list = r.trj2words(trj)  
        if len(trj_words)<min_len:
            continue
        trj_data.append(trj_words)
        time_data.append(time_list)
        # record the adjacent list
        for j in range(len(trj_words) - 1):
            a, b = trj_words[j], trj_words[j + 1]
            adjacent_list.setdefault(a, set()).add(b)
            adjacent_list.setdefault(b, set()).add(a)
    csv_path = f"/data/like/TrajGen/data/{dataset_name}/all_trajectory.csv"
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["trj_list", "time_list"])  
        for trj_words, time_list in zip(trj_data, time_data):
            writer.writerow([json.dumps(trj_words), json.dumps(time_list)])  
    logger.info(f"Trajectories {len(trj_data)} is stored in {csv_path}")
    
    # 2. 生成热度词-经纬度信息表
    logger.info("Getting coordinates of hot words..")
    cell_info = {}
    for word, space_id in word2hotcell.items():
        cell_info[word] = r.spaceId2lonlat(space_id)
    json_path = f"/data/like/TrajGen/data/{dataset_name}/rid_gps.json"
    with open(json_path, 'w') as f:
        json.dump(cell_info, f)
    logger.info(f"Hot word coordinates:{len(cell_info)} is stored in {json_path}")
    
    # 3. 存储邻接表
    logger.info("Storing adjacent table..")
    adjacent_list = {k: list(v) for k, v in adjacent_list.items()}
    adjacent_path = f"/data/like/TrajGen/data/{dataset_name}/adjacent_list.json"
    with open(adjacent_path, 'w') as f:
        json.dump(adjacent_list, f)
    logger.info(f"Adjacent Table {len(adjacent_list)} is stored in {adjacent_path}")
    
    ## 4. 提前计算距离信息
    logger.info("Calculating distance matrix..")
    with open('/data/like/TrajGen/data/{}/rid_gps.json'.format(dataset_name), 'r') as f:
        rid_gps = json.load(f)
    coordinates = np.array([loc for loc in rid_gps.values()])  # (lon, lat)
    lon = coordinates[:, 0]
    lat = coordinates[:, 1]
    distance_matrix = np.zeros((len(rid_gps), len(rid_gps)))
    for i in tqdm(range(len(rid_gps)), desc='calc distance'):
        dist = haversine(lon[i], lat[i], lon, lat)
        distance_matrix[i, :] = dist 
    np.save(f"/data/like/TrajGen/data/{dataset_name}/distance.npy", distance_matrix)
    logger.info(f"Distance matrix {distance_matrix.shape} is strored in /data/like/TrajGen/data/{dataset_name}/distance.npy")
    
    ## 5. split train and test
    logger.info("Splitting train and test..")
    input_csv = f"/data/like/TrajGen/data/{dataset_name}/all_trajectory.csv"
    output_train = f"/data/like/TrajGen/data/{dataset_name}/train.csv"
    output_test = f"/data/like/TrajGen/data/{dataset_name}/test.csv"
    split_dataset(input_csv, train_ratio, output_train, output_test)
    
    ## 6. generate frequent subsequences
    logger.info("Generate co-movement patterns..")
    input_file = f"/data/like/TrajGen/data/{dataset_name}/train.csv"
    output_file = f"/data/like/TrajGen/data/{dataset_name}/patterns.npy"
    # generate
    find_frequent_subsequences(input_file, output_file)
    # test
    loaded_patterns = np.load(output_file, allow_pickle=True).item()
    
    

    

        
    