import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.search import Searcher
import json, os, torch
from generator.generator import Generator
from utils.utils import get_logger
log = get_logger()
from prefixspan import PrefixSpan
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

save_folder = '/data/like/TrajGen/model/{}/save/'.format(dataset_name)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
generator_model_path= os.path.join(save_folder, 'adversarial_generator.pt')


if __name__ == '__main__':
    logger = get_logger("train_gan.py")
    logger.info('loading true trajectory.')
    # 1.1 加载真实轨迹数据
    true_traj = pd.read_csv('/data/like/TrajGen/data/{}/test.csv'.format(dataset_name), nrows=10000)
    total_trace = true_traj.shape[0]
    logger.info(f'loaded true trajectory number: {total_trace}')
    # 1.2 路网邻接表
    with open('/data/like/TrajGen/data/{}/adjacent_list.json'.format(dataset_name), 'r') as f:
        adjacent_list = json.load(f)
    logger.info(f'loaded adjacent_list: {len(adjacent_list)}')
    # 1.3 路网 GPS
    with open('/data/like/TrajGen/data/{}/rid_gps.json'.format(dataset_name), 'r') as f:
        rid_gps = json.load(f)
    logger.info(f'loaded rid_gps: {len(rid_gps)}')
    
    # 2. 加载训练好的生成器
    logger.info('load pretrain generator from ' + generator_model_path)
    generator = Generator(config=gen_config, data_feature=data_feature).to(device)
    generator.load_state_dict(torch.load(generator_model_path, map_location=device, weights_only=True))
    generator.train(False)
    # 3. 生成轨迹
    log.info("Generate a synthetic trajectory for each real trajectory, and store it")
    searcher = Searcher(device, adjacent_list, rid_gps, dataset_name)
    # 存储生成轨迹的经纬度数据
    ## 由编码序列转为经纬度 以便于可视化展示 写出为xyt文件
    paths = ["test_real_lonlat.xyt", "test_gen_lonlat_our.xyt", "test_gen_lonlat_rand.xyt", "test_gen_lonlat_A.xyt"]
    files = {name: open(f"/data/like/TrajGen/data/{dataset_name}/{name}", "w") for name in paths}
    real_f, gen_f_our, gen_f_rand, gen_f_A = files.values()
    # 存储编码轨迹用于下一步 频繁模式挖掘
    all_real_trjs_id, all_gen_trjs_id_our, all_gen_trjs_id_rand, all_gen_trjs_id_A =[], [], [], []
    for index, row in tqdm(true_traj.iterrows(), total=true_traj.shape[0]):
        trj_list = [int(i) for i in row['trj_list'][1:-1].split(',')]
        time_list = [int(i) for i in row['time_list'][1:-1].split(',')]
        real_lonlat_trj = []
        for loc, t in zip(trj_list, time_list):
            lon_lat = [round(i,7) for i in rid_gps[str(loc)]]
            lon_lat.append(t)
            real_lonlat_trj.append(",".join(map(str, lon_lat)))
        lonlat_trj_str = ";".join(real_lonlat_trj)
        real_f.writelines(lonlat_trj_str)
        real_f.writelines("\n")
        with torch.no_grad():
            gen_trace_loc_our, gen_trace_tim_our = searcher.waypoints_search(gen_model=generator, 
                                                                trace_loc=[trj_list[0]],
                                                                trace_tim=[time_list[0]],
                                                                des_id=trj_list[-1])
            gen_trace_loc_rand, gen_trace_tim_rand = searcher.random_search(gen_model=generator, 
                                                                trace_loc=[trj_list[0]],
                                                                trace_tim=[time_list[0]],
                                                                des_id=trj_list[-1])
            gen_trace_loc_A, gen_trace_tim_A = searcher.A_search(gen_model=generator, 
                                                                trace_loc=[trj_list[0]],
                                                                trace_tim=[time_list[0]],
                                                                des_id=trj_list[-1])
            
        gen_lonlat_trj_our = []
        for loc, t in zip(gen_trace_loc_our, gen_trace_tim_our):
            lon_lat = [round(i,7) for i in rid_gps[str(loc)]]
            lon_lat.append(t)
            gen_lonlat_trj_our.append(",".join(map(str, lon_lat)))
        lonlat_trj_str = ";".join(gen_lonlat_trj_our)
        gen_f_our.writelines(lonlat_trj_str)
        gen_f_our.writelines("\n")
            
        gen_lonlat_trj_rand = []
        for loc, t in zip(gen_trace_loc_rand, gen_trace_tim_rand):
            lon_lat = [round(i,7) for i in rid_gps[str(loc)]]
            lon_lat.append(t)
            gen_lonlat_trj_rand.append(",".join(map(str, lon_lat)))
        lonlat_trj_str = ";".join(gen_lonlat_trj_rand)
        gen_f_rand.writelines(lonlat_trj_str)
        gen_f_rand.writelines("\n")
            
        gen_lonlat_trj_A = []
        for loc, t in zip(gen_trace_loc_A, gen_trace_tim_A):
            lon_lat = [round(i,7) for i in rid_gps[str(loc)]]
            lon_lat.append(t)
            gen_lonlat_trj_A.append(",".join(map(str, lon_lat)))
        lonlat_trj_str = ";".join(gen_lonlat_trj_A)
        gen_f_A.writelines(lonlat_trj_str)
        gen_f_A.writelines("\n")
        
        all_real_trjs_id.append(trj_list)
        all_gen_trjs_id_our.append(gen_trace_loc_our)
        all_gen_trjs_id_rand.append(gen_trace_loc_rand)
        all_gen_trjs_id_A.append(gen_trace_loc_A)
    real_f.close()
    gen_f_our.close()
    gen_f_rand.close()
    gen_f_A.close()
    logger.info(f"Save test_real_lonlat_trajectory in {paths[0]}")
    logger.info(f"Save our  test_gen_lonlat_trajectory in {paths[1]}")
    logger.info(f"Save rand test_gen_lonlat_trajectory in {paths[2]}")
    logger.info(f"Save A test_gen_lonlat_trajectory in {paths[3]}")
    
    np.save("/data/like/TrajGen/data/{}/all_real_trjs_id.npy".format(dataset_name), np.array(all_real_trjs_id, dtype=object) )
    np.save("/data/like/TrajGen/data/{}/all_gen_trjs_id_our.npy".format(dataset_name),  np.array(all_gen_trjs_id_our, dtype=object))
    np.save("/data/like/TrajGen/data/{}/all_gen_trjs_id_rand.npy".format(dataset_name),  np.array(all_gen_trjs_id_rand, dtype=object))
    np.save("/data/like/TrajGen/data/{}/all_gen_trjs_id_A.npy".format(dataset_name),  np.array(all_gen_trjs_id_A, dtype=object))
    
    

