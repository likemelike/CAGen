import folium, json
from folium.plugins import HeatMap
import numpy as np
import random
from prefixspan import PrefixSpan
import Settings
from evaluate_funcs import edit_distance, hausdorff_metric, dtw_metric, s_edr
from tqdm import tqdm

dataset_name = Settings.dataset_name


def calc_destination_hit_ratio(real_trjs_id, gen_trjs_id):
    hit = 0
    assert len(real_trjs_id) == len(gen_trjs_id)
    for ii in range(len(real_trjs_id)):
        if real_trjs_id[ii][-1] == gen_trjs_id[ii][-1]:
            hit += 1
    print(f"Destination hit: {hit}  total trajectories: {len(real_trjs_id)}")
    return hit/len(real_trjs_id)

def calc_pattern_hit_ratio(real_patterns, gen_patterns):
    hit = 0
    for key in real_patterns:
        if key in gen_patterns:
            for value in gen_patterns[key]:
                hit += value[1]
    print(f"Pattern hit: {hit}  number of real patterns:{len(real_patterns)}")
    return hit

def find_frequent_subsequences(sequences, min_support=10, min_length=3):
    ps = PrefixSpan(sequences)
    frequent_patterns = ps.frequent(min_support)
    filtered_patterns = [(pattern, support) for support, pattern in frequent_patterns if len(pattern) >= min_length]
    sequence_dict = {}
    for seq, weight in filtered_patterns:
        key = (seq[0], seq[-1])  
        if key not in sequence_dict:
            sequence_dict[key] = []  
        sequence_dict[key].append([seq, weight])  
    print(f"find {len(sequence_dict)} co-movement patterns")
    return sequence_dict


def calc_metrics(real_patterns, gen_patterns):
    total_dtw = 0
    total_hausdorff = 0
    total_edit_distance = 0
    total_edr_distance = 0
    with open('/data/like/TrajGen/data/{}/rid_gps.json'.format(dataset_name), 'r') as f:
        rid_gps = json.load(f)
    total_cnt = len(real_patterns)
    for ii in tqdm(range(total_cnt), desc="calculating metrics"):
        neg_trace_loc = gen_patterns[ii]
        trace_loc = real_patterns[ii]
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
        total_edr_distance += s_edr(true_gps_list, generate_gps_list, 10)
    print('avg EDT {}, hausdorff {}, dtw {}, edr {}'.format(total_edit_distance / total_cnt,
                                                                  total_hausdorff / total_cnt,
                                                                  total_dtw / total_cnt, total_edr_distance/ total_cnt))

if __name__ == "__main__":
    # 1. parameter settings
    min_support = 10 if dataset_name == "Geolife" else 8
    min_length = 3
    # 2. compute and store co-movement patterns
    all_real_trjs_id = np.load("/data/like/TrajGen/data/{}/all_real_trjs_id.npy".format(dataset_name), allow_pickle=True)
    all_gen_trjs_id_our = np.load("/data/like/TrajGen/data/{}/all_gen_trjs_id_our.npy".format(dataset_name), allow_pickle=True)
    all_gen_trjs_id_rand = np.load("/data/like/TrajGen/data/{}/all_gen_trjs_id_rand.npy".format(dataset_name), allow_pickle=True)
    all_gen_trjs_id_A = np.load("/data/like/TrajGen/data/{}/all_gen_trjs_id_A.npy".format(dataset_name), allow_pickle=True)
    assert len(all_real_trjs_id) == len(all_gen_trjs_id_our) == len(all_gen_trjs_id_rand) == len(all_gen_trjs_id_A)
    print(len(all_real_trjs_id))
    
    # averaged length 0f trajectories
    print(sum(len(sublist) for sublist in all_real_trjs_id) / len(all_real_trjs_id))
    print(sum(len(sublist) for sublist in all_gen_trjs_id_our) / len(all_gen_trjs_id_our))
    print(sum(len(sublist) for sublist in all_gen_trjs_id_rand) / len(all_real_trjs_id))
    print(sum(len(sublist) for sublist in all_gen_trjs_id_A) / len(all_real_trjs_id))
    
    calc_metrics(all_real_trjs_id, all_gen_trjs_id_our)
    calc_metrics(all_real_trjs_id, all_gen_trjs_id_rand)
    calc_metrics(all_real_trjs_id, all_gen_trjs_id_A)

# 18.05207963028795
# 17.060966939210807
# 20.699608958407396
# 20.117134731603272
# avg EDT 44329896907, hausdorff 3513192686968, dtw 724421229001
# avg EDT 622822609314, hausdorff 5895305347543, dtw 411168637694
# avg EDT 920014219693, hausdorff 2805480431195, dtw 849849722228
    
#     14.9088
# 15.0526
# 20.9166
# 20.7913
# avg EDT 4, hausdorff 1073176491477, dtw 281033047532
# avg EDT 2, hausdorff 6113087681404, dtw 182439875694
# avg EDT 8, hausdorff 4660124343523, dtw 383249182556
    real_sequence_dict = find_frequent_subsequences(all_real_trjs_id, min_support, min_length)
    gen_sequence_dict_our = find_frequent_subsequences(all_gen_trjs_id_our, min_support, min_length)
    gen_sequence_dict_rand = find_frequent_subsequences(all_gen_trjs_id_rand, min_support, min_length)
    gen_sequence_dict_A = find_frequent_subsequences(all_gen_trjs_id_A, min_support, min_length)
    
    real_id_traj_path = "/data/like/TrajGen/data/{}/test_real_id.npy".format(dataset_name)
    np.save(real_id_traj_path, real_sequence_dict, allow_pickle=True)
    gen_id_traj_path = "/data/like/TrajGen/data/{}/test_gen_id_our.npy".format(dataset_name)
    np.save(gen_id_traj_path, gen_sequence_dict_our, allow_pickle=True)
    gen_id_traj_path = "/data/like/TrajGen/data/{}/test_gen_id_rand.npy".format(dataset_name)
    np.save(gen_id_traj_path, gen_sequence_dict_rand, allow_pickle=True)
    gen_id_traj_path = "/data/like/TrajGen/data/{}/test_gen_id_A.npy".format(dataset_name)
    np.save(gen_id_traj_path, gen_sequence_dict_A, allow_pickle=True)
    # 3. compute the hit ratio between the real patterns and generated patterns
    our_des_hit = calc_destination_hit_ratio(all_real_trjs_id, all_gen_trjs_id_our)
    rand_des_hit = calc_destination_hit_ratio(all_real_trjs_id, all_gen_trjs_id_rand)
    A_des_hit = calc_destination_hit_ratio(all_real_trjs_id, all_gen_trjs_id_A)
    print(f"Destination hit ratio, our: {our_des_hit}  rand: {rand_des_hit}  A: {A_des_hit}")
    
    print(f"\nOur Generated Pattern: {len(gen_sequence_dict_our)}  Rand: {len(gen_sequence_dict_rand)} A: {len(gen_sequence_dict_A)}")
    our_pattern_hit = calc_pattern_hit_ratio(real_sequence_dict, gen_sequence_dict_our)
    rand_pattern_hit = calc_pattern_hit_ratio(real_sequence_dict, gen_sequence_dict_rand)
    A_pattern_hit = calc_pattern_hit_ratio(real_sequence_dict, gen_sequence_dict_A)