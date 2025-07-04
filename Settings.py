dataset_name = 'Geolife'
dataset_name = 'Porto'
road_num =  17823 if dataset_name == "Porto" else 11647
device = 'cuda:1'
clip = 5.0
learning_rate = 0.0005
time_size = 1440
loc_pad = road_num
time_pad = time_size
data_feature = {
    'road_num': road_num + 1,
    'time_size': time_size + 1,
    'road_pad': loc_pad,
    'time_pad': time_pad,
}
# 生成器 config
gen_config = {
    "function_g": {
        "road_emb_size": 256,  # 需要和路网表征预训练部分维度一致
        "time_emb_size": 50,
        "hidden_size": 256,
        "dropout_p": 0.6,
        "lstm_layer_num": 2,
        "pretrain_road_rep": None,
        "device": device
    }
}
# 判别器 config
dis_config = {
    "road_emb_size": 256,  # 需要和路网表征预训练部分维度一致
    "hidden_size": 256,
    "dropout_p": 0.6,
    "lstm_layer_num": 2,
    "pretrain_road_rep": None,
    "device": device
}