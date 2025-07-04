# CA-Gen: Privacy Trajectory Generation with Co-Movement Awareness

This implementation follows the content of the article "CA-Gen: Privacy Trajectory Generation with Co-Movement Awareness" published in the GeoInformatica: Special Issue on Spatial Data Generation.

## Environment

Ubuntu 11

Python 3.8.20

## OpenSource Dataset
Beijing_Taxi Trajectory Dataset: https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/

Porto Trajectory Dataset: https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data

## pre-process
first, run `PortoClean.java` and `Geolife.java` to filter abnormal trajectories and locations, we have

/data/like/clean/Porto.xyt

/data/like/clean/Beijing.xyt

## Quick Start: How to apply our proposed framework 

1. run `generate_encoded_trajectory` to 
- map trajectory lon/lat/time to hot_loc_id/time_encode

    Encoded trajectories is stored in `/data/like/TrajGen/data/{dataset_name}/all_trajectory.csv`

    e.g., ["trj_list", "time_list"]-> "[264, 232, 158]","[180, 181, 181]"

    Hot word coordinates is stored in `/data/like/TrajGen/data/{dataset_name}/rid_gps.json`

    e.g., "11": [-8.5855, 41.149]

- split the train and test datasets

    Train data is stored in `/data/like/TrajGen/data/{dataset_name}/train.csv`

    Test data is stored in `/data/like/TrajGen/data/{dataset_name}/test.csv`

- pre-compute distance among hot locations

    Distance matrix is strored in `/data/like/TrajGen/data/{dataset_name}/distance.npy`

- generate adjacent table

    Adjacent Table is stored in `/data/like/TrajGen/data/{dataset_name}/adjacent_list.json`

    e.g., "710": [16394, 5133, 14874]

- pre-compute all frequent patterns

    Co-movement patterns are stored in `/data/like/TrajGen/data/{dataset_name}/patterns.npy`

 Porto Results

- The number of valid trajectories: 949063
- The number of hot words: 17823
- The number of trajectories in train, test:  Train: 759250 Test: 189813
- Found 317015 co-movement patterns
- Write 65367 frequent seqeunces dict to path: /data/like/TrajGen/data/Porto/patterns.npy

Beijing  Results

- The number of valid trajectories: 28130
- The number of hot words: 11647
- The number of trajectories in train, test:  Train: 22504 Test: 5626
- Found 1003145 co-movement patterns
- Write 4769 frequent seqeunces dict to path: /data/like/TrajGen/data/Porto/patterns.npy

2. run `generate_pretrain_data.py` to get train/eval/test inputs for pretraining the generator
    - inputs: /data/like/TrajGen/data/{dataset_name}/train.csv
              /data/like/TrajGen/data/{dataset_name}/test.csv
    - outputs:
             /data/like/TrajGen/data/{dataset_name}/pre_train.csv
             /data/like/TrajGen/data/{dataset_name}/pre_test.csv
             /data/like/TrajGen/data/{dataset_name}/pre_eval.csv

3. run `pretrain_generator.py` to pre-train the generator
    - inputs:
             /data/like/TrajGen/data/{dataset_name}/pre_train.csv
             /data/like/TrajGen/data/{dataset_name}/pre_test.csv
             /data/like/TrajGen/data/{dataset_name}/pre_eval.csv
    - outputs:
            /data/like/TrajGen/model/{dataset_name}/temp/..
            /data/like/TrajGen/model/Porto/save/function_g_fc.pt

4. run `formal_train.py` for constrastive learning
    - inputs: 
            /data/like/TrajGen/data/{dataset_name}/train.csv
            /data/like/TrajGen/data/{dataset_name}/test.csv
    - outputs:
            /data/like/TrajGen/model/Porto/save/adversarial_discriminator.pt
            /data/like/TrajGen/model/Porto/save/adversarial_generator.pt

5. run `generate_trajectories.py` to generate trajectories based on the OD-input from the test dataset
    - inputs:
           /data/like/TrajGen/model/Porto/save/adversarial_generator.pt
           /data/like/TrajGen/data/{dataset_name}/test.csv
    - outputs:
           /data/like/TrajGen/data/{}/test_real_lonlat.xyt
           /data/like/TrajGen/data/{}/gen_real_lonlat.xyt

6. evaluate 
    run `evaluate.py` to get hit ratios of destinations  and co-movement patterns

6. Visualization
    run `utils/figure.py` to plot trajectory distributions