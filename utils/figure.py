import folium, json, random, os
from folium.plugins import HeatMap
import numpy as np
import Settings
dataset_name = Settings.dataset_name
# visualize region
if dataset_name == 'Porto':
    bounds = [[41.14, -8.68], [41.18,-8.56]]
else:
    bounds = [[39.830, 116.250],[40.030,116.550]]

def overview():
    trajectories, center = loadXYT(f"/data/like/clean/{dataset_name}.xyt")
    trajectories = random.sample(trajectories, 30000)
    print(f"Overview Traj:{len(trajectories)}")
    # init a map
    m = folium.Map(
        location=center,
        zoom_start=12.8,
        control_scale=True,  
        # tiles=None 
    )
    # plot trajectories
    for traj in trajectories:
        folium.PolyLine(traj, color="red", weight=2, opacity=0.3).add_to(m)
    m.save(f"imgs/{dataset_name}/overview.html") 

# Geolife.xyt Porto.xyt
def loadXYT(read_path, max_size=100000000):
    read_lines = 0
    trjs = []
    lon_sum = 0
    lat_sum = 0
    loc_count = 0
    with open(read_path, 'r') as file:
        for line in file:
            line = line.strip()
            read_lines += 1
            if line:
                trj = []
                ts = []
                points = line.split(';')
                for i, point in enumerate(points):
                    try:
                        lon, lat, t = map(float, point.split(','))
                        lon_sum += lon
                        lat_sum += lat
                        loc_count += 1
                        trj.append([lat, lon])
                    except ValueError:
                        print(f"Omit: {point}")
            if read_lines > max_size:
                break
            trjs.append(trj)
    return trjs, [lat_sum/loc_count, lon_sum/loc_count]

def plotHotMap(method, trajectories,center):
    print(f"Plot hotmap {method}")
    all_locations = []
    for points in trajectories:
        for lat, lon in points:
            all_locations.append([lat, lon])
    
    # 创建地图,初始化地图中心和缩放比例
    m = folium.Map(
        location=center,
        zoom_start=12.8,
        control_scale=True,  # 添加比例尺控件
        # tiles=None  # 取消底图
    )
    HeatMap(
    all_locations,      # 坐标点列表 [(lat, lon), (lat, lon, weight), ...]
    min_opacity=0.2,    # 最小透明度（默认 0.2）
    radius=30,          # 点扩散半径（默认 25）
    blur=10,            # 模糊程度（默认 15）
    gradient=None,      # 颜色梯度（可自定义）
    overlay=True,       # 是否覆盖在地图图层上
    control=True,       # 是否显示控制选项
    show=True,          # 是否默认显示
    opacity=0.25        # 热度图整体透明度（范围 0.0~1.0，0.05 近乎透明）
    ).add_to(m)
    
    m.fit_bounds(bounds)
    folium.Rectangle(bounds, color="red", weight=5, fill=False, fill_color="red", fill_opacity=0).add_to(m)
    m.save(f"imgs/{dataset_name}/hotmap_{method}.html")  

def plotTrajectory(method, trajectories,center):
    print(f"Plot  sample trajectory {method}")
    m = folium.Map(
        location=center,
        zoom_start=12.8,
        control_scale=True,  
    )
    for traj in trajectories:
        folium.PolyLine(traj, color="blue", weight=1, opacity=0.4).add_to(m)
    folium.Rectangle(bounds, color="red", weight=5, fill=False, fill_color="red", fill_opacity=0).add_to(m)
    m.fit_bounds(bounds)
    m.save(f"imgs/{dataset_name}/traj_{method}.html") 
    
def plotFrequenctOverlap(method, trajectories,loaded_patterns, center):
    m = folium.Map(
        location=center,
        zoom_start=12.8,
        control_scale=True, 
    )
    
    # 绘制热度图
    all_locations = []
    for points in trajectories:
        for lat, lon in points:
            all_locations.append([lat, lon])
    HeatMap(
    all_locations,      # 坐标点列表 [(lat, lon), (lat, lon, weight), ...]
    min_opacity=0.2,    # 最小透明度（默认 0.2）
    radius=30,          # 点扩散半径（默认 25）
    blur=10,            # 模糊程度（默认 15）
    gradient=None,      # 颜色梯度（可自定义）
    overlay=True,       # 是否覆盖在地图图层上
    control=True,       # 是否显示控制选项
    show=True,          # 是否默认显示
    opacity=0.25        # 热度图整体透明度（范围 0.0~1.0，0.05 近乎透明）
    ).add_to(m)
    # 热度图上绘制 频繁模式
    with open('/data/like/TrajGen/data/{}/rid_gps.json'.format(dataset_name), 'r') as f:
            rid_gps = json.load(f)
    pattern_path = f"/data/like/TrajGen/data/{dataset_name}/test_real_id.npy"
    hot_words = set()
    for key in loaded_patterns:
        for pattern, support in loaded_patterns[key]:
            for word in pattern:
                hot_words.add(word)
    for word in hot_words:
        lonlat = rid_gps[str(word)]
        folium.CircleMarker([lonlat[1],lonlat[0]], color="blue", radius=5, fill=True,fill_opacity=0.9).add_to(m)
    # 显示地图
    m.fit_bounds(bounds)
    folium.Rectangle(bounds, color="red", weight=5, fill=False, fill_color="red", fill_opacity=0).add_to(m)
    m.save(f"imgs/{dataset_name}/frequence_{method}.html")  # 生成 HTML 文件，可在浏览器打开


if __name__ == "__main__":
    select_num = 5000
    real_lonlat_traj_path = "/data/like/TrajGen/data/{}/test_real_lonlat.xyt".format(dataset_name)
    gen_lonlat_traj_path_our = "/data/like/TrajGen/data/{}/test_gen_lonlat_our.xyt".format(dataset_name)
    gen_lonlat_traj_path_rand = "/data/like/TrajGen/data/{}/test_gen_lonlat_rand.xyt".format(dataset_name)
    gen_lonlat_traj_path_A = "/data/like/TrajGen/data/{}/test_gen_lonlat_A.xyt".format(dataset_name)
    
    # load real trajectories and generated trajectories
    trajectories_real,  center1 = loadXYT(real_lonlat_traj_path)
    trajectories_our,  center_gen = loadXYT(gen_lonlat_traj_path_our)
    trajectories_rand,  center_gen = loadXYT(gen_lonlat_traj_path_rand)
    trajectories_A,  center_gen = loadXYT(gen_lonlat_traj_path_A)
    assert len(trajectories_real) == len(trajectories_our) == len(trajectories_rand) == len(trajectories_A)
    print(f"Total Trajectories: {len(trajectories_real)}")
    # random sampling
    indices = random.sample(range(len(trajectories_real)), min(select_num,len(trajectories_real)))
    trajectories_real = [trajectories_real[i] for i in indices]
    trajectories_our = [trajectories_our[i] for i in indices]
    trajectories_rand = [trajectories_rand[i] for i in indices]
    trajectories_A = [trajectories_A[i] for i in indices]
    print(f"Sample Trajectories: {len(trajectories_A)}")
    # plot overview
    # overview()
    
    # plot hotmap
    plotHotMap("real", trajectories_real, center1)
    plotHotMap("our", trajectories_our, center1)
    plotHotMap("rand", trajectories_rand, center1)
    plotHotMap("A", trajectories_A, center1)
    
    # plot trajectory distribution
    plotTrajectory("real", trajectories_real, center1)
    plotTrajectory("our", trajectories_our, center1)
    plotTrajectory("rand", trajectories_rand, center1)
    plotTrajectory("A", trajectories_A, center1)
    
    # plot co-movement pattern distribution
    real_pattern_path = "/data/like/TrajGen/data/{}/test_real_id.npy".format(dataset_name)
    gen_pattern_path_our = "/data/like/TrajGen/data/{}/test_gen_id_our.npy".format(dataset_name)
    gen_pattern_path_rand = "/data/like/TrajGen/data/{}/test_gen_id_rand.npy".format(dataset_name)
    gen_pattern_path_A = "/data/like/TrajGen/data/{}/test_gen_id_A.npy".format(dataset_name)
    
    real_patterns = np.load(real_pattern_path, allow_pickle=True).item()
    gen_patterns_our = np.load(gen_pattern_path_our, allow_pickle=True).item()
    gen_patterns_rand = np.load(gen_pattern_path_rand, allow_pickle=True).item()
    gen_patterns_A = np.load(gen_pattern_path_A, allow_pickle=True).item()
    plotFrequenctOverlap("real", trajectories_real, real_patterns, center1)
    plotFrequenctOverlap("our", trajectories_our, gen_patterns_our, center1)
    plotFrequenctOverlap("rand", trajectories_rand, gen_patterns_rand, center1)
    plotFrequenctOverlap("A", trajectories_A, gen_patterns_A, center1)
    
    
    