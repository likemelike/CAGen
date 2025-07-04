import torch, random, copy
import numpy as np

class Searcher(object):

    def __init__(self, device, adjacent_list, rid_gps, dataset_name):
        self.device = device
        self.adjacent_list = adjacent_list
        self.rid_gps = rid_gps
        save_path = f"/data/like/TrajGen/data/{dataset_name}/distance.npy"
        self.distance_matrix = np.load(save_path)
        pattern_path = f"/data/like/TrajGen/data/{dataset_name}/patterns.npy"
        self.patterns = np.load(pattern_path, allow_pickle=True).item()

    def waypoints_search(self, gen_model, trace_loc, trace_tim, des_id):
        """
        Args:
            gen_model: The trajectory generation model
            trace_loc: Known trajectory location sequence
            trace_tim: Known trajectory time sequence
            des_id: Destination ID
        Returns:
            trace_loc: Generated trajectory's RID sequence
            trace_tim: Generated trajectory's timestamp sequence
        """
        # Retrieve precomputed frequent intermediate sequences from the database
        select_intermediate_point = []
        if (trace_loc[0], des_id) in self.patterns:
            frequence_seq_and_support = self.patterns[(trace_loc[0], des_id)]
            select_intermediate_point = random.choice(frequence_seq_and_support)
            
        trace_loc = copy.deepcopy(trace_loc)
        trace_tim = copy.deepcopy(trace_tim)
        off_time = 0
        now_rid = trace_loc[-1]
        next_time = trace_tim[-1]
        
        if select_intermediate_point:
            des_ids = select_intermediate_point[0][1:]
        else:
            des_ids = [des_id]

        for des_id in des_ids:
            step = 0
            max_step = random.randint(10, 30)
            while now_rid != des_id and step < max_step:
                if str(now_rid) in self.adjacent_list:
                    candidate_set = self.adjacent_list[str(now_rid)]
                    candidate_dis_to_des = []
                    now_dis_to_candidate = []
                    
                    for c_id in candidate_set:
                        candidate_gps = self.rid_gps[str(c_id)]
                        d1 = self.distance_matrix[now_rid, c_id]
                        d2 = self.distance_matrix[c_id, des_id]
                        now_dis_to_candidate.append(d1)
                        candidate_dis_to_des.append(d2)

                    now_dis_to_candidate = torch.tensor(now_dis_to_candidate, dtype=torch.float32).to(self.device)
                    inv_dis = 1.0 / (now_dis_to_candidate + 1e-9)  # Avoid division by zero
                    can_dis_based_prob = inv_dis / torch.sum(inv_dis)  # Normalize

                    candidate_dis_to_des = torch.tensor(candidate_dis_to_des, dtype=torch.float32).to(self.device)
                    inv_dis = 1.0 / (candidate_dis_to_des + 1e-9)  # Avoid division by zero
                    des_dis_based_prob = inv_dis / torch.sum(inv_dis)  # Normalize

                    if max(now_dis_to_candidate) > 20:
                        exit(0)

                    # Prepare model input
                    trace_loc_tensor = torch.LongTensor(trace_loc).to(self.device).unsqueeze(0)
                    trace_tim_tensor = torch.LongTensor(trace_tim).to(self.device).unsqueeze(0)
                    candidate_set_tensor = torch.LongTensor(candidate_set).to(self.device).unsqueeze(0)

                    candidate_prob = gen_model.predict(
                        trace_loc=trace_loc_tensor,
                        trace_time=trace_tim_tensor,
                        candidate_set=candidate_set_tensor
                    )
                    candidate_prob = candidate_prob[0]

                    r_id = random.randint(0, 3)
                    if r_id == 0:
                        select_candidate_index = torch.argmax(can_dis_based_prob, dim=-1)
                    elif r_id == 1:
                        select_candidate_index = torch.argmax(des_dis_based_prob, dim=-1)
                    else:
                        select_candidate_index = torch.argmax(candidate_prob * can_dis_based_prob*des_dis_based_prob, dim=-1)
                        # Alternative sampling method:
                        # select_candidate_index = torch.multinomial(candidate_prob * des_dis_based_prob, num_samples=1)

                    # Append the selected point to the trajectory
                    now_rid = candidate_set[select_candidate_index]
                    if now_rid in trace_loc:
                        now_rid = random.sample(candidate_set, 1)[0]
                    trace_loc.append(now_rid)
                    next_time += random.randint(0, 1)
                    trace_tim.append(next_time % 1440)
                    step += 1
                else:
                    print("Dead Route")
                    break
        return trace_loc, trace_tim
    
    def random_search(self, gen_model, trace_loc, trace_tim, des_id):
        trace_loc = copy.deepcopy(trace_loc)
        trace_tim = copy.deepcopy(trace_tim)
        off_time = 0
        now_rid = trace_loc[-1]
        next_time = trace_tim[-1]
        des_ids = [des_id]
        for des_id in des_ids:
            step = 0
            max_step = random.randint(10,30) 
            while now_rid != des_id and step < max_step:
                if str(now_rid) in self.adjacent_list:
                    candidate_set = self.adjacent_list[str(now_rid)]
                    trace_loc_tensor = torch.LongTensor(trace_loc).to(self.device).unsqueeze(0)
                    trace_tim_tensor = torch.LongTensor(trace_tim).to(self.device).unsqueeze(0)
                    candidate_set_tensor = torch.LongTensor(candidate_set).to(self.device).unsqueeze(0)
                    candidate_prob = gen_model.predict(trace_loc=trace_loc_tensor, trace_time=trace_tim_tensor, candidate_set=candidate_set_tensor)
                    candidate_prob = candidate_prob[0]
                    select_candidate_index = torch.multinomial(candidate_prob, num_samples=1)  
                    now_rid = candidate_set[select_candidate_index]
                    if now_rid in trace_loc:
                        now_rid = random.sample(candidate_set,1)[0]
                    trace_loc.append(now_rid)
                    next_time += random.randint(0,1)
                    trace_tim.append(next_time%1440)
                    step += 1
                else:
                    break
        return trace_loc, trace_tim
    
    def A_search(self, gen_model, trace_loc, trace_tim, des_id):
        trace_loc = copy.deepcopy(trace_loc)
        trace_tim = copy.deepcopy(trace_tim)
        off_time = 0
        now_rid = trace_loc[-1]
        next_time = trace_tim[-1]
        des_ids = [des_id]
        for des_id in des_ids:
            step = 0
            max_step = random.randint(10,30) 
            while now_rid != des_id and step < max_step:
                if str(now_rid) in self.adjacent_list:
                    candidate_set = self.adjacent_list[str(now_rid)]
                    candidate_dis_to_des = []
                    for c_id in candidate_set:
                        candidate_gps = self.rid_gps[str(c_id)]
                        d1 = self.distance_matrix[now_rid, c_id]
                        d2 = self.distance_matrix[c_id, des_id]
                        candidate_dis_to_des.append(d1+d2)
                    candidate_dis_to_des = torch.tensor(candidate_dis_to_des, dtype=torch.float32).to(self.device)
                    inv_dis = 1.0 / (candidate_dis_to_des + 1e-9)  
                    dis_based_prob = inv_dis / torch.sum(inv_dis)  
                    trace_loc_tensor = torch.LongTensor(trace_loc).to(self.device).unsqueeze(0)
                    trace_tim_tensor = torch.LongTensor(trace_tim).to(self.device).unsqueeze(0)
                    candidate_set_tensor = torch.LongTensor(candidate_set).to(self.device).unsqueeze(0)
                    candidate_prob = gen_model.predict(trace_loc=trace_loc_tensor, trace_time=trace_tim_tensor, candidate_set=candidate_set_tensor)
                    candidate_prob = candidate_prob[0]
                    select_candidate_index = torch.multinomial(candidate_prob*dis_based_prob, num_samples=1)
                    now_rid = candidate_set[select_candidate_index]
                    if now_rid in trace_loc:
                        now_rid = random.sample(candidate_set,1)[0]
                    trace_loc.append(now_rid)
                    next_time += random.randint(0,1)
                    trace_tim.append(next_time%1440)
                    step += 1
                else:
                    print("Dead Route")
                    break
        return trace_loc, trace_tim