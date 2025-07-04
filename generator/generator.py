import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from generator.function_g_fc import FunctionGFC


class Generator(nn.Module):
    """第二版生成器
    将第一版生成器作为轨迹连贯性预测模块，并引入目的导向模块

    Args:
        nn.Module (torch.nn.Module): 继承 pytorch Module 接口类
    """

    def __init__(self, config, data_feature):
        """模型初始化

        Args:
            config (dict): 配置字典
            data_feature (dict): 数据集相关参数
        """
        super(Generator, self).__init__()
        # 模型参数
        self.function_g = FunctionGFC(config['function_g'], data_feature)

    def forward(self, trace_loc, trace_time,  candidate_set,  trace_mask=None):
        """前馈过程

        Args:
            trace_loc (tensor): 当前轨迹位置序列. (batch_size, seq_len)
            trace_time (tensor): 当前轨迹时间序列. (batch_size, seq_len)
            candidate_set (tensor): 候选下一跳集合. (batch_size, candidate_size)
            trace_mask (tensor, optional): 轨迹 padding mask, 1 表示非补齐值, 0 表示为补齐值. Defaults to None. (batch_size, seq_len)
            cache (bool): 是否缓存加速

        Return:
            candidate_prob (tensor): 对候选集下一跳的概率预测 (batch_size, candidate_size)
        """
        g_score = self.function_g.forward(trace_loc, trace_time,  candidate_set, trace_mask)
        return g_score

    def predict(self, trace_loc, trace_time,  candidate_set, trace_mask=None):
        """预测

        Args:
            trace_loc (tensor): 当前轨迹位置序列. (batch_size, seq_len)
            trace_time (tensor): 当前轨迹时间序列. (batch_size, seq_len)
            candidate_set (tensor): 候选下一跳集合. (batch_size, candidate_size)
            trace_mask (tensor, optional): 轨迹 padding mask, 1 表示非补齐值, 0 表示为补齐值. Defaults to None. (batch_size, seq_len)
        Return:
            candidate_prob (tensor): softmax 后对候选集下一跳的概率预测 (batch_size, candidate_size)
        """
        score = self.forward(trace_loc, trace_time,  candidate_set,  trace_mask)
        return torch.softmax(score, dim=1)

