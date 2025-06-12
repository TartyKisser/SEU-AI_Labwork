# Foundation packages
import torch
import pandas as pd
import numpy as np

# packages for data processing
import utils
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline

use_cuda = True
device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")


# input data
occ, dur, vol, prc, adj, col, dis, weather, inf = utils.read_data()
zone = 42
timestamp = 1024  # note: timestamp > 12

from datetime import datetime

def time_to_index_optimized(time_str):
    """
    将时间字符串直接转换为CSV数据中的行索引，无需遍历CSV
    
    参数:
    time_str (str): 时间字符串，格式为 'YYYY-MM-DD HH:MM:SS' 或 'YYYY/MM/DD HH:MM'
    
    返回:
    int: 对应的行索引，如果时间超出范围则返回-1
    """
    # 标准化时间格式
    try:
        if '/' in time_str:
            # 处理格式为 'YYYY/MM/DD HH:MM' 或 'YYYY/MM/DD HH:MM:SS'
            if len(time_str.split(':')) == 2:  # 没有秒
                input_time = datetime.strptime(time_str, '%Y/%m/%d %H:%M')
            else:  # 有秒
                input_time = datetime.strptime(time_str, '%Y/%m/%d %H:%M:%S')
        else:
            # 处理格式为 'YYYY-MM-DD HH:MM' 或 'YYYY-MM-DD HH:MM:SS'
            if len(time_str.split(':')) == 2:  # 没有秒
                input_time = datetime.strptime(time_str, '%Y-%m-%d %H:%M')
            else:  # 有秒
                input_time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
    except ValueError as e:
        print(f"时间格式错误: {e}")
        return -1
    
    # 数据开始和结束时间
    start_time = datetime.strptime('2022-09-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime('2023-02-28 23:00:00', '%Y-%m-%d %H:%M:%S')
    
    # 检查输入时间是否在范围内
    if input_time < start_time or input_time > end_time:
        print(f"时间 {time_str} 超出数据范围 ({start_time} 到 {end_time})")
        return -1
    
    # 计算时间差（小时数）
    time_diff = (input_time - start_time).total_seconds() / 3600
    
    # 根据时间差计算行索引（假设每小时一条数据，行索引从0开始）
    # 四舍五入到最接近的整数，以处理可能的秒数差异
    row_index = round(time_diff) + 1
    
    return row_index

def prompting_with_time_optimized(time_str):
    """
    使用优化的时间索引方法，基于输入的时间字符串生成提示模板
    
    参数:
    time_str (str): 时间字符串
    zone, inf, data, prc, weather: 与原prompting函数相同的参数
    length, future: 可选参数，与原prompting函数相同
    
    返回:
    str: 生成的提示模板或错误信息
    """
    # 获取时间戳（行索引）
    timestamp = time_to_index_optimized(time_str)
    if timestamp == -1:
        return f"错误: 无法获取时间 '{time_str}' 的索引"
    
    return timestamp

# 使用示例
timestamp = prompting_with_time_optimized('2022-10-13 15:00:00')
print(timestamp)
input_prompt = utils.prompting(zone, timestamp, inf, occ, prc, weather)
target = utils.output_template(np.round(occ.iloc[timestamp+6, zone], decimals=4))
print(input_prompt)

model, tokenizer, config = utils.load_llm()  # load model and tokenizer

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map=device)
messages = [
    {"role": "user", "content": input_prompt}
]

outputs = pipe(
    messages,
    max_new_tokens=128,
    do_sample=True
)
print(outputs[0]["generated_text"][-1])
print('Groundtruth =', target)
