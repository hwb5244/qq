import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import os
import csv

# ==============================================
# 页面基础配置
# ==============================================
st.set_page_config(page_title="快乐8专业数据分析系统", page_icon="🎰", layout="wide")

# ==============================================
# 全局常量定义（底层库通用配置）
# ==============================================
DATA_FILE = "kl8_history_data.csv"
# 1-80质数列表（质合计算用）
PRIME_NUMBERS = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79]
# 区间划分标准
ZONE_RULE = {
    "zone1": [1,20],
    "zone2": [21,40],
    "zone3": [41,60],
    "zone4": [61,80]
}
# 冷热温阈值系数
HOT_COLD_FACTOR = 2
# 完整88期初始数据源（沿用当前代码的数据源）
INIT_DATA = [
    ["2026001",2,5,6,11,24,25,27,32,34,35,39,41,44,51,54,62,70,71,72,75],
    ["2026002",3,8,10,17,22,24,25,28,39,51,61,62,67,69,70,71,72,73,74,80],
    ["2026003",2,7,14,16,22,25,28,31,39,42,47,53,54,55,61,68,69,72,73,78],
    ["2026004",4,5,9,13,16,21,23,24,32,35,37,38,45,50,52,54,55,62,63,64],
    ["2026005",7,8,9,14,18,21,24,26,33,35,41,43,49,54,56,59,60,63,68,76],
    ["2026006",3,5,7,9,19,28,30,32,34,38,49,52,56,61,62,66,73,76,78,79],
    ["2026007",3,13,15,18,20,21,25,32,42,43,45,54,57,62,63,68,72,74,76,80],
    ["2026008",2,4,15,20,21,23,24,34,47,50,51,52,57,58,60,61,66,71,77,79],
    ["2026009",3,4,8,17,18,31,34,37,42,46,47,55,56,61,65,70,74,75,76,80],
    ["2026010",6,7,13,16,19,27,33,37,39,42,43,44,55,59,62,64,65,67,76,80],
    ["2026011",1,3,12,16,22,25,27,30,32,49,52,56,59,61,62,63,66,68,69,79],
    ["2026012",4,11,12,15,16,20,21,26,27,28,30,32,33,41,53,60,62,64,65,76],
    ["2026013",1,5,9,10,11,12,14,15,16,22,28,32,37,41,44,64,72,77,78,80],
    ["2026014",6,12,13,14,18,24,28,29,30,34,38,43,49,52,59,60,64,74,78,80],
    ["2026015",2,8,9,11,14,17,18,19,27,29,31,34,36,41,55,60,64,70,72,79],
    ["2026016",4,19,24,27,28,35,36,38,39,45,50,56,58,59,60,61,64,66,73,78],
    ["2026017",11,15,28,33,36,37,38,46,47,50,51,52,61,62,63,67,68,71,73,80],
    ["2026018",11,12,14,15,18,21,22,23,29,32,33,35,40,41,55,65,69,74,75,78],
    ["2026019",16,20,21,23,24,37,48,51,52,53,54,59,60,62,66,68,75,76,77,78],
    ["2026020",7,13,18,22,24,30,32,33,37,43,47,48,52,62,63,67,69,73,78,79],
    ["2026021",4,5,6,9,10,11,14,19,21,31,34,43,45,48,51,53,56,60,65,70],
    ["2026022",2,9,10,11,15,16,17,18,28,33,35,43,54,55,60,61,63,65,68,73],
    ["2026023",9,13,17,27,28,31,36,39,42,45,50,55,57,61,67,68,73,74,75,79],
    ["2026024",3,8,16,19,22,25,28,35,36,37,49,50,52,54,57,58,64,67,71,73],
    ["2026025",2,3,4,6,8,13,17,20,21,25,26,29,34,40,47,49,51,58,59,69],
    ["2026026",2,6,14,17,21,22,26,34,35,36,48,51,52,58,60,62,71,73,75,76],
    ["2026027",1,4,7,14,18,26,28,37,38,49,51,58,61,62,64,67,71,75,76,79],
    ["2026028",9,11,14,16,17,20,28,30,32,33,46,51,54,57,58,62,66,68,71,74],
    ["2026029",1,4,11,15,19,27,29,31,33,39,43,45,52,58,65,67,69,73,75,77],
    ["2026030",2,8,12,13,14,16,23,31,37,39,44,46,50,54,55,59,61,69,70,73],
    ["2026031",8,10,15,22,25,26,35,42,43,45,47,52,56,57,62,63,72,76,78,80],
    ["2026032",2,4,9,22,24,26,27,28,31,34,36,39,43,44,53,55,59,62,65,70],
    ["2026033",2,3,8,10,16,19,20,21,29,30,33,34,36,48,53,54,55,58,59,76],
    ["2026034",4,6,19,25,26,29,30,31,32,43,44,48,50,62,65,66,68,71,72,80],
    ["2026035",5,7,18,19,23,24,29,32,34,47,48,53,57,62,64,65,70,72,77,80],
    ["2026036",3,11,19,27,32,33,35,38,43,46,49,50,63,64,66,67,74,75,76,77],
    ["2026037",6,7,8,17,24,25,27,28,29,34,40,41,49,53,57,58,60,67,69,72],
    ["2026038",3,4,5,6,7,9,11,20,21,26,36,44,46,47,54,68,73,74,75,80],
    ["2026039",3,5,13,17,22,27,36,38,40,43,49,50,51,61,62,64,68,73,78,80],
    ["2026040",12,13,16,24,31,32,35,37,41,43,48,50,51,55,64,66,68,73,76,79],
    ["2026041",8,12,17,18,23,24,25,26,27,28,30,31,38,41,46,56,66,68,69,79],
    ["2026042",2,8,14,15,16,17,25,34,37,41,47,52,56,60,62,63,72,74,77,79],
    ["2026043",1,6,12,15,16,19,22,23,25,28,32,34,35,36,41,48,56,64,65,77],
    ["2026044",1,3,8,9,13,16,19,23,33,43,47,49,52,53,55,60,68,74,76,79],
    ["2026045",2,3,5,9,11,12,13,18,19,25,35,38,45,57,61,65,68,70,72,76],
    ["2026046",1,10,13,14,15,23,25,26,28,48,49,50,52,54,59,61,64,69,70,79],
    ["2026047",5,14,16,21,22,28,31,34,37,40,50,51,55,62,63,64,74,75,77,80],
    ["2026048",3,4,10,12,16,17,21,25,33,35,41,43,49,50,60,65,69,75,76,80],
    ["2026049",1,6,7,8,11,17,28,35,36,39,41,42,44,48,53,60,61,69,72,75],
    ["2026050",12,14,25,34,39,41,43,45,46,49,51,57,58,60,62,64,65,66,72,74],
    ["2026051",6,7,9,20,21,28,31,32,33,34,36,39,41,47,56,62,67,70,71,77],
    ["2026052",2,6,11,13,19,23,27,29,32,34,40,45,49,51,56,59,66,69,74,80],
    ["2026053",2,3,7,14,20,22,23,25,26,27,28,30,35,36,38,41,44,63,67,68],
    ["2026054",2,8,11,14,15,18,21,23,39,40,42,43,44,51,55,60,65,71,79,80],
    ["2026055",3,4,8,10,12,16,19,22,24,26,31,49,54,55,58,63,64,67,75,79],
    ["2026056",1,8,14,16,25,39,40,48,50,51,56,58,65,67,69,71,73,74,75,79],
    ["2026057",4,8,12,13,17,21,26,27,30,31,35,38,43,50,52,57,60,72,79,80],
    ["2026058",3,5,8,17,19,20,21,27,29,42,47,49,51,53,62,66,69,71,74,80],
    ["2026059",3,10,18,22,24,26,28,30,37,41,46,47,48,51,57,63,69,70,75,76],
    ["2026060",2,3,4,14,15,19,25,28,32,38,39,40,42,45,48,60,61,63,65,67],
    ["2026061",4,5,6,7,13,18,20,21,28,32,34,39,40,42,44,47,50,56,75,76],
    ["2026062",3,4,5,7,14,16,22,27,38,43,46,48,56,58,68,71,73,77,79,80],
    ["2026063",2,4,8,9,14,30,33,35,37,38,40,47,52,59,62,65,73,75,76,79],
    ["2026064",13,16,21,25,31,32,38,45,49,57,58,59,61,63,67,69,72,75,77,78],
    ["2026065",1,2,4,7,9,11,14,16,24,32,34,37,48,49,50,52,53,59,63,70],
    ["2026066",2,3,4,6,8,17,20,27,30,37,41,43,54,56,57,58,60,63,65,69],
    ["2026067",10,11,12,13,14,16,19,23,24,26,39,42,43,50,53,61,62,66,75,80],
    ["2026068",1,4,10,18,19,20,22,31,35,36,44,49,50,52,61,67,68,74,75,78],
    ["2026069",1,5,6,9,15,19,21,25,31,40,47,50,54,56,59,62,65,67,72,80],
    ["2026070",1,3,4,7,11,19,25,27,37,38,42,45,46,48,54,56,69,72,76,77],
    ["2026071",2,3,8,12,13,14,15,21,25,29,42,44,45,46,47,48,52,58,63,66],
    ["2026072",1,5,19,22,28,30,34,37,38,40,41,44,48,50,51,54,59,74,75,80],
    ["2026073",2,3,6,8,9,20,26,27,29,35,52,57,64,67,73,74,75,76,79,80],
    ["2026074",2,5,7,8,11,13,15,16,20,31,32,33,41,43,52,62,70,71,72,73],
    ["2026075",1,4,17,18,21,22,23,24,25,30,41,47,48,50,54,55,56,57,62,78],
    ["2026076",6,10,12,14,16,21,24,38,39,40,46,48,52,58,61,69,71,74,75,77],
    ["2026077",3,6,19,21,24,32,36,37,44,45,51,55,58,61,62,66,71,73,74,75],
    ["2026078",1,6,7,11,15,21,24,27,31,32,34,38,46,49,52,60,67,71,73,74],
    ["2026079",4,10,12,23,38,39,41,44,45,50,51,55,57,60,64,66,68,73,75,76],
    ["2026080",7,11,14,24,25,27,32,33,34,46,49,52,54,56,59,60,61,66,69,72],
    ["2026081",1,2,8,9,11,12,20,32,34,39,42,43,53,54,55,58,59,63,66,77],
    ["2026082",2,5,11,16,23,25,26,29,33,39,57,62,64,68,69,74,76,77,79,80],
    ["2026083",1,2,7,13,14,18,23,27,28,38,41,48,51,54,63,65,68,71,75,78],
    ["2026084",5,8,17,33,34,35,38,42,43,46,49,50,57,59,60,66,71,72,74,80],
    ["2026085",1,3,4,11,13,14,26,31,33,38,40,50,56,58,61,63,65,66,69,79],
    ["2026086",1,3,5,7,11,15,17,24,28,29,34,39,45,47,53,57,63,66,72,79],
    ["2026087",2,5,7,10,14,23,24,32,37,41,42,44,45,46,48,63,64,71,73,76],
    ["2026088",8,9,13,14,18,22,25,33,39,40,44,46,49,55,64,66,68,71,77,80]
]

# ==============================================
# 【模块代码_快乐8分析底层库.py】核心模块嵌入
# ==============================================

# ----------------------
# 底层模块1：数据持久化与校验模块
# ----------------------
def load_data():
    """加载历史数据，自动去重，保证期号唯一性（三重保障第一重）"""
    if not os.path.exists(DATA_FILE):
        # 初始化数据文件
        with open(DATA_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['period'] + [f'n{i}' for i in range(1,21)])
            writer.writerows(INIT_DATA)
    # 读取数据
    df = pd.read_csv(DATA_FILE, dtype={'period': str})
    # 自动去重：确保期号唯一，保留最新的记录
    df = df.drop_duplicates(subset=['period'], keep='last')
    # 按期号降序排列，最新的在最前面
    df = df.sort_values('period', ascending=False).reset_index(drop=True)
    return df

def save_new_data(period, numbers):
    """保存新期数据到本地文件"""
    numbers = sorted(numbers)
    with open(DATA_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([period] + numbers)
    return True

def validate_period_unique(period, df):
    """校验期号唯一性（三重保障第二重：录入前置校验）"""
    if period in df['period'].values:
        return False, "期号已存在，禁止重复录入"
    if not period.isdigit():
        return False, "期号必须为合法数字"
    return True, "校验通过"

def validate_numbers(numbers):
    """校验开奖号码合法性"""
    try:
        nums = [int(x) for x in numbers]
        if len(nums) != 20:
            return False, "必须输入20个号码"
        if len(set(nums)) != 20:
            return False, "号码不能重复"
        if max(nums) > 80 or min(nums) < 1:
            return False, "号码必须在1-80之间"
        return True, sorted(nums)
    except:
        return False, "号码格式错误，必须为数字"

# ----------------------
# 底层模块2：核心分析引擎（全量数据统一分析入口）
# ----------------------
def analyze_full_data(df, window=None):
    """
    全量数据分析核心引擎，返回所有分析结果
    :param df: 数据源DataFrame
    :param window: 分析周期，None为全量数据
    :return: 全量分析结果字典
    """
    if window:
        data = df.head(window).copy()
    else:
        data = df.copy()
    
    # 基础数据处理
    numbers_list = []
    for _, row in data.iterrows():
        nums = row[1:].tolist()
        numbers_list.append(nums)
    flat_nums = [n for p in numbers_list for n in p]
    total_periods = len(data)
    avg_appear = len(flat_nums) / 80  # 单号码平均出现次数

    # 调用子模块分析
    hot_cold_result = calc_hot_cold(flat_nums)
    miss_analysis = calc_miss_analysis(numbers_list, total_periods)
    co_occur_matrix = calc_co_occur_matrix(numbers_list)
    follow_matrix = calc_follow_matrix(numbers_list)
    road_distribution = calc_road_distribution(flat_nums)
    zone_distribution = calc_zone_distribution(flat_nums)
    consecutive_stats = calc_consecutive_stats(numbers_list)

    return {
        'numbers_list': numbers_list,
        'flat_nums': flat_nums,
        'total_periods': total_periods,
        'avg_appear': avg_appear,
        'hot_cold': hot_cold_result,
        'miss_analysis': miss_analysis,
        'co_occur_matrix': co_occur_matrix,
        'follow_matrix': follow_matrix,
        'road_distribution': road_distribution,
        'zone_distribution': zone_distribution,
        'consecutive_stats': consecutive_stats
    }

# ----------------------
# 底层模块3：冷热号分析模块
# ----------------------
def calc_hot_cold(flat_nums):
    """计算冷热号统计"""
    counter = Counter(flat_nums)
    hot_top10 = counter.most_common(10)
    cold_top10 = counter.most_common()[-10:]
    cold_top10.reverse()
    full_counter = {n: counter.get(n, 0) for n in range(1, 81)}
    return {
        'hot_top10': hot_top10,
        'cold_top10': cold_top10,
        'full_counter': full_counter
    }

# ----------------------
# 底层模块4：遗漏值深度分析模块
# ----------------------
def calc_miss_analysis(numbers_list, total_periods):
    """深度计算遗漏值相关指标"""
    last_appear = {}
    miss_current = {}
    miss_avg = {}
    miss_max = {}
    all_miss = defaultdict(list)
    
    for idx, nums in enumerate(numbers_list):
        for n in nums:
            if n in last_appear:
                miss = idx - last_appear[n]
                all_miss[n].append(miss)
            last_appear[n] = idx
    
    for n in range(1, 81):
        # 当前遗漏
        if n in last_appear:
            miss_current[n] = total_periods - 1 - last_appear[n]
        else:
            miss_current[n] = total_periods
        # 平均/最大遗漏
        if all_miss[n]:
            miss_avg[n] = np.mean(all_miss[n])
            miss_max[n] = max(all_miss[n])
        else:
            miss_avg[n] = 0
            miss_max[n] = 0
        # 回补概率（简易模型）
        if miss_avg[n] > 0:
            back_prob = min(100, (miss_current[n] / miss_avg[n]) * 100)
        else:
            back_prob = 0
    
    # 生成DataFrame
    miss_df = pd.DataFrame({
        '号码': range(1, 81),
        '当前遗漏': [miss_current[n] for n in range(1, 81)],
        '平均遗漏': [f"{miss_avg[n]:.1f}" for n in range(1, 81)],
        '最大遗漏': [miss_max[n] for n in range(1, 81)],
        '出现次数': [len(all_miss[n]) + 1 if n in last_appear else 0 for n in range(1, 81)],
        '回补概率': [f"{back_prob:.1f}%" for n in range(1, 81)]
    }).sort_values('当前遗漏', ascending=False)
    
    return {
        'miss_df': miss_df,
        'miss_current': miss_current,
        'miss_avg': miss_avg,
        'miss_max': miss_max
    }

# ----------------------
# 底层模块5：相随号/跟随号矩阵模块
# ----------------------
def calc_co_occur_matrix(numbers_list):
    """计算全量相随号同现矩阵"""
    co_occur = defaultdict(int)
    for p in numbers_list:
        for i in range(20):
            for j in range(i+1, 20):
                a, b = p[i], p[j]
                co_occur[(a, b)] += 1
    # 按同现次数排序
    co_sorted = sorted(co_occur.items(), key=lambda x: x[1], reverse=True)
    return {
        'co_occur_dict': co_occur,
        'co_top10': co_sorted[:10],
        'co_sorted': co_sorted
    }

def calc_follow_matrix(numbers_list):
    """计算全量跨期跟随号矩阵"""
    follow = defaultdict(int)
    for i in range(1, len(numbers_list)):
        prev_nums = numbers_list[i-1]
        curr_nums = numbers_list[i]
        for a in prev_nums:
            for b in curr_nums:
                follow[(a, b)] += 1
    # 按跟随次数排序
    follow_sorted = sorted(follow.items(), key=lambda x: x[1], reverse=True)
    return {
        'follow_dict': follow,
        'follow_top10': follow_sorted[:10],
        'follow_sorted': follow_sorted
    }

# ----------------------
# 底层模块6：号码结构分析模块
# ----------------------
def calc_road_distribution(flat_nums):
    """计算012路分布"""
    road0 = len([n for n in flat_nums if n % 3 == 0])
    road1 = len([n for n in flat_nums if n % 3 == 1])
    road2 = len([n for n in flat_nums if n % 3 == 2])
    total = len(flat_nums)
    return {
        'road0': road0,
        'road1': road1,
        'road2': road2,
        'road0_rate': f"{road0/total*100:.1f}%",
        'road1_rate': f"{road1/total*100:.1f}%",
        'road2_rate': f"{road2/total*100:.1f}%"
    }

def calc_zone_distribution(flat_nums):
    """计算区间分布"""
    z1 = len([n for n in flat_nums if ZONE_RULE['zone1'][0] <= n <= ZONE_RULE['zone1'][1]])
    z2 = len([n for n in flat_nums if ZONE_RULE['zone2'][0] <= n <= ZONE_RULE['zone2'][1]])
    z3 = len([n for n in flat_nums if ZONE_RULE['zone3'][0] <= n <= ZONE_RULE['zone3'][1]])
    z4 = len([n for n in flat_nums if ZONE_RULE['zone4'][0] <= n <= ZONE_RULE['zone4'][1]])
    total = len(flat_nums)
    return {
        'zone1': z1, 'zone2': z2, 'zone3': z3, 'zone4': z4,
        'zone1_rate': f"{z1/total*100:.1f}%",
        'zone2_rate': f"{z2/total*100:.1f}%",
        'zone3_rate': f"{z3/total*100:.1f}%",
        'zone4_rate': f"{z4/total*100:.1f}%"
    }

def calc_consecutive_stats(numbers_list):
    """计算连号统计"""
    consecutive_list = []
    for p in numbers_list:
        cnt = 0
        for i in range(1, 20):
            if p[i] == p[i-1] + 1:
                cnt += 1
        consecutive_list.append(cnt)
    return {
        'avg_consecutive': np.mean(consecutive_list) if consecutive_list else 0,
        'max_consecutive': max(consecutive_list) if consecutive_list else 0,
        'min_consecutive': min(consecutive_list) if consecutive_list else 0,
        'full_list': consecutive_list
    }

def calc_number_structure(numbers, prev_numbers=None):
    """计算单组号码的结构特征（奇偶/大小/质合/连号等）"""
    numbers = sorted(numbers)
    # 奇偶
    odd = len([n for n in numbers if n % 2 == 1])
    even = 20 - odd
    # 大小
    small = len([n for n in numbers if n <= 40])
    large = 20 - small
    # 012路
    road0 = len([n for n in numbers if n % 3 == 0])
    road1 = len([n for n in numbers if n % 3 == 1])
    road2 = len([n for n in numbers if n % 3 == 2])
    # 质合
    prime = len([n for n in numbers if n in PRIME_NUMBERS])
    composite = 20 - prime
    # 和值与跨度
    sum_val = sum(numbers)
    span = numbers[-1] - numbers[0]
    # 连号
    consecutive = []
    i = 0
    while i < 19:
        if numbers[i+1] == numbers[i] + 1:
            start = numbers[i]
            while i < 19 and numbers[i+1] == numbers[i] + 1:
                i += 1
            end = numbers[i]
            consecutive.append(f"{start}-{end}")
        i += 1
    # 重号/斜连号
    repeat = [n for n in numbers if n in prev_numbers] if prev_numbers else []
    oblique = [n for n in numbers if (n-1 in prev_numbers) or (n+1 in prev_numbers)] if prev_numbers else []
    # 同尾号
    tail_counter = Counter([n % 10 for n in numbers])
    same_tail = {tail: [n for n in numbers if n%10 == tail] for tail, cnt in tail_counter.items() if cnt >=2}
    # 区间分布
    z1 = len([n for n in numbers if ZONE_RULE['zone1'][0] <= n <= ZONE_RULE['zone1'][1]])
    z2 = len([n for n in numbers if ZONE_RULE['zone2'][0] <= n <= ZONE_RULE['zone2'][1]])
    z3 = len([n for n in numbers if ZONE_RULE['zone3'][0] <= n <= ZONE_RULE['zone3'][1]])
    z4 = len([n for n in numbers if ZONE_RULE['zone4'][0] <= n <= ZONE_RULE['zone4'][1]])
    
    return {
        'numbers': numbers,
        'odd': odd, 'even': even, 'odd_even_ratio': f"{odd}:{even}",
        'small': small, 'large': large, 'size_ratio': f"{small}:{large}",
        'road0': road0, 'road1': road1, 'road2': road2, 'road_ratio': f"{road0}:{road1}:{road2}",
        'prime': prime, 'composite': composite, 'prime_composite_ratio': f"{prime}:{composite}",
        'sum': sum_val, 'span': span,
        'consecutive': consecutive, 'consecutive_count': len(consecutive),
        'repeat': repeat, 'repeat_count': len(repeat),
        'oblique': oblique, 'oblique_count': len(oblique),
        'same_tail': same_tail, 'same_tail_count': len(same_tail),
        'zone1': z1, 'zone2': z2, 'zone3': z3, 'zone4': z4, 'zone_ratio': f"{z1}:{z2}:{z3}:{z4}"
    }

# ----------------------
# 底层模块7：单期深度复盘模块
# ----------------------
def generate_deep_review(numbers, prev_numbers=None, period="未知期号"):
    """生成单期深度复盘报告"""
    structure = calc_number_structure(numbers, prev_numbers)
    review = {
        'period': period,
        'numbers_str': "、".join([f"{n:02d}" for n in structure['numbers']]),
        **structure
    }
    return review

# ----------------------
# 底层模块8：预测号码池分级模块
# ----------------------
def generate_leveled_pool(current_nums, co_occur_dict, follow_dict, num_status):
    """
    生成分级预测号码池，统计每个号码的关联出现次数
    :param current_nums: 本期开奖号码
    :param co_occur_dict: 相随号矩阵
    :param follow_dict: 跟随号矩阵
    :param num_status: 号码状态字典
    :return: 分级号码池结果
    """
    # 一级候选：本期开奖号码
    level1 = current_nums
    # 二级候选：相随号，统计出现次数
    level2_counter = Counter()
    co_map = defaultdict(list)
    for n in level1:
        temp = []
        for (a, b), cnt in co_occur_dict.items():
            if a == n:
                temp.append((b, cnt))
            elif b == n:
                temp.append((a, cnt))
        temp.sort(key=lambda x: x[1], reverse=True)
        top3 = temp[:3]
        co_map[n] = top3
        for b, _ in top3:
            level2_counter[b] += 1
    # 三级候选：跟随号，统计出现次数
    level3_counter = Counter()
    follow_map = defaultdict(list)
    for n in level2_counter.keys():
        temp = []
        for (a, b), cnt in follow_dict.items():
            if a == n:
                temp.append((b, cnt))
        temp.sort(key=lambda x: x[1], reverse=True)
        top2 = temp[:2]
        follow_map[n] = top2
        for b, _ in top2:
            level3_counter[b] += 1
    
    # 按次数分组
    def group_by_count(counter):
        groups = defaultdict(list)
        for n, cnt in counter.items():
            groups[cnt].append(n)
        # 按次数降序排序
        sorted_groups = sorted(groups.items(), key=lambda x: x[0], reverse=True)
        return sorted_groups
    
    return {
        'level1': level1,
        'level2_counter': level2_counter,
        'level3_counter': level3_counter,
        'level2_groups': group_by_count(level2_counter),
        'level3_groups': group_by_count(level3_counter),
        'co_map': co_map,
        'follow_map': follow_map
    }

# ==============================================
# 全局数据加载
# ==============================================
df = load_data()
total_periods = len(df)

# ==============================================
# 号码状态格式化工具（界面渲染用）
# ==============================================
def get_num_status_dict(full_analysis):
    """生成号码状态字典，用于界面渲染"""
    counter = full_analysis['hot_cold']['full_counter']
    avg_appear = full_analysis['avg_appear']
    hot_threshold = avg_appear + HOT_COLD_FACTOR
    cold_threshold = avg_appear - HOT_COLD_FACTOR
    
    num_status = {}
    for n in range(1, 81):
        cnt = counter.get(n, 0)
        road = n % 3
        road_name = f"{road}路" if road != 0 else "0路"
        if cnt >= hot_threshold:
            status = 'hot'
        elif cnt <= cold_threshold:
            status = 'cold'
        else:
            status = 'warm'
        num_status[n] = {
            'status': status,
            'road': road_name,
            'count': cnt
        }
    return num_status

def format_num(n, num_status):
    """格式化号码渲染，带颜色、012路、出现次数"""
    s = num_status[n]
    if s['status'] == 'hot':
        return f'<span style="color:red; font-weight:bold; margin:0 2px;">{n:02d}</span><small style="color:#666;">({s["road"]}, {s["count"]}次)</small>'
    elif s['status'] == 'cold':
        return f'<span style="color:blue; font-weight:bold; margin:0 2px;">{n:02d}</span><small style="color:#666;">({s["road"]}, {s["count"]}次)</small>'
    else:
        return f'<span style="color:black; margin:0 2px;">{n:02d}</span><small style="color:#666;">({s["road"]}, {s["count"]}次)</small>'

# ==============================================
# Streamlit界面布局（沿用当前代码的6大标签页架构）
# ==============================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 首页说明", 
    "📋 号码库管理", 
    "📊 多周期分析", 
    "🔮 选号参考",
    "📝 单期深度复盘",
    "🔄 跨期对比与预测池"
])

# ----------------------
# 标签页1：首页说明
# ----------------------
with tab1:
    st.title("🎰 福彩快乐8 专业数据分析系统")
    st.subheader(f"当前已收录数据：{total_periods}期 | 底层库版本：V2.0 融合版")
    st.divider()
    st.error("""
    ⚠️ 【重要法律与风险提醒】
    本软件仅用于历史开奖数据的统计与展示，**彩票开奖号码为完全随机事件**，
    任何历史走势、分析指标都无法预测未来结果，软件内的"选号参考"仅为娱乐性思路参考，
    不构成任何购彩建议，请理性购彩，量力而行！
    """)
    st.info("""
    软件核心功能（底层库+界面一体化）：
    1. 永久更新的号码库：支持手动录入新开奖号码，自动保存，**三重保障期号唯一**
    2. 多周期深度分析：支持近10/20/50/100期/全量数据的多维度分析，底层库统一计算
    3. 全维度指标覆盖：冷热号、遗漏值（含回补概率）、相随号/跟随号矩阵、012路、连号统计
    4. 娱乐性选号参考：基于底层库分析模型，提供多维度选号思路与参考号码
    5. 单期深度复盘：输入期号或号码，自动调用底层库生成全维度深度复盘报告
    6. 跨期对比与分级预测池：上期+本期复盘，自动生成分层级、按关联次数排序的预测号码池
    """)

# ----------------------
# 标签页2：号码库管理
# ----------------------
with tab2:
    st.header("📋 开奖号码库")
    
    # 新增数据表单
    st.subheader("➕ 录入新一期开奖号码")
    with st.form("add_form"):
        col1, col2 = st.columns(2)
        with col1:
            new_period = st.text_input("期号（如：2026089）", placeholder="例如：2026089")
        with col2:
            nums_input = st.text_input("开奖号码（20个数字，空格分隔）", placeholder="例如：01 02 03 ... 20")
        
        submit = st.form_submit_button("保存到号码库")
        if submit:
            # 调用底层库校验
            period_valid, period_msg = validate_period_unique(new_period, df)
            if not period_valid:
                st.error(f"❌ {period_msg}")
            else:
                num_valid, num_msg = validate_numbers(nums_input.strip().split())
                if not num_valid:
                    st.error(f"❌ {num_msg}")
                else:
                    # 校验通过，保存数据
                    save_new_data(new_period, num_msg)
                    st.success(f"✅ 成功录入{new_period}期数据！软件将自动刷新...")
                    st.rerun()
    
    st.divider()
    # 浏览历史数据
    st.subheader("📜 历史开奖数据")
    search = st.text_input("搜索期号")
    for i in range(0, total_periods, 10):
        end = min(i+9, total_periods-1)
        st.subheader(f"{df.iloc[end]['period']} - {df.iloc[i]['period']}")
        cols = st.columns(2)
        for j in range(5):
            if i+j >= total_periods:
                break
            row = df.iloc[i+j]
            p_num = row['period']
            if search and search not in p_num:
                continue
            nums = row[1:].tolist()
            nums_str = " ".join([f"{n:02d}" for n in nums])
            with cols[j%2]:
                st.markdown(f"**{p_num} 期**: `{nums_str}`")

# ----------------------
# 标签页3：多周期分析
# ----------------------
with tab3:
    st.header("📊 多周期数据分析")
    
    # 周期选择
    window_options = {
        "近10期": 10,
        "近20期": 20,
        "近50期": 50,
        "近100期": 100,
        "全量数据": None
    }
    selected_window = st.selectbox("选择分析周期", list(window_options.keys()))
    w = window_options[selected_window]
    
    if w and total_periods < w:
        st.warning(f"当前数据只有{total_periods}期，不足{w}期，请先补充更多数据！")
    else:
        # 调用底层库核心分析引擎
        full_analysis = analyze_full_data(df, w)
        st.info(f"当前分析：{selected_window}，共{full_analysis['total_periods']}期数据")
        
        # 1. 冷热号
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔥 热号TOP10")
            hot_df = pd.DataFrame(full_analysis['hot_cold']['hot_top10'], columns=["号码", "出现次数"])
            st.dataframe(hot_df, hide_index=True, use_container_width=True)
        with col2:
            st.subheader("❄️ 冷号TOP10")
            cold_df = pd.DataFrame(full_analysis['hot_cold']['cold_top10'], columns=["号码", "出现次数"])
            st.dataframe(cold_df, hide_index=True, use_container_width=True)
        
        # 2. 遗漏值
        st.subheader("📉 遗漏值统计（按当前遗漏降序，含回补概率）")
        st.dataframe(full_analysis['miss_analysis']['miss_df'], hide_index=True, use_container_width=True)
        
        st.divider()
        # 3. 基础分布统计
        col3, col4, col5 = st.columns(3)
        with col3:
            st.subheader("012路分布")
            road = full_analysis['road_distribution']
            st.metric("0路号码", f"{road['road0']}次", road['road0_rate'])
            st.metric("1路号码", f"{road['road1']}次", road['road1_rate'])
            st.metric("2路号码", f"{road['road2']}次", road['road2_rate'])
        
        with col4:
            st.subheader("连号统计")
            con = full_analysis['consecutive_stats']
            st.metric("平均连号数", f"{con['avg_consecutive']:.1f}个")
            st.metric("最多连号数", f"{con['max_consecutive']}个")
            st.metric("最少连号数", f"{con['min_consecutive']}个")
        
        with col5:
            st.subheader("区间分布")
            zone = full_analysis['zone_distribution']
            st.metric("1-20小号区", f"{zone['zone1']}次", zone['zone1_rate'])
            st.metric("21-40中号区", f"{zone['zone2']}次", zone['zone2_rate'])
            st.metric("41-60大号区", f"{zone['zone3']}次", zone['zone3_rate'])
            st.metric("61-80超大号区", f"{zone['zone4']}次", zone['zone4_rate'])
        
        st.divider()
        # 4. 相随号和跟随号
        col6, col7 = st.columns(2)
        with col6:
            st.subheader("👥 相随号TOP10（同现频率最高）")
            co_data = []
            for (a,b), cnt in full_analysis['co_occur_matrix']['co_top10']:
                co_data.append({"号码对": f"{a:02d} & {b:02d}", "同现次数": cnt})
            st.dataframe(pd.DataFrame(co_data), hide_index=True, use_container_width=True)
        
        with col7:
            st.subheader("👣 跟随号TOP10（跨期跟随最高）")
            follow_data = []
            for (a,b), cnt in full_analysis['follow_matrix']['follow_top10']:
                follow_data.append({"上期A→下期B": f"{a:02d} → {b:02d}", "跟随次数": cnt})
            st.dataframe(pd.DataFrame(follow_data), hide_index=True, use_container_width=True)

# ----------------------
# 标签页4：选号参考
# ----------------------
with tab4:
    st.header("🔮 选号参考（娱乐性）")
    st.warning("""
    ⚠️ 注意：以下内容仅为基于历史数据的娱乐性参考思路，**完全无法预测开奖结果**，
    彩票开奖完全随机，请仅作为娱乐参考，切勿当真！
    """)
    
    if total_periods <10:
        st.info("数据不足，无法生成参考")
    else:
        # 调用底层库分析近50期数据
        full_analysis = analyze_full_data(df, 50)
        last_period = df.iloc[0]
        last_nums = last_period[1:].tolist()
        num_status = get_num_status_dict(full_analysis)
        
        # 上期复盘
        st.subheader("📋 上期复盘")
        last_nums_formatted = " ".join([format_num(n, num_status) for n in last_nums])
        st.markdown(f"上期({last_period['period']}期)开奖号码：{last_nums_formatted}", unsafe_allow_html=True)
        
        # 计算上期的指标
        last_structure = calc_number_structure(last_nums)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("上期奇偶", f"{last_structure['odd']}奇{last_structure['even']}偶", f"历史平均:10奇10偶")
        with col2:
            st.metric("上期大小", f"{last_structure['small']}小{last_structure['large']}大", f"历史平均:10小10大")
        with col3:
            st.metric("上期012路", last_structure['road_ratio'], f"历史平均:~7/7/6")
        
        st.divider()
        # 选号思路
        st.subheader("💡 选号参考思路")
        st.markdown("""
        1. **冷热搭配**：热号代表近期活跃，冷号有回归需求，建议选3-4个热号 + 2-3个冷号搭配
        2. **遗漏回归**：优先选择当前遗漏值接近平均遗漏的号码，这类号码大概率要"回补"
        3. **012路均衡**：尽量让选的号码的012路分布接近7:7:6的历史平均
        4. **连号参考**：历史平均每期4-5个连号，建议选1-2组连号
        """)
        
        # 生成参考号码
        st.subheader("🎯 参考号码组合（娱乐性）")
        # 取热号前5，冷号前3，遗漏接近平均的前7，012路均衡
        hot_nums = [x[0] for x in full_analysis['hot_cold']['hot_top10'][:5]]
        cold_nums = [x[0] for x in full_analysis['hot_cold']['cold_top10'][:3]]
        miss_df = full_analysis['miss_analysis']['miss_df']
        miss_ok = miss_df[
            (miss_df['当前遗漏'] >= miss_df['平均遗漏'].astype(float)*0.8) &
            (miss_df['当前遗漏'] <= miss_df['平均遗漏'].astype(float)*1.2)
        ]['号码'].tolist()[:7]
        
        ref_nums = list(set(hot_nums + cold_nums + miss_ok))[:20]
        ref_nums.sort()
        ref_nums_formatted = " ".join([format_num(n, num_status) for n in ref_nums])
        st.success(f"参考号码：{ref_nums_formatted}", unsafe_allow_html=True)
        st.caption("再次提醒：这只是基于历史数据的随机参考，完全不代表会开出这些号码！")

# ----------------------
# 标签页5：单期深度复盘
# ----------------------
with tab5:
    st.header("📝 单期深度复盘")
    st.info("支持选择历史期号一键复盘，或手动输入新期号码实时生成复盘报告")
    
    # 选择复盘方式
    review_mode = st.radio("选择复盘方式", ["选择历史期号", "手动输入新期号码"], horizontal=True)
    
    if review_mode == "选择历史期号":
        # 历史期号选择器
        period_list = df['period'].tolist()
        selected_period = st.selectbox("选择要复盘的期号", period_list)
        
        if st.button("生成复盘报告", type="primary"):
            # 获取当期和上期号码
            current_row = df[df['period'] == selected_period].iloc[0]
            current_nums = current_row[1:].tolist()
            
            # 获取上期号码
            current_idx = df[df['period'] == selected_period].index[0]
            prev_nums = None
            if current_idx < len(df)-1:
                prev_row = df.iloc[current_idx+1]
                prev_nums = prev_row[1:].tolist()
            
            # 调用底层库生成复盘
            review = generate_deep_review(current_nums, prev_nums, selected_period)
            # 全量分析用于号码格式化
            full_analysis = analyze_full_data(df)
            num_status = get_num_status_dict(full_analysis)
            
            # 展示复盘报告
            st.divider()
            st.subheader(f"福彩快乐8 {review['period']}期 深度复盘报告")
            
            # 一、开奖号码
            st.markdown("### 一、官方开奖号码")
            nums_formatted = " ".join([format_num(n, num_status) for n in review['numbers']])
            st.markdown(f"**开奖号码（升序）**：{nums_formatted}", unsafe_allow_html=True)
            
            # 二、核心指标汇总
            st.markdown("### 二、核心基础指标汇总")
            metrics_df = pd.DataFrame([
                ["奇偶比", review['odd_even_ratio'], "10:10", f"偶数仅比奇数多出{review['even']-review['odd']}个" if review['even']>review['odd'] else f"奇数仅比偶数多出{review['odd']-review['even']}个"],
                ["大小比", review['size_ratio'], "10:10", "完全均衡" if review['small']==review['large'] else "微偏"],
                ["012路比", review['road_ratio'], "7:7:6", "整体均衡"],
                ["质合比", review['prime_composite_ratio'], "6:14", "合数热开" if review['composite']>14 else "质数热开"],
                ["和值", review['sum'], "810", "常规区间"],
                ["跨度", review['span'], "60", "覆盖全区间"],
                ["连号组数", review['consecutive_count'], "4.2", "连号退潮" if review['consecutive_count']<3 else "连号活跃"],
                ["重号数量", review['repeat_count'], "3.5", "重号活跃" if review['repeat_count']>4 else "正常"]
            ], columns=["指标", "本期结果", "理论均值", "核心定性"])
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
            
            # 三、奇偶分布深度拆解
            st.markdown("### 三、核心特征：奇偶分布深度拆解")
            st.markdown(f"- **奇数号码（{review['odd']}个）**：{'、'.join([f'{n:02d}' for n in review['numbers'] if n%2==1])}")
            st.markdown(f"- **偶数号码（{review['even']}个）**：{'、'.join([f'{n:02d}' for n in review['numbers'] if n%2==0])}")
            diff = abs(review['even'] - review['odd'])
            if diff == 2:
                st.markdown(f"✅ **核心特征**：**差值仅为2个**，属于极致弱偏态，无限贴近理论均衡值")
            else:
                st.markdown(f"- 奇偶差值：{diff}个")
            
            # 四、区间分布
            st.markdown("### 四、区间分布拆解")
            col_z1, col_z2, col_z3, col_z4 = st.columns(4)
            with col_z1:
                st.metric("01-20区", f"{review['zone1']}个")
            with col_z2:
                st.metric("21-40区", f"{review['zone2']}个")
            with col_z3:
                st.metric("41-60区", f"{review['zone3']}个")
            with col_z4:
                st.metric("61-80区", f"{review['zone4']}个")
            
            # 五、号码结构
            st.markdown("### 五、号码结构深度拆解")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown(f"- **连号**：{review['consecutive'] if review['consecutive'] else '无'}")
                st.markdown(f"- **重号（与上期）**：{review['repeat'] if review['repeat'] else '无'}（共{review['repeat_count']}个）")
            with col_s2:
                st.markdown(f"- **斜连号（与上期）**：{review['oblique'] if review['oblique'] else '无'}（共{review['oblique_count']}个）")
                st.markdown(f"- **同尾号**：{review['same_tail'] if review['same_tail'] else '无'}（共{review['same_tail_count']}组）")
            
            # 六、总结
            st.markdown("### 六、本期复盘总结")
            st.markdown(f"""
            1.  本期核心特征：**{review['odd_even_ratio']}的奇偶比**，差值仅为{diff}个，整体走势均衡
            2.  大小比{review['size_ratio']}，{'完全均分' if review['small']==review['large'] else '微偏'}
            3.  连号数量{review['consecutive_count']}组，{'低于历史均值' if review['consecutive_count']<4 else '高于历史均值'}
            4.  重号数量{review['repeat_count']}个，{'重号活跃' if review['repeat_count']>3 else '重号正常'}
            """)
            
            st.caption("以上仅为历史数据复盘，不构成任何购彩建议")
    
    else:
        # 手动输入新期号码
        with st.form("manual_review_form"):
            new_period = st.text_input("期号（如：2026089）", placeholder="例如：2026089")
            nums_input = st.text_input("开奖号码（20个数字，空格分隔）", placeholder="例如：08 09 13 14 ... 80")
            submit_review = st.form_submit_button("生成复盘报告")
            
            if submit_review:
                # 校验号码
                num_valid, num_msg = validate_numbers(nums_input.strip().split())
                if not num_valid:
                    st.error(f"❌ {num_msg}")
                else:
                    # 获取上期号码（最新一期）
                    prev_nums = df.iloc[0][1:].tolist() if total_periods>0 else None
                    # 调用底层库生成复盘
                    review = generate_deep_review(num_msg, prev_nums, new_period)
                    # 全量分析用于号码格式化
                    full_analysis = analyze_full_data(df)
                    num_status = get_num_status_dict(full_analysis)
                    
                    # 展示复盘报告
                    st.divider()
                    st.subheader(f"福彩快乐8 {review['period']}期 深度复盘报告")
                    
                    # 一、开奖号码
                    st.markdown("### 一、开奖号码")
                    nums_formatted = " ".join([format_num(n, num_status) for n in review['numbers']])
                    st.markdown(f"**开奖号码（升序）**：{nums_formatted}", unsafe_allow_html=True)
                    
                    # 二、核心指标汇总
                    st.markdown("### 二、核心基础指标汇总")
                    metrics_df = pd.DataFrame([
                        ["奇偶比", review['odd_even_ratio'], "10:10", f"偶数仅比奇数多出{review['even']-review['odd']}个" if review['even']>review['odd'] else f"奇数仅比偶数多出{review['odd']-review['even']}个"],
                        ["大小比", review['size_ratio'], "10:10", "完全均衡" if review['small']==review['large'] else "微偏"],
                        ["012路比", review['road_ratio'], "7:7:6", "整体均衡"],
                        ["质合比", review['prime_composite_ratio'], "6:14", "合数热开" if review['composite']>14 else "质数热开"],
                        ["和值", review['sum'], "810", "常规区间"],
                        ["跨度", review['span'], "60", "覆盖全区间"],
                        ["连号组数", review['consecutive_count'], "4.2", "连号退潮" if review['consecutive_count']<3 else "连号活跃"],
                        ["重号数量", review['repeat_count'], "3.5", "重号活跃" if review['repeat_count']>4 else "正常"]
                    ], columns=["指标", "本期结果", "理论均值", "核心定性"])
                    st.dataframe(metrics_df, hide_index=True, use_container_width=True)
                    
                    # 三、奇偶分布深度拆解
                    st.markdown("### 三、核心特征：奇偶分布深度拆解")
                    st.markdown(f"- **奇数号码（{review['odd']}个）**：{'、'.join([f'{n:02d}' for n in review['numbers'] if n%2==1])}")
                    st.markdown(f"- **偶数号码（{review['even']}个）**：{'、'.join([f'{n:02d}' for n in review['numbers'] if n%2==0])}")
                    diff = abs(review['even'] - review['odd'])
                    if diff == 2:
                        st.markdown(f"✅ **核心特征**：**差值仅为2个**，属于极致弱偏态，无限贴近理论均衡值")
                    else:
                        st.markdown(f"- 奇偶差值：{diff}个")
                    
                    # 四、区间分布
                    st.markdown("### 四、区间分布拆解")
                    col_z1, col_z2, col_z3, col_z4 = st.columns(4)
                    with col_z1:
                        st.metric("01-20区", f"{review['zone1']}个")
                    with col_z2:
                        st.metric("21-40区", f"{review['zone2']}个")
                    with col_z3:
                        st.metric("41-60区", f"{review['zone3']}个")
                    with col_z4:
                        st.metric("61-80区", f"{review['zone4']}个")
                    
                    # 五、号码结构
                    st.markdown("### 五、号码结构深度拆解")
                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        st.markdown(f"- **连号**：{review['consecutive'] if review['consecutive'] else '无'}")
                        st.markdown(f"- **重号（与上期）**：{review['repeat'] if review['repeat'] else '无'}（共{review['repeat_count']}个）")
                    with col_s2:
                        st.markdown(f"- **斜连号（与上期）**：{review['oblique'] if review['oblique'] else '无'}（共{review['oblique_count']}个）")
                        st.markdown(f"- **同尾号**：{review['same_tail'] if review['same_tail'] else '无'}（共{review['same_tail_count']}组）")
                    
                    # 六、总结
                    st.markdown("### 六、本期复盘总结")
                    st.markdown(f"""
                    1.  本期核心特征：**{review['odd_even_ratio']}的奇偶比**，差值仅为{diff}个，整体走势均衡
                    2.  大小比{review['size_ratio']}，{'完全均分' if review['small']==review['large'] else '微偏'}
                    3.  连号数量{review['consecutive_count']}组，{'低于历史均值' if review['consecutive_count']<4 else '高于历史均值'}
                    4.  重号数量{review['repeat_count']}个，{'重号活跃' if review['repeat_count']>3 else '重号正常'}
                    """)
                    
                    # 一键保存到号码库
                    if new_period and new_period not in df['period'].values:
                        if st.button("✅ 一键保存到号码库", type="primary"):
                            save_new_data(new_period, num_msg)
                            st.success(f"成功将{new_period}期数据保存到号码库！")
                            st.rerun()
                    
                    st.caption("以上仅为历史数据复盘，不构成任何购彩建议")

# ----------------------
# 标签页6：跨期对比与预测池
# ----------------------
with tab6:
    st.header("🔄 跨期对比与预测号码池")
    st.info("""
    自动对比上期+本期的复盘数据，同时基于本期号码，生成分层级的下一期预测号码池：
    - 第一层：本期20个开奖号码
    - 第二层：每个本期号码的Top3相随号（经常一起出的）
    - 第三层：每个相随号的Top2跟随号（跨期跟随的）
    
    🔴 红色 = 热号 | 🔵 蓝色 = 冷号 | ⚫ 黑色 = 温号 | 号码后标注：012路, 历史出现次数
    每个级别内，按号码在当前延伸关系中的出现次数分类，次数越高代表被越多上游号码关联，优先级越高
    """)
    
    # 选择要分析的本期期号
    period_list = df['period'].tolist()
    selected_current_period = st.selectbox("选择要分析的【本期】期号（系统会自动获取上期数据）", period_list)
    
    if st.button("生成跨期对比与预测池", type="primary"):
        # 获取本期和上期数据
        current_idx = df[df['period'] == selected_current_period].index[0]
        current_row = df.iloc[current_idx]
        current_nums = current_row[1:].tolist()
        
        prev_nums = None
        prev_period = None
        if current_idx < len(df)-1:
            prev_row = df.iloc[current_idx+1]
            prev_nums = prev_row[1:].tolist()
            prev_period = prev_row['period']
        
        # 调用底层库全量分析
        full_analysis = analyze_full_data(df)
        num_status = get_num_status_dict(full_analysis)
        co_occur_dict = full_analysis['co_occur_matrix']['co_occur_dict']
        follow_dict = full_analysis['follow_matrix']['follow_dict']
        
        # 调用底层库生成分级预测池
        pool_result = generate_leveled_pool(
            current_nums=current_nums,
            co_occur_dict=co_occur_dict,
            follow_dict=follow_dict,
            num_status=num_status
        )
        
        # 1. 生成上期和本期的复盘
        st.divider()
        col_prev, col_curr = st.columns(2)
        
        with col_prev:
            st.subheader(f"📋 上期复盘：{prev_period}期")
            if prev_nums:
                prev_review = generate_deep_review(prev_nums, None, prev_period)
                # 格式化号码
                prev_nums_formatted = " ".join([format_num(n, num_status) for n in prev_nums])
                st.markdown(f"**开奖号码**：{prev_nums_formatted}", unsafe_allow_html=True)
                st.markdown(f"- 奇偶比：{prev_review['odd_even_ratio']}")
                st.markdown(f"- 大小比：{prev_review['size_ratio']}")
                st.markdown(f"- 012路：{prev_review['road_ratio']}")
                st.markdown(f"- 连号：{prev_review['consecutive']}")
                st.markdown(f"- 重号：{prev_review['repeat_count']}个")
            else:
                st.info("没有上期数据")
        
        with col_curr:
            st.subheader(f"📋 本期复盘：{selected_current_period}期")
            curr_review = generate_deep_review(current_nums, prev_nums, selected_current_period)
            # 格式化号码
            curr_nums_formatted = " ".join([format_num(n, num_status) for n in current_nums])
            st.markdown(f"**开奖号码**：{curr_nums_formatted}", unsafe_allow_html=True)
            st.markdown(f"- 奇偶比：{curr_review['odd_even_ratio']}")
            st.markdown(f"- 大小比：{curr_review['size_ratio']}")
            st.markdown(f"- 012路：{curr_review['road_ratio']}")
            st.markdown(f"- 连号：{curr_review['consecutive']}")
            st.markdown(f"- 重号：{curr_review['repeat_count']}个")
        
        # 2. 分层级号码池详情
        st.divider()
        st.subheader("🎯 下一期预测号码池（分层级）")
        co_map = pool_result['co_map']
        follow_map = pool_result['follow_map']
        
        for n in current_nums:
            # 第一层：本期号码
            n_formatted = format_num(n, num_status)
            st.markdown(f"#### 🔹 第一层：本期号码 {n_formatted}", unsafe_allow_html=True)
            # 第二层：它的相随号
            co_list = co_map.get(n, [])
            if co_list:
                co_str = "、".join([format_num(b, num_status) + f"(同现{c}次)" for b,c in co_list])
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;🔸 第二层：相随号 → {co_str}", unsafe_allow_html=True)
                # 第三层：相随号的跟随号
                for b, c_co in co_list:
                    fo_list = follow_map.get(b, [])
                    if fo_list:
                        fo_str = "、".join([format_num(f, num_status) + f"(跟随{c_f}次)" for f,c_f in fo_list])
                        b_formatted = format_num(b, num_status)
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;🔹 第三层：{b_formatted}的跟随号 → {fo_str}", unsafe_allow_html=True)
        
        st.divider()
        # 按出现次数分类的候选汇总（完全符合你的要求的架构）
        st.subheader("📊 预测候选号码汇总（按出现次数分类）")
        
        # 1. 一级候选：本期开奖号码
        st.markdown("#### 🔹 一级候选：本期开奖号码")
        st.markdown("这些是本期开出的核心号码，历史上这类号码的跨期跟随性最强，优先级最高")
        # 一级候选都是1次，按历史出现次数降序排序
        l1_nums = sorted(pool_result['level1'], key=lambda x: num_status[x]['count'], reverse=True)
        l1_formatted = " ".join([format_num(n, num_status) for n in l1_nums])
        st.markdown(f"**出现1次**：{l1_formatted}", unsafe_allow_html=True)
        
        # 2. 二级候选：相随号，按出现次数分类
        level2_groups = pool_result['level2_groups']
        if level2_groups:
            st.markdown("#### 🔸 二级候选：本期号码的Top3相随号")
            st.markdown("这些是和本期号码历史上经常一起开出的号码，同现频率最高，优先级次之，次数越高代表被越多本期号码关联，优先级越高")
            for cnt, nums in level2_groups:
                # 按历史出现次数降序排序
                nums_sorted = sorted(nums, key=lambda x: num_status[x]['count'], reverse=True)
                nums_formatted = " ".join([format_num(n, num_status) for n in nums_sorted])
                st.markdown(f"**出现{cnt}次**：{nums_formatted}", unsafe_allow_html=True)
        
        # 3. 三级候选：跟随号，按出现次数分类
        level3_groups = pool_result['level3_groups']
        if level3_groups:
            st.markdown("#### 🔹 三级候选：相随号的Top2跟随号")
            st.markdown("这些是上期出了相随号后，下期最容易跟随开出的号码，跨期跟随性较强，优先级最低，次数越高代表被越多上游号码关联，优先级越高")
            for cnt, nums in level3_groups:
                # 按历史出现次数降序排序
                nums_sorted = sorted(nums, key=lambda x: num_status[x]['count'], reverse=True)
                nums_formatted = " ".join([format_num(n, num_status) for n in nums_sorted])
                st.markdown(f"**出现{cnt}次**：{nums_formatted}", unsafe_allow_html=True)
        
        # 总统计
        total_nums = len(pool_result['level1']) + len(pool_result['level2_counter']) + len(pool_result['level3_counter'])
        st.info(f"共 {total_nums} 个候选号码，建议优先从高次数、高优先级的候选中搭配选择（仅娱乐参考）")
        
        st.caption("以上仅为历史数据统计，不构成任何购彩建议")

# 底部提醒
st.divider()
st.caption("福彩快乐8数据分析系统 | 底层库V2.0融合版 | 仅作历史数据展示，不构成购彩建议 | 理性购彩，量力而行")
