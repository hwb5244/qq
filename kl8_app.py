# ====================== 核心依赖库（仅保留运行必需项，无额外依赖）======================
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import os
import csv

# ====================== 页面基础配置（必须为首行Streamlit命令，禁止移动）======================
st.set_page_config(
    page_title="快乐8专业数据分析系统",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== 全局常量定义（底层库通用配置）======================
DATA_FILE = "kl8_history_data.csv"
# 1-80质数列表（严格遵循数学定义，1非质数）
PRIME_NUMBERS = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79]
# 区间划分：全覆盖1-80，无重叠无遗漏
ZONE_RULE = {"zone1":[1,20],"zone2":[21,40],"zone3":[41,60],"zone4":[61,80]}
# 冷热温阈值系数
HOT_COLD_FACTOR = 2
# 快乐8玩法对应选号数量
PLAY_RULE = {"选10":10,"选8":8,"选7":7,"选5":5,"选20":20}

# ====================== 88期原始基准数据（100%完整保留，初始化/重置自动还原）======================
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

# ====================== 缓存装饰器（优化性能，避免重复计算）======================
@st.cache_data(ttl=3600)
def load_data_cached():
    return load_data()

@st.cache_data(ttl=3600)
def get_full_analysis_cached(df, window=None):
    return analyze_full_data(df, window)

# ====================== 底层模块1：数据读写/校验/删除 ======================
def load_data():
    """加载历史数据，自动去重，异常兜底，初始化自动还原原始数据"""
    try:
        # 无文件则自动创建，写入完整原始数据
        if not os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['period'] + [f'n{i}' for i in range(1,21)])
                writer.writerows(INIT_DATA)
        # 读取数据并校验
        df = pd.read_csv(DATA_FILE, dtype={'period': str})
        required_cols = ['period'] + [f'n{i}' for i in range(1,21)]
        if not all(col in df.columns for col in required_cols):
            raise ValueError("CSV表头损坏，自动重置")
        # 去重+按期号降序排序
        df = df.drop_duplicates(subset=['period'], keep='last')
        df['period_num'] = df['period'].astype(int)
        df = df.sort_values('period_num', ascending=False).reset_index(drop=True)
        df = df.drop(columns=['period_num'])
        return df
    except Exception as e:
        st.warning(f"数据异常，已自动还原为原始基准数据：{str(e)}")
        # 异常兜底，重置为完整原始数据
        with open(DATA_FILE, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['period'] + [f'n{i}' for i in range(1,21)])
            w.writerows(INIT_DATA)
        df = pd.read_csv(DATA_FILE, dtype={'period': str})
        df['period_num'] = df['period'].astype(int)
        df = df.sort_values('period_num', ascending=False).reset_index(drop=True)
        df = df.drop(columns=['period_num'])
        return df

def save_new_data(period, numbers):
    """保存新期数据到本地文件"""
    try:
        numbers = sorted(numbers)
        with open(DATA_FILE, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([period] + numbers)
        return True
    except Exception as e:
        st.error(f"保存数据失败：{str(e)}")
        return False

def delete_period_data(period, df):
    """删除指定期号数据，同步更新本地文件"""
    new_df = df[df['period'] != period].reset_index(drop=True)
    new_df.to_csv(DATA_FILE, index=False, encoding='utf-8')
    return new_df

def validate_period_unique(period, df):
    """校验期号唯一性和格式合法性"""
    if not period or not period.isdigit():
        return False, "期号必须为非空纯数字"
    if period in df['period'].values:
        return False, "期号已存在，禁止重复录入"
    return True, "校验通过"

def validate_numbers(numbers):
    """校验开奖号码合法性"""
    try:
        ns = [int(x.strip()) for x in numbers if x.strip()]
        if len(ns) != 20:
            return False, f"必须输入20个号码，当前仅输入{len(ns)}个"
        if len(set(ns)) != 20:
            return False, "号码不能重复"
        if min(ns) < 1 or max(ns) > 80:
            return False, "号码必须在1-80区间内"
        return True, sorted(ns)
    except ValueError:
        return False, "号码格式错误，仅支持纯数字"

# ====================== 底层模块2：核心数据分析引擎 ======================
def analyze_full_data(df, window=None):
    """全量数据分析核心引擎，返回所有维度分析结果"""
    data = df.head(window).copy() if window else df.copy()
    num_list = [row.iloc[1:21].tolist() for _, row in data.iterrows()]
    flat_nums = [n for p in num_list for n in p]
    total_periods = len(num_list)
    avg_appear = len(flat_nums) / 80

    # 子模块调用
    hotcold = calc_hot_cold(flat_nums)
    miss = calc_miss_analysis(num_list, total_periods)
    co_mat = calc_co_occur_matrix(num_list)
    follow_mat = calc_follow_matrix(num_list)
    road = calc_road_distribution(flat_nums)
    zone = calc_zone_distribution(flat_nums)
    con_stat = calc_consecutive_stats(num_list)

    return {
        "numbers_list": num_list,
        "flat_nums": flat_nums,
        "total_periods": total_periods,
        "avg_appear": avg_appear,
        "hot_cold": hotcold,
        "miss_analysis": miss,
        "co_occur_matrix": co_mat,
        "follow_matrix": follow_mat,
        "road_distribution": road,
        "zone_distribution": zone,
        "consecutive_stats": con_stat
    }

def calc_hot_cold(flat_nums):
    """冷热号统计，全覆盖1-80号"""
    counter = Counter(flat_nums)
    full_counter = {n: counter.get(n, 0) for n in range(1, 81)}
    hot_top10 = counter.most_common(10)
    cold_top10 = counter.most_common()[-10:][::-1]
    return {
        "hot_top10": hot_top10,
        "cold_top10": cold_top10,
        "full_counter": full_counter
    }

def calc_miss_analysis(num_list, total_periods):
    """遗漏值深度分析，含回补概率，处理除以0异常"""
    last_appear = {}
    miss_current = {}
    miss_avg = {}
    miss_max = {}
    all_miss = defaultdict(list)
    
    for idx, nums in enumerate(num_list):
        for n in nums:
            if n in last_appear:
                all_miss[n].append(idx - last_appear[n])
            last_appear[n] = idx
    
    for n in range(1, 81):
        miss_current[n] = total_periods - 1 - last_appear.get(n, 0)
        arr = all_miss[n]
        miss_avg[n] = np.mean(arr) if arr else 0
        miss_max[n] = max(arr) if arr else 0

    # 生成完整DataFrame
    miss_df = pd.DataFrame({
        "号码": range(1, 81),
        "当前遗漏": [miss_current[n] for n in range(1, 81)],
        "平均遗漏": [f"{miss_avg[n]:.1f}" for n in range(1, 81)],
        "最大遗漏": [miss_max[n] for n in range(1, 81)],
        "出现次数": [len(all_miss[n]) + 1 if n in last_appear else 0 for n in range(1, 81)],
        "回补概率%": [f"{min(100, round((miss_current[n]/miss_avg[n]*100),1)) if miss_avg[n]>0 else 0.0}" for n in range(1, 81)]
    }).sort_values("当前遗漏", ascending=False)
    
    return {
        "miss_df": miss_df,
        "miss_current": miss_current,
        "miss_avg": miss_avg,
        "miss_max": miss_max
    }

def calc_co_occur_matrix(num_list):
    """相随号同现矩阵，无重复统计"""
    co_occur = defaultdict(int)
    for p in num_list:
        sorted_p = sorted(p)
        for i in range(20):
            for j in range(i+1, 20):
                co_occur[(sorted_p[i], sorted_p[j])] += 1
    co_sorted = sorted(co_occur.items(), key=lambda x: x[1], reverse=True)
    return {
        "co_occur_dict": co_occur,
        "co_top10": co_sorted[:10],
        "co_sorted": co_sorted
    }

def calc_follow_matrix(num_list):
    """跨期跟随号矩阵，严格区分上期→下期"""
    follow = defaultdict(int)
    for i in range(1, len(num_list)):
        pre_nums = num_list[i-1]
        curr_nums = num_list[i]
        for a in pre_nums:
            for b in curr_nums:
                follow[(a, b)] += 1
    follow_sorted = sorted(follow.items(), key=lambda x: x[1], reverse=True)
    return {
        "follow_dict": follow,
        "follow_top10": follow_sorted[:10],
        "follow_sorted": follow_sorted
    }

def calc_road_distribution(flat_nums):
    """012路分布统计"""
    road0 = sum(1 for n in flat_nums if n % 3 == 0)
    road1 = sum(1 for n in flat_nums if n % 3 == 1)
    road2 = sum(1 for n in flat_nums if n % 3 == 2)
    total = len(flat_nums) or 1
    return {
        "road0": road0, "road1": road1, "road2": road2,
        "road0_rate": f"{road0/total*100:.1f}%",
        "road1_rate": f"{road1/total*100:.1f}%",
        "road2_rate": f"{road2/total*100:.1f}%"
    }

def calc_zone_distribution(flat_nums):
    """4区间分布统计"""
    z1 = sum(1 for n in flat_nums if 1 <= n <= 20)
    z2 = sum(1 for n in flat_nums if 21 <= n <= 40)
    z3 = sum(1 for n in flat_nums if 41 <= n <= 60)
    z4 = sum(1 for n in flat_nums if 61 <= n <= 80)
    total = len(flat_nums) or 1
    return {
        "zone1": z1, "zone2": z2, "zone3": z3, "zone4": z4,
        "zone1_rate": f"{z1/total*100:.1f}%",
        "zone2_rate": f"{z2/total*100:.1f}%",
        "zone3_rate": f"{z3/total*100:.1f}%",
        "zone4_rate": f"{z4/total*100:.1f}%"
    }

def calc_consecutive_stats(num_list):
    """连号统计"""
    consecutive_list = []
    for p in num_list:
        sorted_p = sorted(p)
        cnt = 0
        for i in range(1, 20):
            if sorted_p[i] == sorted_p[i-1] + 1:
                cnt += 1
        consecutive_list.append(cnt)
    return {
        "avg_consecutive": np.mean(consecutive_list) if consecutive_list else 0,
        "max_consecutive": max(consecutive_list) if consecutive_list else 0,
        "min_consecutive": min(consecutive_list) if consecutive_list else 0,
        "full_list": consecutive_list
    }

# ====================== 底层模块3：号码结构/复盘/预测池 ======================
def calc_number_structure(numbers, prev_numbers=None):
    """单组号码全维度结构特征计算"""
    # 【强制清洗】把Pandas/Numpy读出的int64转为原生Python int，根除页面乱码
    numbers = [int(n) for n in numbers]
    if prev_numbers is not None:
        prev_numbers = [int(n) for n in prev_numbers]
    numbers = sorted(numbers)

    # 奇偶
    odd = sum(n % 2 for n in numbers)
    even = 20 - odd
    # 大小
    small = sum(1 for n in numbers if n <= 40)
    large = 20 - small
    # 012路
    road0 = sum(1 for n in numbers if n % 3 == 0)
    road1 = sum(1 for n in numbers if n % 3 == 1)
    road2 = sum(1 for n in numbers if n % 3 == 2)
    # 质合
    prime = sum(1 for n in numbers if n in PRIME_NUMBERS)
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
    z1 = sum(1 for n in numbers if 1 <= n <= 20)
    z2 = sum(1 for n in numbers if 21 <= n <= 40)
    z3 = sum(1 for n in numbers if 41 <= n <= 60)
    z4 = sum(1 for n in numbers if 61 <= n <= 80)
    
    return {
        "numbers": numbers,
        "odd": odd, "even": even, "odd_even_ratio": f"{odd}:{even}",
        "small": small, "large": large, "size_ratio": f"{small}:{large}",
        "road0": road0, "road1": road1, "road2": road2, "road_ratio": f"{road0}:{road1}:{road2}",
        "prime": prime, "composite": composite, "prime_composite_ratio": f"{prime}:{composite}",
        "sum": sum_val, "span": span,
        "consecutive": consecutive, "consecutive_count": len(consecutive),
        "repeat": repeat, "repeat_count": len(repeat),
        "oblique": oblique, "oblique_count": len(oblique),
        "same_tail": same_tail, "same_tail_count": len(same_tail),
        "z1": z1, "z2": z2, "z3": z3, "z4": z4, "zone_ratio": f"{z1}:{z2}:{z3}:{z4}"
    }

def generate_deep_review(numbers, prev_numbers=None, period="未知期号"):
    """生成单期深度复盘报告"""
    structure = calc_number_structure(numbers, prev_numbers)
    review = {
        "period": period,
        "numbers_str": "、".join([f"{n:02d}" for n in structure['numbers']]),
        **structure
    }
    return review

def generate_leveled_pool(current_nums, co_occur_dict, follow_dict, num_status):
    """生成分级预测号码池，严格去重，按关联次数排序"""
    # 一级候选：本期开奖号码
    level1 = sorted(current_nums)
    level1_set = set(level1)

    # 二级候选：相随号，统计出现次数，排除一级候选
    level2_counter = Counter()
    co_map = defaultdict(list)
    for n in level1:
        temp = []
        for (a, b), cnt in co_occur_dict.items():
            if a == n and b not in level1_set:
                temp.append((b, cnt))
            elif b == n and a not in level1_set:
                temp.append((a, cnt))
        temp.sort(key=lambda x: x[1], reverse=True)
        top3 = temp[:3]
        co_map[n] = top3
        for b, _ in top3:
            level2_counter[b] += 1

    level2_set = set(level2_counter.keys()) - level1_set
    level2_counter = Counter({n: cnt for n, cnt in level2_counter.items() if n in level2_set})

    # 三级候选：跟随号，统计出现次数，排除一、二级候选
    level3_counter = Counter()
    follow_map = defaultdict(list)
    for n in level2_set:
        temp = []
        for (a, b), cnt in follow_dict.items():
            if a == n and b not in level1_set and b not in level2_set:
                temp.append((b, cnt))
        temp.sort(key=lambda x: x[1], reverse=True)
        top2 = temp[:2]
        follow_map[n] = top2
        for b, _ in top2:
            level3_counter[b] += 1

    level3_set = set(level3_counter.keys()) - level1_set - level2_set
    level3_counter = Counter({n: cnt for n, cnt in level3_counter.items() if n in level3_set})

    # 按次数分组
    def group_by_count(counter):
        groups = defaultdict(list)
        for n, cnt in counter.items():
            groups[cnt].append(n)
        sorted_groups = sorted(groups.items(), key=lambda x: x[0], reverse=True)
        return sorted_groups

    return {
        "level1": level1,
        "level1_set": level1_set,
        "level2_counter": level2_counter,
        "level2_set": level2_set,
        "level3_counter": level3_counter,
        "level3_set": level3_set,
        "level2_groups": group_by_count(level2_counter),
        "level3_groups": group_by_count(level3_counter),
        "co_map": co_map,
        "follow_map": follow_map
    }

def generate_multi_play_plan(full_analysis, play_type, plan_count=3):
    """生成多玩法选号方案"""
    need_num = PLAY_RULE[play_type]
    hot_nums = [x[0] for x in full_analysis["hot_cold"]["hot_top10"]]
    cold_nums = [x[0] for x in full_analysis["hot_cold"]["cold_top10"]]
    miss_df = full_analysis["miss_analysis"]["miss_df"]
    miss_ok = miss_df[
        (miss_df["当前遗漏"] >= miss_df["平均遗漏"].astype(float)*0.8) &
        (miss_df["当前遗漏"] <= miss_df["平均遗漏"].astype(float)*1.2)
    ]["号码"].tolist()

    plans = []
    for i in range(plan_count):
        if i == 0:
            nums = list(set(hot_nums[:4] + cold_nums[:2] + miss_ok[:need_num-6]))[:need_num]
        elif i == 1:
            nums = list(set(hot_nums[:6] + cold_nums[:1] + miss_ok[:need_num-7]))[:need_num]
        else:
            nums = list(set(hot_nums[:2] + cold_nums[:5] + miss_ok[:need_num-7]))[:need_num]
        nums.sort()
        plans.append(nums)
    return plans

# ====================== 工具函数：号码格式化渲染 ======================
def get_num_status_dict(full_analysis):
    """生成号码状态字典，用于界面冷热温渲染"""
    counter = full_analysis["hot_cold"]["full_counter"]
    avg_appear = full_analysis["avg_appear"]
    hot_threshold = max(avg_appear + HOT_COLD_FACTOR, 5)
    cold_threshold = min(avg_appear - HOT_COLD_FACTOR, avg_appear * 0.5)

    num_status = {}
    for n in range(1, 81):
        cnt = counter.get(n, 0)
        road = n % 3
        road_name = f"{road}路" if road != 0 else "0路"
        if cnt >= hot_threshold:
            status = "hot"
        elif cnt <= cold_threshold:
            status = "cold"
        else:
            status = "warm"
        num_status[n] = {
            "status": status,
            "road": road_name,
            "count": cnt
        }
    return num_status

def format_num(n, num_status):
    """格式化号码渲染，带颜色、012路、出现次数"""
    s = num_status[n]
    if s["status"] == "hot":
        return f'<span style="color:red; font-weight:bold; margin:0 2px;">{n:02d}</span><small style="color:#666;">({s["road"]},{s["count"]}次)</small>'
    elif s["status"] == "cold":
        return f'<span style="color:blue; font-weight:bold; margin:0 2px;">{n:02d}</span><small style="color:#666;">({s["road"]},{s["count"]}次)</small>'
    else:
        return f'<span style="color:black; margin:0 2px;">{n:02d}</span><small style="color:#666;">({s["road"]},{s["count"]}次)</small>'

# ====================== 全局初始化数据加载 ======================
df = load_data_cached()
total_periods = len(df)

# ====================== 侧边栏配置 ======================
with st.sidebar:
    st.title("🎰 系统设置")
    st.divider()
    st.metric("已收录总期数", f"{total_periods}期")
    st.divider()
    if st.button("🔄 清除缓存刷新数据", type="primary", use_container_width=True):
        load_data_cached.clear()
        get_full_analysis_cached.clear()
        st.rerun()
    st.divider()
    st.error("""
    ⚠️ 风险提示
    彩票开奖为完全随机事件，
    本工具仅用于历史数据统计娱乐，
    不构成任何购彩建议，
    请理性购彩，量力而行！
    """)

# ====================== 主页面7标签页完整渲染 ======================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 首页说明",
    "📋 号码库管理",
    "📊 多周期分析",
    "🔮 多玩法选号参考",
    "📝 单期深度复盘",
    "🔄 跨期对比与预测池",
    "⚙️ 数据管理与重置"
])

# ---------------------- Tab1 首页说明 ----------------------
with tab1:
    st.title("🎰 福彩快乐8 专业数据分析系统")
    st.subheader(f"当前已收录数据：{total_periods}期 | 零依赖复核终版")
    st.divider()
    st.error("""
    ⚠️ 【重要法律与风险提醒】
    本软件仅用于历史开奖数据的统计与展示，**彩票开奖号码为完全随机事件**，
    任何历史走势、分析指标都无法预测未来结果，软件内的"选号参考"仅为娱乐性思路参考，
    不构成任何购彩建议，请理性购彩，量力而行！
    """)
    st.info("""
    软件核心功能：
    1. 永久更新号码库：手动录入/删除/三重数据校验，期号唯一防重复
    2. 多周期深度分析：近10/20/50/100期/全量数据多维度指标拆解，可视化图表直观展示
    3. 全维度指标覆盖：冷热号、遗漏值（含回补概率）、相随号/跟随号矩阵、012路、连号统计、区间分布
    4. 多玩法选号参考：支持选5/选7/选8/选10/选20全玩法，自动生成多组均衡配比方案
    5. 单期深度复盘：历史期号一键复盘/手动输入号码实时生成全维度结构拆解报告
    6. 跨期对比与分级预测池：上期+本期双期联动复盘，自动生成分层级、去重后的预测号码池
    7. 数据管理与重置：支持CSV原始数据备份、一键重置初始数据，保障数据安全不丢失
    """)
    st.divider()
    # 最新一期开奖展示
    if total_periods > 0:
        latest_row = df.iloc[0]
        st.subheader(f"📌 最新一期({latest_row['period']}期)开奖号码")
        st.markdown(f"**{' '.join([f'{n:02d}' for n in latest_row.iloc[1:21].tolist()])}**")

# ---------------------- Tab2 号码库管理 ----------------------
with tab2:
    st.header("📋 开奖号码库管理")
    
    # 新增数据表单
    st.subheader("➕ 录入新一期开奖号码")
    with st.form("add_form", border=True):
        col1, col2 = st.columns(2)
        with col1:
            new_period = st.text_input("期号（如：2026089）", placeholder="例如：2026089")
        with col2:
            nums_input = st.text_input("开奖号码（20个数字，空格分隔）", placeholder="例如：01 02 03 ... 20")
        
        submit = st.form_submit_button("保存到号码库", use_container_width=True, type="primary")
        if submit:
            period_valid, period_msg = validate_period_unique(new_period, df)
            if not period_valid:
                st.error(f"❌ {period_msg}")
            else:
                num_valid, num_msg = validate_numbers(nums_input.strip().split())
                if not num_valid:
                    st.error(f"❌ {num_msg}")
                else:
                    save_success = save_new_data(new_period, num_msg)
                    if save_success:
                        st.success(f"✅ 成功录入{new_period}期数据！软件将自动刷新...")
                        load_data_cached.clear()
                        get_full_analysis_cached.clear()
                        st.rerun()
    
    st.divider()
    # 删除数据功能
    st.subheader("🗑️ 删除错误期号数据")
    with st.form("delete_form", border=True):
        del_period = st.selectbox("选择要删除的期号", df["period"].tolist())
        del_submit = st.form_submit_button("确认删除", use_container_width=True, type="secondary")
        if del_submit:
            df = delete_period_data(del_period, df)
            st.success(f"✅ 成功删除{del_period}期数据！软件将自动刷新...")
            load_data_cached.clear()
            get_full_analysis_cached.clear()
            st.rerun()
    
    st.divider()
    # 历史数据总览
    st.subheader("📜 历史开奖数据总览")
    st.dataframe(df, hide_index=True, use_container_width=True, height=400)

# ---------------------- Tab3 多周期分析 ----------------------
with tab3:
    st.header("📊 多周期数据分析")
    
    # 周期选择
    window_options = {
    "近12期": 12,
    "近24期": 24,
    "近60期": 60,
    "近120期": 120,
    "150期以上全量汇总": None
     }

    selected_window = st.selectbox("选择分析周期", list(window_options.keys()))
    w = window_options[selected_window]
    
    if w and total_periods < w:
        st.warning(f"当前数据只有{total_periods}期，不足{w}期，请先补充更多数据！")
    else:
        full_analysis = get_full_analysis_cached(df, w)
        st.info(f"当前分析：{selected_window}，共{full_analysis['total_periods']}期数据")
        st.divider()
        
        # 1. 冷热号统计
        st.subheader("🔥 冷热号统计")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**热号TOP10（出现次数最多）**")
            hot_df = pd.DataFrame(full_analysis["hot_cold"]["hot_top10"], columns=["号码", "出现次数"])
            st.dataframe(hot_df, hide_index=True, use_container_width=True)
            st.bar_chart(hot_df.set_index("号码"), use_container_width=True)
        with col2:
            st.markdown("**冷号TOP10（出现次数最少）**")
            cold_df = pd.DataFrame(full_analysis["hot_cold"]["cold_top10"], columns=["号码", "出现次数"])
            st.dataframe(cold_df, hide_index=True, use_container_width=True)
            st.bar_chart(cold_df.set_index("号码"), use_container_width=True)
        
        st.divider()
        # 2. 遗漏值统计
        st.subheader("📉 遗漏值统计（按当前遗漏降序，含回补概率）")
        st.dataframe(full_analysis["miss_analysis"]["miss_df"], hide_index=True, use_container_width=True, height=400)
        
        st.divider()
        # 3. 基础分布统计
        st.subheader("📈 基础分布统计")
        col3, col4, col5 = st.columns(3)
        with col3:
            st.markdown("**012路分布**")
            road = full_analysis["road_distribution"]
            road_df = pd.DataFrame({
                "路数": ["0路", "1路", "2路"],
                "出现次数": [road["road0"], road["road1"], road["road2"]],
                "占比": [road["road0_rate"], road["road1_rate"], road["road2_rate"]]
            })
            st.dataframe(road_df, hide_index=True, use_container_width=True)
            st.bar_chart(road_df.set_index("路数"), use_container_width=True)
        
        with col4:
            st.markdown("**连号统计**")
            con = full_analysis["consecutive_stats"]
            con_df = pd.DataFrame({
                "指标": ["平均连号数", "最多连号数", "最少连号数"],
                "数值": [f"{con['avg_consecutive']:.1f}个", f"{con['max_consecutive']}个", f"{con['min_consecutive']}个"]
            })
            st.dataframe(con_df, hide_index=True, use_container_width=True)
        
        with col5:
            st.markdown("**区间分布**")
            zone = full_analysis["zone_distribution"]
            zone_df = pd.DataFrame({
                "区间": ["1-20小号区", "21-40中号区", "41-60大号区", "61-80超大号区"],
                "出现次数": [zone["zone1"], zone["zone2"], zone["zone3"], zone["zone4"]],
                "占比": [zone["zone1_rate"], zone["zone2_rate"], zone["zone3_rate"], zone["zone4_rate"]]
            })
            st.dataframe(zone_df, hide_index=True, use_container_width=True)
            st.bar_chart(zone_df.set_index("区间"), use_container_width=True)
        
        st.divider()
        # 4. 相随号和跟随号
        col6, col7 = st.columns(2)
        with col6:
            st.subheader("👥 相随号TOP10（同现频率最高）")
            co_data = []
            for (a,b), cnt in full_analysis["co_occur_matrix"]["co_top10"]:
                co_data.append({"号码对": f"{a:02d} & {b:02d}", "同现次数": cnt})
            st.dataframe(pd.DataFrame(co_data), hide_index=True, use_container_width=True)
        
        with col7:
            st.subheader("👣 跟随号TOP10（跨期跟随最高）")
            follow_data = []
            for (a,b), cnt in full_analysis["follow_matrix"]["follow_top10"]:
                follow_data.append({"上期A→下期B": f"{a:02d} → {b:02d}", "跟随次数": cnt})
            st.dataframe(pd.DataFrame(follow_data), hide_index=True, use_container_width=True)

# ---------------------- Tab4 多玩法选号参考 ----------------------
with tab4:
    st.header("🔮 多玩法选号参考（娱乐性）")
    st.warning("""
    ⚠️ 注意：以下内容仅为基于历史数据的娱乐性参考思路，**完全无法预测开奖结果**，
    彩票开奖完全随机，请仅作为娱乐参考，切勿当真！
    """)
    
    if total_periods < 10:
        st.info("数据不足，无法生成参考")
    else:
        # 玩法选择
        play_type = st.selectbox("选择玩法", list(PLAY_RULE.keys()))
        plan_count = st.slider("生成方案数量", min_value=1, max_value=5, value=3)
        
        # 调用分析
        full_analysis = get_full_analysis_cached(df, 50)
        last_period = df.iloc[0]
        last_nums = last_period.iloc[1:21].tolist()
        num_status = get_num_status_dict(full_analysis)
        
        # 上期复盘
        st.subheader("📋 上期复盘")
        last_nums_formatted = " ".join([format_num(n, num_status) for n in last_nums])
        st.markdown(f"上期({last_period['period']}期)开奖号码：{last_nums_formatted}", unsafe_allow_html=True)
        
        # 上期指标
        last_structure = calc_number_structure(last_nums)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("上期奇偶", f"{last_structure['odd']}奇{last_structure['even']}偶", "历史平均:10奇10偶")
        with col2:
            st.metric("上期大小", f"{last_structure['small']}小{last_structure['large']}大", "历史平均:10小10大")
        with col3:
            st.metric("上期012路", last_structure["road_ratio"], "历史平均:~7/7/6")
        
        st.divider()
        # 选号思路
        st.subheader("💡 选号参考思路")
        st.markdown("""
        1. **冷热搭配**：热号代表近期活跃，冷号有回归需求，建议选3-4个热号 + 2-3个冷号搭配
        2. **遗漏回归**：优先选择当前遗漏值接近平均遗漏的号码，这类号码大概率要"回补"
        3. **012路均衡**：尽量让选的号码的012路分布接近历史平均
        4. **连号参考**：历史平均每期4-5个连号，建议选1-2组连号
        """)
        
        st.divider()
        # 生成方案
        st.subheader(f"🎯 {play_type}玩法参考方案（共{plan_count}组）")
        plans = generate_multi_play_plan(full_analysis, play_type, plan_count)
        num_status = get_num_status_dict(full_analysis)
        
        for i, plan in enumerate(plans):
            st.markdown(f"#### 方案{i+1}")
            plan_formatted = " ".join([format_num(n, num_status) for n in plan])
            st.markdown(plan_formatted, unsafe_allow_html=True)
            st.caption(f"纯号码：{' '.join([f'{n:02d}' for n in plan])}")
        
        st.caption("再次提醒：这只是基于历史数据的随机参考，完全不代表会开出这些号码！")

# ---------------------- Tab5 单期深度复盘 ----------------------
with tab5:
    st.header("📝 单期深度复盘")
    st.info("支持选择历史期号一键复盘，或手动输入新期号码实时生成复盘报告")
    
    # 选择复盘方式
    review_mode = st.radio("选择复盘方式", ["选择历史期号", "手动输入新期号码"], horizontal=True)
    
    if review_mode == "选择历史期号":
        period_list = df["period"].tolist()
        selected_period = st.selectbox("选择要复盘的期号", period_list)
        
        if st.button("生成复盘报告", type="primary", use_container_width=True):
            # 获取当期和上期号码
            current_row = df[df["period"] == selected_period].iloc[0]
            current_nums = current_row.iloc[1:21].tolist()
            
            current_idx = df[df["period"] == selected_period].index[0]
            prev_nums = None
            if current_idx < len(df)-1:
                prev_row = df.iloc[current_idx+1]
                prev_nums = prev_row.iloc[1:21].tolist()
            
            # 生成复盘
            review = generate_deep_review(current_nums, prev_nums, selected_period)
            full_analysis = get_full_analysis_cached(df)
            num_status = get_num_status_dict(full_analysis)
            
            # 展示报告
            st.divider()
            st.subheader(f"福彩快乐8 {review['period']}期 深度复盘报告")
            
            st.markdown("### 一、官方开奖号码")
            nums_formatted = " ".join([format_num(n, num_status) for n in review["numbers"]])
            st.markdown(nums_formatted, unsafe_allow_html=True)
            
            st.markdown("### 二、核心基础指标汇总")
            diff = abs(review["even"] - review["odd"])
            metrics_df = pd.DataFrame([
                ["奇偶比", review["odd_even_ratio"], "10:10", f"差值{diff}个"],
                ["大小比", review["size_ratio"], "10:10", "完全均衡" if review["small"]==review["large"] else "微偏"],
                ["012路比", review["road_ratio"], "7:7:6", "整体均衡"],
                ["质合比", review["prime_composite_ratio"], "6:14", "合数热开" if review["composite"]>14 else "质数热开"],
                ["和值", review["sum"], "810", "常规区间"],
                ["跨度", review["span"], "60", "覆盖全区间"],
                ["连号组数", review["consecutive_count"], "4.2", "连号退潮" if review["consecutive_count"]<3 else "连号活跃"],
                ["重号数量", review["repeat_count"], "3.5", "重号活跃" if review["repeat_count"]>4 else "正常"]
            ], columns=["指标", "本期结果", "理论均值", "核心定性"])
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
            
            st.markdown("### 三、奇偶分布深度拆解")
            st.markdown(f"- **奇数号码（{review['odd']}个）**：{'、'.join([f'{n:02d}' for n in review['numbers'] if n%2==1])}")
            st.markdown(f"- **偶数号码（{review['even']}个）**：{'、'.join([f'{n:02d}' for n in review['numbers'] if n%2==0])}")
            
            st.markdown("### 四、区间分布拆解")
            col_z1, col_z2, col_z3, col_z4 = st.columns(4)
            with col_z1:
                st.metric("01-20区", f"{review['z1']}个")
            with col_z2:
                st.metric("21-40区", f"{review['z2']}个")
            with col_z3:
                st.metric("41-60区", f"{review['z3']}个")
            with col_z4:
                st.metric("61-80区", f"{review['z4']}个")
            
            st.markdown("### 五、号码结构深度拆解")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                # ========== 新增：预处理所有数据，拼接纯文本，无np标记、不截断 ==========
# 连号格式化拼接
                con_show = "、".join(review['consecutive']) if review['consecutive'] else "无"
# 重号补02格式展示
                repeat_show = "、".join([f"{x:02d}" for x in review['repeat']]) if review['repeat'] else "无"
# 斜连号补02格式，解决末尾字符截断
                oblique_show = "、".join([f"{x:02d}" for x in review['oblique']]) if review['oblique'] else "无"

# 同尾号字典拆解重构，彻底清除np.int64键污染
                tail_format_list = []
                for tail_key, tail_nums in review['same_tail'].items():
                clean_tail = int(tail_key)
                clean_num_str = "、".join([f"{n:02d}" for n in tail_nums])
                tail_format_list.append(f"尾{clean_tail}：{clean_num_str}")
                tail_show = " | ".join(tail_format_list) if tail_format_list else "无"

# ========== 最终页面渲染输出（干净极简，无任何代码残留标记） ==========
                st.markdown(f"- 连号：{con_show}")
                st.markdown(f"- 重号（与上期）：{repeat_show}（共{review['repeat_count']}个）")
                st.markdown(f"- 同尾号：{tail_show}（共{review['same_tail_count']}组）")
                st.markdown(f"- 斜连号（与上期）：{oblique_show}（共{review['oblique_count']}个）")

            with col_s2:
                st.markdown(f"- **质合比**：{review['prime']}质{review['composite']}合")
                st.markdown(f"- **和值**：{review['sum']} | **跨度**：{review['span']}")
            
            st.caption("以上仅为历史数据复盘，不构成任何购彩建议")
    
    else:
        # 手动输入新期号码
        with st.form("manual_review_form", border=True):
            new_period = st.text_input("期号（如：2026089）", placeholder="例如：2026089")
            nums_input = st.text_input("开奖号码（20个数字，空格分隔）", placeholder="例如：08 09 13 14 ... 80")
            submit_review = st.form_submit_button("生成复盘报告", use_container_width=True, type="primary")
            
            if submit_review:
                if not new_period or not new_period.isdigit():
                    st.error("❌ 期号必须为非空数字！")
                else:
                    num_valid, num_msg = validate_numbers(nums_input.strip().split())
                    if not num_valid:
                        st.error(f"❌ {num_msg}")
                    else:
                        prev_nums = df.iloc[0].iloc[1:21].tolist() if total_periods>0 else None
                        review = generate_deep_review(num_msg, prev_nums, new_period)
                        full_analysis = get_full_analysis_cached(df)
                        num_status = get_num_status_dict(full_analysis)
                        
                        st.divider()
                        st.subheader(f"福彩快乐8 {review['period']}期 深度复盘报告")
                        
                        st.markdown("### 一、开奖号码")
                        nums_formatted = " ".join([format_num(n, num_status) for n in review["numbers"]])
                        st.markdown(nums_formatted, unsafe_allow_html=True)
                        
                        st.markdown("### 二、核心基础指标汇总")
                        diff = abs(review["even"] - review["odd"])
                        metrics_df = pd.DataFrame([
                            ["奇偶比", review["odd_even_ratio"], "10:10", f"差值{diff}个"],
                            ["大小比", review["size_ratio"], "10:10", "完全均衡" if review["small"]==review["large"] else "微偏"],
                            ["012路比", review["road_ratio"], "7:7:6", "整体均衡"],
                            ["质合比", review["prime_composite_ratio"], "6:14", "合数热开" if review["composite"]>14 else "质数热开"],
                            ["和值", review["sum"], "810", "常规区间"],
                            ["跨度", review["span"], "60", "覆盖全区间"],
                            ["连号组数", review["consecutive_count"], "4.2", "连号退潮" if review["consecutive_count"]<3 else "连号活跃"],
                            ["重号数量", review["repeat_count"], "3.5", "重号活跃" if review["repeat_count"]>4 else "正常"]
                        ], columns=["指标", "本期结果", "理论均值", "核心定性"])
                        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
                        
                        st.markdown("### 三、奇偶分布深度拆解")
                        st.markdown(f"- **奇数号码（{review['odd']}个）**：{'、'.join([f'{n:02d}' for n in review['numbers'] if n%2==1])}")
                        st.markdown(f"- **偶数号码（{review['even']}个）**：{'、'.join([f'{n:02d}' for n in review['numbers'] if n%2==0])}")
                        
                        st.markdown("### 四、区间分布拆解")
                        col_z1, col_z2, col_z3, col_z4 = st.columns(4)
                        with col_z1:
                            st.metric("01-20区", f"{review['z1']}个")
                        with col_z2:
                            st.metric("21-40区", f"{review['z2']}个")
                        with col_z3:
                            st.metric("41-60区", f"{review['z3']}个")
                        with col_z4:
                            st.metric("61-80区", f"{review['z4']}个")
                        
                        st.markdown("### 五、号码结构深度拆解")
                        col_s1, col_s2 = st.columns(2)
                        with col_s1:
                            st.markdown(f"- **连号**：{review['consecutive'] if review['consecutive'] else '无'}")
                            st.markdown(f"- **重号（与上期）**：{review['repeat'] if review['repeat'] else '无'}（共{review['repeat_count']}个）")
                            st.markdown(f"- **同尾号**：{review['same_tail'] if review['same_tail'] else '无'}（共{review['same_tail_count']}组）")
                        with col_s2:
                            st.markdown(f"- **斜连号（与上期）**：{review['oblique'] if review['oblique'] else '无'}（共{review['oblique_count']}个）")
                            st.markdown(f"- **质合比**：{review['prime']}质{review['composite']}合")
                            st.markdown(f"- **和值**：{review['sum']} | **跨度**：{review['span']}")
                        
                        # 一键保存到号码库
                        if new_period and new_period not in df["period"].values:
                            if st.button("✅ 一键保存到号码库", type="primary", use_container_width=True):
                                save_success = save_new_data(new_period, num_msg)
                                if save_success:
                                    st.success(f"成功将{new_period}期数据保存到号码库！")
                                    load_data_cached.clear()
                                    get_full_analysis_cached.clear()
                                    st.rerun()
                        
                        st.caption("以上仅为历史数据复盘，不构成任何购彩建议")

# ---------------------- Tab6 跨期对比与预测池 ----------------------
with tab6:
    st.header("🔄 跨期对比与预测号码池")
    st.info("""
    自动对比上期+本期的复盘数据，同时基于本期号码，生成分层级的下一期预测号码池：
    - 第一层：本期20个开奖号码
    - 第二层：每个本期号码的Top3相随号（经常一起出的）
    - 第三层：每个相随号的Top2跟随号（跨期跟随的）
    
    🔴 红色 = 热号 | 🔵 蓝色 = 冷号 | ⚫ 黑色 = 温号 | 号码后标注：012路, 历史出现次数
    每个级别内，按号码在当前延伸关系中的出现次数分类，次数越高代表被越多上游号码关联，优先级越高
    【已自动去重】高优先级号码不会重复出现在低优先级中
    """)
    
    period_list = df["period"].tolist()
    selected_current_period = st.selectbox("选择要分析的【本期】期号（系统会自动获取上期数据）", period_list)
    
    if st.button("生成跨期对比与预测池", type="primary", use_container_width=True):
        # 获取本期和上期数据
        current_idx = df[df["period"] == selected_current_period].index[0]
        current_row = df.iloc[current_idx]
        current_nums = current_row.iloc[1:21].tolist()
        
        prev_nums = None
        prev_period = None
        if current_idx < len(df)-1:
            prev_row = df.iloc[current_idx+1]
            prev_nums = prev_row.iloc[1:21].tolist()
            prev_period = prev_row["period"]
        
        # 调用分析
        full_analysis = get_full_analysis_cached(df)
        num_status = get_num_status_dict(full_analysis)
        co_occur_dict = full_analysis["co_occur_matrix"]["co_occur_dict"]
        follow_dict = full_analysis["follow_matrix"]["follow_dict"]
        
        # 生成预测池
        pool_result = generate_leveled_pool(current_nums, co_occur_dict, follow_dict, num_status)
        
        # 双期复盘
        st.divider()
        col_prev, col_curr = st.columns(2)
        
        with col_prev:
            st.subheader(f"📋 上期复盘：{prev_period}期")
            if prev_nums:
                prev_review = generate_deep_review(prev_nums, None, prev_period)
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
            curr_nums_formatted = " ".join([format_num(n, num_status) for n in current_nums])
            st.markdown(f"**开奖号码**：{curr_nums_formatted}", unsafe_allow_html=True)
            st.markdown(f"- 奇偶比：{curr_review['odd_even_ratio']}")
            st.markdown(f"- 大小比：{curr_review['size_ratio']}")
            st.markdown(f"- 012路：{curr_review['road_ratio']}")
            st.markdown(f"- 连号：{curr_review['consecutive']}")
            st.markdown(f"- 重号：{curr_review['repeat_count']}个")
        
        st.divider()
        # 分层级号码池详情
        st.subheader("🎯 下一期预测号码池（分层级）")
        co_map = pool_result["co_map"]
        follow_map = pool_result["follow_map"]
        
        for n in current_nums:
            n_formatted = format_num(n, num_status)
            st.markdown(f"#### 🔹 第一层：本期号码 {n_formatted}", unsafe_allow_html=True)
            co_list = co_map.get(n, [])
            if co_list:
                co_str = "、".join([format_num(b, num_status) + f"(同现{c}次)" for b,c in co_list])
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;🔸 第二层：相随号 → {co_str}", unsafe_allow_html=True)
                for b, c_co in co_list:
                    fo_list = follow_map.get(b, [])
                    if fo_list:
                        fo_str = "、".join([format_num(f, num_status) + f"(跟随{c_f}次)" for f,c_f in fo_list])
                        b_formatted = format_num(b, num_status)
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;🔹 第三层：{b_formatted}的跟随号 → {fo_str}", unsafe_allow_html=True)
        
        st.divider()
        # 按出现次数分类的候选汇总
        st.subheader("📊 预测候选号码汇总（按出现次数分类）")
        
        st.markdown("#### 🔹 一级候选：本期开奖号码")
        st.markdown("这些是本期开出的核心号码，历史上这类号码的跨期跟随性最强，优先级最高")
        l1_nums = sorted(pool_result["level1"], key=lambda x: num_status[x]["count"], reverse=True)
        l1_formatted = " ".join([format_num(n, num_status) for n in l1_nums])
        st.markdown(f"**出现1次**：{l1_formatted}", unsafe_allow_html=True)
        
        level2_groups = pool_result["level2_groups"]
        if level2_groups:
            st.markdown("#### 🔸 二级候选：本期号码的Top3相随号")
            st.markdown("这些是和本期号码历史上经常一起开出的号码，同现频率最高，优先级次之，次数越高代表被越多本期号码关联，优先级越高")
            for cnt, nums in level2_groups:
                nums_sorted = sorted(nums, key=lambda x: num_status[x]["count"], reverse=True)
                nums_formatted = " ".join([format_num(n, num_status) for n in nums_sorted])
                st.markdown(f"**出现{cnt}次**：{nums_formatted}", unsafe_allow_html=True)
        
        level3_groups = pool_result["level3_groups"]
        if level3_groups:
            st.markdown("#### 🔹 三级候选：相随号的Top2跟随号")
            st.markdown("这些是上期出了相随号后，下期最容易跟随开出的号码，跨期跟随性较强，优先级最低，次数越高代表被越多上游号码关联，优先级越高")
            for cnt, nums in level3_groups:
                nums_sorted = sorted(nums, key=lambda x: num_status[x]["count"], reverse=True)
                nums_formatted = " ".join([format_num(n, num_status) for n in nums_sorted])
                st.markdown(f"**出现{cnt}次**：{nums_formatted}", unsafe_allow_html=True)

# ---------------------- Tab7 数据管理与重置【终极修复版，解决括号未闭合报错】----------------------
with tab7:
    st.header("⚙️ 数据管理与重置")
    st.info("支持原始CSV数据备份、一键重置，保障数据安全")

    # CSV备份下载
    st.subheader("📄 原始CSV数据备份")
    st.markdown("下载系统底层使用的CSV原始文件，可用于系统迁移、数据恢复")
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            csv_data = f.read()
        st.download_button(
            label="📥 下载CSV原始备份文件",
            data=csv_data,
            file_name=f"kl8_history_data_backup_{df.iloc[0]['period']}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.warning("数据文件不存在，请先初始化系统")

    st.divider()
    # 数据统计总览
    st.subheader("📈 数据统计总览")
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("总收录期数", f"{total_periods}期")
    with col_stat2:
        st.metric("最早期号", df.iloc[-1]["period"] if total_periods > 0 else "无")
    with col_stat3:
        st.metric("最新期号", df.iloc[0]["period"] if total_periods > 0 else "无")
    with col_stat4:
        st.metric("总号码记录数", f"{total_periods * 20}个")

    # 号码出现次数统计
    st.markdown("#### 号码出现次数统计概览")
    full_analysis = get_full_analysis_cached(df)
    count_df = pd.DataFrame({
        "号码": range(1, 81),
        "总出现次数": [full_analysis["hot_cold"]["full_counter"][n] for n in range(1, 81)]
    }).sort_values("总出现次数", ascending=False)
    st.dataframe(count_df, hide_index=True, use_container_width=True, height=300)

    st.divider()
    # 数据重置功能【修复括号BUG核心段】
    st.subheader("⚠️ 数据重置（危险操作）")
    st.error("此操作会清空所有自定义录入的数据，恢复到系统初始的88期基准数据，不可恢复！")
    with st.form("reset_form", border=True):
        reset_confirm = st.checkbox("我已阅读风险提示，确认要重置所有数据，恢复到初始88期基准数据")
        reset_submit = st.form_submit_button("执行数据重置", type="secondary", use_container_width=True)
        
        if reset_submit:
            if reset_confirm:
                # ✅ 修复点：完整闭合 open() 括号、引号、参数，根治语法报错
                with open(DATA_FILE, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['period'] + [f'n{i}' for i in range(1,21)])
                    writer.writerows(INIT_DATA)
                # 清空缓存+刷新
                load_data_cached.clear()
                get_full_analysis_cached.clear()
                st.success("✅ 数据已重置为原始88期基准数据！页面自动刷新中...")
                st.rerun()
            else:
                st.error("❌ 请先勾选确认框，再执行重置操作！")

# ====================== 全局尾部闭合 + 合规声明【补全所有代码块】======================
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 14px; line-height: 1.8;">
⚠️ 本系统仅用于历史开奖数据统计娱乐，彩票开奖为完全随机事件<br>
不构成任何购彩建议，请理性购彩、量力而行，遵守国家法律法规
</div>
""", unsafe_allow_html=True)
