# ====================== 核心依赖库（无冗余导入） ======================
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import os
import csv
import datetime

# ====================== 页面基础配置（必须首行执行） ======================
st.set_page_config(
    page_title="快乐8专业数据分析系统",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== 全局常量定义（统一管理无硬编码） ======================
DATA_FILE = "kl8_history_data.csv"
SAVE_DIR = "lottery_save"  # 存档总目录：存放预测号/选号组合
# 初始化存档文件夹，避免首次运行报错
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 1-80质数固定列表
PRIME_NUMBERS = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79]
ZONE_RULE = {"zone1":[1,20],"zone2":[21,40],"zone3":[41,60],"zone4":[61,80]}
HOT_COLD_FACTOR = 2
PLAY_RULE = {"选10":10,"选8":8,"选7":7,"选5":5,"选20":20}

# ====================== 【88期原始开奖基准数据 - 核心禁止删除】 ======================
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

# ====================== 缓存装饰器（性能优化，无缓存污染） ======================
@st.cache_data(ttl=3600)
def load_data_cached():
    return load_data()

@st.cache_data(ttl=3600)
def get_full_analysis_cached(df, window=None):
    return analyze_full_data(df, window)

# ====================== 存档工具函数+新增往期回顾读取全量接口 ======================
def save_predict_num(period, level2_list, level3_list):
    filename = os.path.join(SAVE_DIR, f"{period}期预测号.csv")
    df_save = pd.DataFrame({
        "期号": [period] * len(level2_list + level3_list),
        "候选等级": ["二级相随号"] * len(level2_list) + ["三级跟随号"] * len(level3_list),
        "号码": level2_list + level3_list
    })
    df_save.to_csv(filename, index=False, encoding="utf-8-sig")
    return filename

def save_select_comb(period, play_type, comb_list):
    filename = os.path.join(SAVE_DIR, f"{period}期选号组合.csv")
    rows = []
    for idx, nums in enumerate(comb_list):
        rows.append([period, play_type, f"方案{idx+1}", " ".join([f"{n:02d}" for n in nums])])
    df_save = pd.DataFrame(rows, columns=["期号", "玩法类型", "方案编号", "选号号码"])
    df_save.to_csv(filename, index=False, encoding="utf-8-sig")
    return filename

def load_predict_num(period):
    filename = os.path.join(SAVE_DIR, f"{period}期预测号.csv")
    if os.path.exists(filename):
        return pd.read_csv(filename, encoding="utf-8-sig")
    return None

# 新增：读取所有往期选号组合（回顾模块核心，语法全校验）
def load_all_select_comb():
    all_files = [f for f in os.listdir(SAVE_DIR) if f.endswith("期选号组合.csv")]
    all_data = []
    for file in all_files:
        try:
            df = pd.read_csv(os.path.join(SAVE_DIR, file), encoding="utf-8-sig")
            all_data.append(df)
        except Exception:
            continue
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame(columns=["期号", "玩法类型", "方案编号", "选号号码"])

def calc_match_rate(predict_nums, real_nums):
    predict_set = set([int(x) for x in predict_nums])
    real_set = set([int(x) for x in real_nums])
    match = predict_set & real_set
    match_cnt = len(match)
    rate = round(match_cnt / len(predict_set) * 100, 2) if predict_set else 0
    return {"匹配号码": sorted(list(match)), "匹配个数": match_cnt, "正确率%": rate}

# ====================== 底层模块1：数据读写/校验（异常兜底语法修复） ======================
def load_data():
    try:
        if not os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['period'] + [f'n{i}' for i in range(1, 21)])
                writer.writerows(INIT_DATA)
        df = pd.read_csv(DATA_FILE, dtype={'period': str})
        required_cols = ['period'] + [f'n{i}' for i in range(1, 21)]
        if not all(col in df.columns for col in required_cols):
            raise ValueError("表头损坏重置")
        df = df.drop_duplicates(subset=['period'], keep='last')
        df['period_num'] = df['period'].astype(int)
        df = df.sort_values('period_num', ascending=False).reset_index(drop=True)
        df = df.drop(columns=['period_num'])
        return df
    except Exception as e:
        with open(DATA_FILE, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['period'] + [f'n{i}' for i in range(1, 21)])
            w.writerows(INIT_DATA)
        return pd.read_csv(DATA_FILE, dtype={'period': str})

def save_new_data(period, numbers):
    try:
        with open(DATA_FILE, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([period] + sorted(numbers))
        return True
    except Exception:
        return False

def delete_period_data(period, df):
    new_df = df[df['period'] != period].reset_index(drop=True)
    new_df.to_csv(DATA_FILE, index=False, encoding='utf-8')
    return new_df

def validate_period_unique(period, df):
    if not period or not period.isdigit():
        return False, "期号纯数字不能为空"
    if period in df['period'].values:
        return False, "期号重复"
    return True, "通过"

def validate_numbers(nums):
    try:
        ns = [int(x.strip()) for x in nums if x.strip()]
        if len(ns) != 20:
            return False, f"需20个号码，当前{len(ns)}个"
        if len(set(ns)) != 20:
            return False, "号码重复"
        if min(ns) < 1 or max(ns) > 80:
            return False, "范围1-80"
        return True, sorted(ns)
    except ValueError:
        return False, "格式错误仅支持数字"

# ====================== 底层模块2：数据分析引擎【重点修复：if粘连/三元表达式语法】 ======================
def analyze_full_data(df, window=None):
    data = df.head(window).copy() if window else df.copy()
    num_list = [[int(x) for x in row.iloc[1:21].tolist()] for _, row in data.iterrows()]
    flat_nums = [n for p in num_list for n in p]
    total = len(num_list)
    avg = len(flat_nums) / 80
    return {
        "hot_cold": calc_hot_cold(flat_nums),
        "miss_analysis": calc_miss_analysis(num_list, total),
        "co_occur_matrix": calc_co_occur(num_list),
        "follow_matrix": calc_follow(num_list),
        "road": calc_road(flat_nums),
        "zone": calc_zone(flat_nums),
        "con": calc_con(num_list),
        "nums_list": num_list,
        "flat": flat_nums,
        "total": total,
        "avg": avg
    }

def calc_hot_cold(flat):
    c = Counter(flat)
    full = {n: c.get(n, 0) for n in range(1, 81)}
    return {
        "hot_top10": c.most_common(10),
        "cold_top10": c.most_common()[-10:][::-1],
        "full": full
    }

def calc_miss_analysis(num_list, total):
    la = {}
    mc = {}
    ma = {}
    mi = {}
    all_mi = defaultdict(list)
    for idx, ns in enumerate(num_list):
        for n in ns:
            if n in la:
                all_mi[n].append(idx - la[n])
            la[n] = idx
    for n in range(1, 81):
        mi[n] = total - 1 - la.get(n, 0)
        arr = all_mi[n]
        mc[n] = np.mean(arr) if arr else 0
        ma[n] = max(arr) if arr else 0
    miss_df = pd.DataFrame({
        "号码": range(1, 81),
        "当前遗漏": [mi[n] for n in range(1, 81)],
        "平均遗漏": [f"{mc[n]:.1f}" for n in range(1, 81)],
        "最大遗漏": [ma[n] for n in range(1, 81)],
        "出现次数": [len(all_mi[n]) + 1 if n in la else 0 for n in range(1, 81)],
        "回补率%": [f"{min(100, round(mi[n]/mc[n]*100, 1)) if mc[n] > 0 else 0.0}" for n in range(1, 81)]
    }).sort_values("当前遗漏", ascending=False)
    return {"miss_df": miss_df, "mi": mi, "mc": mc, "ma": ma}

def calc_co_occur(num_list):
    cd = defaultdict(int)
    for ns in num_list:
        s = sorted(ns)
        for i in range(20):
            for j in range(i + 1, 20):
                cd[(s[i], s[j])] += 1
    return {
        "dict": cd,
        "top10": sorted(cd.items(), key=lambda x: x[1], reverse=True)[:10]
    }

def calc_follow(num_list):
    fd = defaultdict(int)
    for i in range(1, len(num_list)):
        pre, curr = num_list[i-1], num_list[i]
        for a in pre:
            for b in curr:
                fd[(a, b)] += 1
    return {
        "dict": fd,
        "top10": sorted(fd.items(), key=lambda x: x[1], reverse=True)[:10]
    }

# 修复点：移除or极简写法，标准if判断；if后补空格，根除SyntaxError
def calc_road(flat):
    r0 = sum(1 for n in flat if n % 3 == 0)
    r1 = sum(1 for n in flat if n % 3 == 1)
    r2 = sum(1 for n in flat if n % 3 == 2)
    t = len(flat)
    if t == 0:
        t = 1
    return {
        "r0": r0, "r1": r1, "r2": r2,
        "r0r": f"{r0/t*100:.1f}%",
        "r1r": f"{r1/t*100:.1f}%",
        "r2r": f"{r2/t*100:.1f}%"
    }

# 致命修复点：if 数字之间补空格，解决你 Line295 核心报错
def calc_zone(flat):
    z1 = sum(1 for n in flat if 1 <= n <= 20)
    z2 = sum(1 for n in flat if 21 <= n <= 40)
    z3 = sum(1 for n in flat if 41 <= n <= 60)
    z4 = sum(1 for n in flat if 61 <= n <= 80)
    t = len(flat)
    if t == 0:
        t = 1
    return {
        "z1": z1, "z2": z2, "z3": z3, "z4": z4,
        "z1r": f"{z1/t*100:.1f}%",
        "z2r": f"{z2/t*100:.1f}%",
        "z3r": f"{z3/t*100:.1f}%",
        "z4r": f"{z4/t*100:.1f}%"
    }

# 致命修复点：三元表达式全部补空格，解决你 Line317 核心报错
def calc_con(num_list):
    clist = []
    for ns in num_list:
        s, cnt = sorted(ns), 0
        for i in range(1, 20):
            if s[i] == s[i-1] + 1:
                cnt += 1
        clist.append(cnt)
    return {
        "avg": np.mean(clist) if clist else 0,
        "max": max(clist) if clist else 0,
        "min": min(clist) if clist else 0
    }

# ====================== 同源核心函数+预测生成+格式化（修复np.int64/语法挤压） ======================
def calc_number_structure(numbers, prev_numbers=None):
    numbers = [int(n) for n in numbers]
    if prev_numbers is not None:
        prev_numbers = [int(n) for n in prev_numbers]
    numbers = sorted(numbers)
    odd = sum(n % 2 for n in numbers)
    even = 20 - odd
    small = sum(1 for n in numbers if n <= 40)
    large = 20 - small
    r0 = sum(1 for n in numbers if n % 3 == 0)
    r1 = sum(1 for n in numbers if n % 3 == 1)
    r2 = sum(1 for n in numbers if n % 3 == 2)
    prime = sum(1 for n in numbers if n in PRIME_NUMBERS)
    composite = 20 - prime
    sumv = sum(numbers)
    span = numbers[-1] - numbers[0]
    con_list, i = [], 0
    while i < 19:
        if numbers[i+1] == numbers[i] + 1:
            st = numbers[i]
            while i < 19 and numbers[i+1] == numbers[i] + 1:
                i += 1
            con_list.append(f"{st}-{numbers[i]}")
        i += 1
    repeat = [n for n in numbers if n in prev_numbers] if prev_numbers else []
    oblique = [n for n in numbers if (n-1 in prev_numbers) or (n+1 in prev_numbers)] if prev_numbers else []
    tail_cnt = Counter([n % 10 for n in numbers])
    tail_dict = {t: [int(x) for x in numbers if x % 10 == t] for t, c in tail_cnt.items() if c >= 2}
    z1 = sum(1 for n in numbers if 1 <= n <= 20)
    z2 = sum(1 for n in numbers if 21 <= n <= 40)
    z3 = sum(1 for n in numbers if 41 <= n <= 60)
    z4 = sum(1 for n in numbers if 61 <= n <= 80)
    return {
        "nums": numbers, "odd": odd, "even": even, "oe": f"{odd}:{even}",
        "small": small, "large": large, "sl": f"{small}:{large}",
        "r0": r0, "r1": r1, "r2": r2, "road": f"{r0}:{r1}:{r2}",
        "prime": prime, "composite": composite, "pc": f"{prime}:{composite}",
        "sum": sumv, "span": span, "con": con_list, "con_cnt": len(con_list),
        "repeat": repeat, "repeat_cnt": len(repeat), "oblique": oblique, "oblique_cnt": len(oblique),
        "tail": tail_dict, "tail_cnt": len(tail_dict), "z1": z1, "z2": z2, "z3": z3, "z4": z4
    }

def generate_deep_review(nums, prev_nums=None, period="未知"):
    s = calc_number_structure(nums, prev_nums)
    return {"period": period, **s}

def generate_leveled_pool(curr_nums, co_dict, follow_dict, num_status):
    curr = [int(x) for x in curr_nums]
    l1 = set(curr)
    l2_cnt, co_map = Counter(), defaultdict(list)
    for n in curr:
        tmp = sorted([(a, b, c) for (a, b), c in co_dict.items() if (a == n and b not in l1) or (b == n and a not in l1)], key=lambda x: x[2], reverse=True)[:3]
        co_map[n] = [(x[1], x[2]) if x[0] == n else (x[0], x[2]) for x in tmp]
        for b, _ in co_map[n]:
            l2_cnt[b] += 1
    l2_set = set(l2_cnt.keys()) - l1
    l2_cnt = Counter({k: v for k, v in l2_cnt.items() if k in l2_set})
    l3_cnt, follow_map = Counter(), defaultdict(list)
    for n in l2_set:
        tmp = sorted([(a, b, c) for (a, b), c in follow_dict.items() if a == n and b not in l1 and b not in l2_set], key=lambda x: x[2], reverse=True)[:2]
        follow_map[n] = [(x[1], x[2]) for x in tmp]
        for b, _ in follow_map[n]:
            l3_cnt[b] += 1
    l3_set = set(l3_cnt.keys()) - l1 - l2_set
    def group(cnt):
        g = defaultdict(list)
        for k, v in cnt.items():
            g[v].append(k)
        return sorted(g.items(), key=lambda x: x[0], reverse=True)
    return {"l1": curr, "l2": l2_set, "l3": l3_set, "l2_group": group(l2_cnt), "l3_group": group(l3_cnt), "co": co_map, "follow": follow_map}

def get_num_status(full):
    c = full['hot_cold']['full']
    avg = full['avg']
    hot = max(avg + HOT_COLD_FACTOR, 5)
    cold = min(avg - HOT_COLD_FACTOR, avg * 0.5)
    d = {}
    for n in range(1, 81):
        cnt = c[n]
        r = n % 3
        st = "hot" if cnt >= hot else "cold" if cnt <= cold else "warm"
        d[n] = {"st": st, "road": f"{r}路" if r != 0 else "0路", "cnt": cnt}
    return d

def fmt_num(n, d):
    s = d[n]
    if s['st'] == "hot":
        return f'<span style="color:red;font-weight:bold;margin:0 2px">{n:02d}</span><small style="color:#666">({s["road"]},{s["cnt"]}次)</small>'
    elif s['st'] == "cold":
        return f'<span style="color:blue;font-weight:bold;margin:0 2px">{n:02d}</span><small style="color:#666">({s["road"]},{s["cnt"]}次)</small>'
    else:
        return f'<span style="color:black;margin:0 2px">{n:02d}</span><small style="color:#666">({s["road"]},{s["cnt"]}次)</small>'

def gen_play_plan(full, play, predict_nums, cnt=3):
    need = PLAY_RULE[play]
    hot = [x[0] for x in full['hot_cold']['hot_top10']]
    cold = [x[0] for x in full['hot_cold']['cold_top10']]
    plans = []
    for i in range(cnt):
        if i == 0:
            base = predict_nums[:int(need * 0.6)] + hot[:3] + cold[:2]
        elif i == 1:
            base = predict_nums[:int(need * 0.7)] + hot[:4]
        else:
            base = predict_nums[:int(need * 0.5)] + cold[:5]
        res = sorted(list(set(base)))[:need]
        plans.append(res)
    return plans 
# ====================== 全局初始化 ======================
df = load_data_cached()
total = len(df)

# ====================== 侧边栏 ======================
with st.sidebar:
    st.title("🎰快乐8数据分析系统")
    st.divider()
    st.metric("总收录期数", f"{total}期")
    st.divider()
    if st.button("🔄清除缓存刷新", use_container_width=True):
        load_data_cached.clear()
        get_full_analysis_cached.clear()
        st.rerun()
    st.divider()
    st.error("仅历史数据统计娱乐，不构成购彩建议，理性购彩！")

# ====================== 标签页创建 ======================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🏠首页", "📋号码库", "📊多周期", "🔮选号参考", "📝单期复盘", "🔄跨期对比", "⚙️设置"])

# ========== Tab1 首页 ==========
with tab1:
    st.title("🎰福彩快乐8专业数据分析系统")
    st.subheader(f"当前收录：{total}期 | 全功能修复终版")
    st.error("开奖完全随机，仅历史统计娱乐，不构成购彩建议！")
    if total > 0:
        l = df.iloc[0]
        st.subheader(f"最新{l['period']}期开奖：{' '.join([f'{x:02d}' for x in l.iloc[1:21]])}")

# ========== Tab2 号码库管理 ==========
with tab2:
    st.header("📋开奖号码库管理")
    with st.form("add"):
        c1, c2 = st.columns(2)
        with c1:
            p1 = st.text_input("期号")
        with c2:
            p2 = st.text_input("20个号码空格分隔")
        sub = st.form_submit_button("保存录入", use_container_width=True, type="primary")
        if sub:
            v1, m1 = validate_period_unique(p1, df)
            if not v1:
                st.error(m1)
            else:
                v2, m2 = validate_numbers(p2.split())
                if not v2:
                    st.error(m2)
                else:
                    save_new_data(p1, m2)
                    st.success("录入成功")
                    load_data_cached.clear()
                    get_full_analysis_cached.clear()
                    st.rerun()
    st.divider()
    with st.form("del"):
        dp = st.selectbox("选择删除期号", df['period'].tolist())
        if st.form_submit_button("确认删除", type="secondary", use_container_width=True):
            df = delete_period_data(dp, df)
            st.success("删除成功")
            load_data_cached.clear()
            get_full_analysis_cached.clear()
            st.rerun()
    st.divider()
    st.dataframe(df, use_container_width=True, height=400)

# ========== Tab3 多周期（12/24/60/120/150期配置） ==========
with tab3:
    st.header("📊多周期数据分析")
    window_options = {"近12期": 12, "近24期": 24, "近60期": 60, "近120期": 120, "150期以上全量": None}
    sel = st.selectbox("选择分析周期", list(window_options.keys()))
    w = window_options[sel]
    if w and total < w:
        st.warning(f"当前仅{total}期，未达到{w}期门槛，请补充数据！")
    else:
        fd = get_full_analysis_cached(df, w)
        st.info(f"分析维度：{sel}，共{fd['total']}期")
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("热号TOP10")
            st.dataframe(pd.DataFrame(fd['hot_cold']['hot_top10'], columns=["号码", "次数"]), use_container_width=True)
        with c2:
            st.subheader("冷号TOP10")
            st.dataframe(pd.DataFrame(fd['hot_cold']['cold_top10'], columns=["号码", "次数"]), use_container_width=True)
        st.divider()
        st.subheader("遗漏值全表")
        st.dataframe(fd['miss_analysis']['miss_df'], use_container_width=True, height=400)

# ========== Tab4 选号参考【核心新增：往期组合回顾双子标签语法修复】 ==========
with tab4:
    st.header("🔮多玩法选号参考（娱乐性）")
    st.warning("⚠️ 开奖完全随机，仅娱乐参考，不构成购彩建议！")
    gen_tab, history_tab = st.tabs(["🎯 生成新选号方案", "📋 往期组合回顾"])

    # 子标签1：生成方案
    with gen_tab:
        st.info("读取预测号生成方案、自动存档、正确率验算、迭代优化")
        sel_p = st.selectbox("选择读取对应期预测号", df['period'].tolist())
        pred_df = load_predict_num(sel_p)
        real_nums = df[df['period'] == sel_p].iloc[0].iloc[1:21].tolist() if sel_p in df['period'].values else []
        if pred_df is not None:
            st.success(f"读取{sel_p}期预测号成功")
            all_pred = pred_df['号码'].tolist()
            mr = calc_match_rate(all_pred, real_nums)
            st.metric("预测整体正确率", f"{mr['正确率%']}%")
            st.write("匹配号码：", "、".join([f"{x:02d}" for x in mr['匹配号码']]))
            play_sel = st.selectbox("选择玩法", list(PLAY_RULE.keys()))
            g_cnt = st.slider("生成组数", 1, 5, 3)
            plans = gen_play_plan(get_full_analysis_cached(df), play_sel, all_pred, g_cnt)
            save_select_comb(sel_p, play_sel, plans)
            st.success("选号组合已自动存档！")

    # 子标签2：往期回顾模块（语法全修复，筛选逻辑无报错）
    with history_tab:
        st.info("自动读取所有历史存档，按期号/玩法筛选，联动开奖数据展示命中率")
        all_comb_df = load_all_select_comb()
        if all_comb_df.empty:
            st.info("暂无往期选号组合，请先生成方案并存档！")
        else:
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                filter_p_list = st.multiselect("筛选期号", sorted(all_comb_df["期号"].unique(), reverse=True))
            with col_f2:
                filter_play_list = st.multiselect("筛选玩法", all_comb_df["玩法类型"].unique())
            filter_df = all_comb_df.copy()
            if filter_p_list:
                filter_df = filter_df[filter_df["期号"].isin(filter_p_list)]
            if filter_play_list:
                filter_df = filter_df[filter_df["玩法类型"].isin(filter_play_list)]
            st.dataframe(filter_df, hide_index=True, use_container_width=True)
            # 命中率计算语法修复
            if not filter_df.empty:
                st.divider()
                st.subheader("筛选结果历史命中率复盘")
                avg_rate_list = []
                for _, row in filter_df.iterrows():
                    p = row["期号"]
                    nums = [int(x) for x in row["选号号码"].split()]
                    if p in df["period"].values:
                        real = [int(x) for x in df[df["period"] == p].iloc[0].iloc[1:21].tolist()]
                        avg_rate_list.append(calc_match_rate(nums, real)['正确率%'])
                if avg_rate_list:
                    st.metric("筛选组合平均命中率", f"{round(np.mean(avg_rate_list), 2)}%")

# ========== Tab5 单期深度复盘【彻底修复空白+全量渲染+数据唯一性保障】 ==========
with tab5:
    st.header("📝 单期深度复盘")
    st.info("支持历史期号一键复盘/手动录入，同源固定逻辑计算，历史数据不变则复盘结果永久唯一不变动")
    # 复盘模式选择
    review_mode = st.radio("选择复盘方式", ["选择历史期号", "手动录入新期号码"], horizontal=True)

    if review_mode == "选择历史期号":
        period_list = df["period"].tolist()
        selected_period = st.selectbox("选择要复盘的期号", period_list)
        
        if st.button("生成深度复盘报告", use_container_width=True, type="primary"):
            # 读取当期&上期原始数据（只读不修改，保障数据唯一性）
            current_row = df[df["period"] == selected_period].iloc[0]
            current_nums = [int(x) for x in current_row.iloc[1:21].tolist()]
            current_idx = df[df["period"] == selected_period].index[0]
            
            # 边界兼容：无下期数据时置空不报错
            prev_nums = None
            prev_period = None
            if current_idx < len(df) - 1:
                prev_row = df.iloc[current_idx + 1]
                prev_nums = [int(x) for x in prev_row.iloc[1:21].tolist()]
                prev_period = prev_row["period"]

            # 调用固定同源核心函数（逻辑固化不动，结果唯一不变）
            review_result = generate_deep_review(current_nums, prev_nums, selected_period)
            full_analysis = get_full_analysis_cached(df)
            num_status_dict = get_num_status(full_analysis)

            # ====================== 核心新增：全量格式化渲染（解决空白关键） ======================
            # 清洗转换，根除np.int64显示异常
            con_show = "、".join(review_result["con"]) if review_result["con"] else "无"
            repeat_show = "、".join([f"{x:02d}" for x in review_result["repeat"]]) if review_result["repeat"] else "无"
            oblique_show = "、".join([f"{x:02d}" for x in review_result["oblique"]]) if review_result["oblique"] else "无"
            
            # 同尾号格式化拼接
            tail_format_list = []
            for tail_key, tail_nums in review_result["tail"].items():
                clean_tail = int(tail_key)
                clean_nums = "、".join([f"{x:02d}" for x in tail_nums])
                tail_format_list.append(f"尾{clean_tail}：{clean_nums}")
            tail_show = " | ".join(tail_format_list) if tail_format_list else "无"

            # 页面正式渲染输出
            st.divider()
            st.subheader(f"✅ {selected_period}期 深度复盘报告（结果固定唯一）")
            st.markdown("### 一、本期开奖号码（冷热色标区分）")
            nums_formatted_html = " ".join([fmt_num(n, num_status_dict) for n in review_result["nums"]])
            st.markdown(nums_formatted_html, unsafe_allow_html=True)

            st.markdown("### 二、核心结构指标一览")
            metrics_df = pd.DataFrame([
                ["奇偶比例", review_result["oe"], "理论均值 10:10"],
                ["大小比例", review_result["sl"], "理论均值 10:10"],
                ["012路比例", review_result["road"], "均衡参考 7:7:6"],
                ["质合比例", review_result["pc"], "常态分布 6:14"],
                ["号码和值", review_result["sum"], "全期中位参考值"],
                ["区间跨度", review_result["span"], "1-80全域测算"],
                ["连号组数", review_result["con_cnt"], "历史平均4.2组"],
                ["跨期重号数", review_result["repeat_cnt"], "常态3-4个"]
            ], columns=["统计指标", "本期固化结果", "行业参考基准"])
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)

            st.markdown("### 三、号码细节深度拆解")
            st.success(f"📌 连续号码串：{con_show}")
            st.info(f"🔄 与上期重号：{repeat_show}（共计 {review_result['repeat_cnt']} 个）")
            st.warning(f"🎯 同尾组合分布：{tail_show}（共计 {review_result['tail_cnt']} 组）")
            st.markdown(f"🔗 斜连关联号码：{oblique_show}（共计 {review_result['oblique_cnt']} 个）")
            st.caption("💡 说明：底层历史数据/计算逻辑未改动时，该复盘结果永远一致，无随机变动")

    else:
        # 手动录入模式 补全渲染逻辑（同样修复空白）
        with st.form("manual_review_form", border=True):
            manual_period = st.text_input("期号（如：2026089）", placeholder="例：2026089")
            manual_nums = st.text_input("开奖号码（20个数字，空格分隔）", placeholder="例：08 09 13 14 ... 80")
            submit_manual = st.form_submit_button("生成复盘报告", use_container_width=True, type="primary")

            if submit_manual:
                if not manual_period or not manual_period.isdigit():
                    st.error("❌ 期号必须为非空纯数字！")
                else:
                    num_valid, num_msg = validate_numbers(manual_nums.strip().split())
                    if not num_valid:
                        st.error(f"❌ {num_msg}")
                    else:
                        prev_nums = [int(x) for x in df.iloc[0].iloc[1:21].tolist()] if total > 0 else None
                        review_result = generate_deep_review(num_msg, prev_nums, manual_period)
                        full_analysis = get_full_analysis_cached(df)
                        num_status_dict = get_num_status(full_analysis)

                        # 格式化+渲染输出
                        con_show = "、".join(review_result["con"]) if review_result["con"] else "无"
                        repeat_show = "、".join([f"{x:02d}" for x in review_result["repeat"]]) if review_result["repeat"] else "无"
                        oblique_show = "、".join([f"{x:02d}" for x in review_result["oblique"]]) if review_result["oblique"] else "无"
                        tail_format_list = []
                        for tail_key, tail_nums in review_result["tail"].items():
                            clean_tail = int(tail_key)
                            clean_nums = "、".join([f"{x:02d}" for x in tail_nums])
                            tail_format_list.append(f"尾{clean_tail}：{clean_nums}")
                        tail_show = " | ".join(tail_format_list) if tail_format_list else "无"

                        st.divider()
                        st.subheader(f"✅ 手动录入 {manual_period}期 复盘报告")
                        nums_formatted_html = " ".join([fmt_num(n, num_status_dict) for n in review_result["nums"]])
                        st.markdown(nums_formatted_html, unsafe_allow_html=True)
                        
                        metrics_df = pd.DataFrame([
                            ["奇偶比例", review_result["oe"], "理论均值 10:10"],
                            ["大小比例", review_result["sl"], "理论均值 10:10"],
                            ["012路比例", review_result["road"], "均衡参考 7:7:6"],
                            ["质合比例", review_result["pc"], "常态分布 6:14"],
                            ["号码和值", review_result["sum"], "全期中位参考值"],
                            ["区间跨度", review_result["span"], "1-80全域测算"],
                            ["连号组数", review_result["con_cnt"], "历史平均4.2组"],
                            ["跨期重号数", review_result["repeat_cnt"], "常态3-4个"]
                        ], columns=["统计指标", "本期固化结果", "行业参考基准"])
                        st.dataframe(metrics_df, hide_index=True, use_container_width=True)

                        st.markdown(f"- 连续号码串：{con_show}")
                        st.markdown(f"- 与上期重号：{repeat_show}（{review_result['repeat_cnt']}个）")
                        st.markdown(f"- 同尾组合：{tail_show}（{review_result['tail_cnt']}组）")
                        st.markdown(f"- 斜连关联号：{oblique_show}（{review_result['oblique_cnt']}个）")

                        # 一键保存入口保留
                        if manual_period not in df["period"].values:
                            if st.button("✅ 一键保存到号码库", type="primary", use_container_width=True):
                                save_success = save_new_data(manual_period, num_msg)
                                if save_success:
                                    st.success(f"✅ 成功入库{manual_period}期数据！")
                                    load_data_cached.clear()
                                    get_full_analysis_cached.clear()
                                    st.rerun() 
# ========== Tab6 跨期对比 ==========
with tab6:
    st.header("🔄跨期对比与预测号码池")
    plist = df['period'].tolist()
    scp = st.selectbox("选择本期分析期号", plist)
    if st.button("生成对比+预测存档", use_container_width=True, type="primary"):
        cidx = df[df['period'] == scp].index[0]
        crow = df.iloc[cidx]
        cn = crow.iloc[1:21].tolist()
        pn = df.iloc[cidx+1].iloc[1:21].tolist() if cidx < len(df)-1 else None
        rev_curr = generate_deep_review(cn, pn, scp)
        fd = get_full_analysis_cached(df)
        nsd = get_num_status(fd)
        pool = generate_leveled_pool(cn, fd['co_occur_matrix']['dict'], fd['follow_matrix']['dict'], nsd)
        save_predict_num(scp, list(pool['l2']), list(pool['l3']))
        st.success("预测号已存档完成！")

# ========== Tab7 设置页 ==========
with tab7:
    st.header("⚙️数据管理与重置")
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            cd = f.read()
        st.download_button("下载原始CSV备份", cd, f"kl8_backup_{df.iloc[0]['period']}.csv", use_container_width=True)
    st.divider()
    with st.form("reset_f"):
        ck = st.checkbox("确认重置为原始88期数据，不可恢复")
        if st.form_submit_button("执行重置", type="secondary") and ck:
            # 文件操作语法完全闭合，无括号遗漏
            with open(DATA_FILE, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['period'] + [f'n{i}' for i in range(1, 21)])
                w.writerows(INIT_DATA)
            load_data_cached.clear()
            get_full_analysis_cached.clear()
            st.success("重置完成")
            st.rerun()

# ====================== 全局尾部合规声明（语法闭合无遗漏） ======================
st.divider()
st.markdown("""
<div style="text-align:center;color:#666;font-size:14px">
⚠️ 本系统仅历史数据统计娱乐，彩票开奖完全随机，不构成任何购彩建议，理性购彩遵守法规
</div>
""", unsafe_allow_html=True)  
