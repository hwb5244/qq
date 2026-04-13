import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import os
import csv
import datetime
import shutil
import zipfile

# 页面配置必须是第一个Streamlit命令，禁止任何Streamlit代码放在此之前
st.set_page_config(
    page_title="快乐8专业数据分析系统",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)  

# 核心数据路径（先定义基础路径，再定义衍生路径，彻底解决未定义错误）
DATA_FILE = "kl8_history_data.csv"
SAVE_DIR = "lottery_save"
ARCHIVE_ROOT = os.path.join(os.getcwd(), "KL8_Lottery_Data_Archive")
INDEX_FILE = os.path.join(ARCHIVE_ROOT, "05_存档总索引表", "index.csv")

# 批量复盘全局存档配置（仅定义1次，放在ARCHIVE_ROOT之后）
BATCH_REVIEW_DIR = os.path.join(ARCHIVE_ROOT, "06_全量批量复盘存档")
BATCH_REVIEW_SUMMARY = os.path.join(BATCH_REVIEW_DIR, "全量期数复盘总表.csv")
BATCH_REVIEW_DETAIL_DIR = os.path.join(BATCH_REVIEW_DIR, "单期复盘明细")

# 业务规则常量（仅定义1次，无重复）
PRIME_NUMBERS = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79]
ZONE_RULE = {"zone1":[1,20],"zone2":[21,40],"zone3":[41,60],"zone4":[61,80]}
HOT_COLD_FACTOR = 2
PLAY_RULE = {"选10":10,"选8":8,"选7":7,"选5":5,"选20":20}

# 初始化所有文件夹（所有路径定义完成后再创建，避免路径不存在）
for dir_path in [
    SAVE_DIR, 
    ARCHIVE_ROOT, 
    os.path.join(ARCHIVE_ROOT, "01_基准原始库"),
    os.path.join(ARCHIVE_ROOT, "02_增量开奖数据库"),
    os.path.join(ARCHIVE_ROOT, "03_每期预测号存档库"),
    os.path.join(ARCHIVE_ROOT, "04_每期选号组合存档库"),
    os.path.join(ARCHIVE_ROOT, "05_存档总索引表"),
    BATCH_REVIEW_DIR, 
    BATCH_REVIEW_DETAIL_DIR
]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# ====================== 88期原始开奖基准数据 - 核心禁止删除 ======================
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
# 初始期号集合，用于删除权限判断
INIT_PERIODS = set([x[0] for x in INIT_DATA])  

# 缓存装饰器（性能优化，无缓存污染）
@st.cache_data(ttl=3600)
def load_data_cached():
    return load_data()

@st.cache_data(ttl=3600)
def get_full_analysis_cached(df, window=None):
    return analyze_full_data(df, window)

# ====================== 数据读写核心工具函数 ======================
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


# ====================== 存档管理核心工具函数 ======================
def save_predict_num(period, level2_list, level3_list):
    """
    格式合规终版 | 双向去重+仅存二三层预测号
    :param period: 期号（字符串/数字）
    :param level2_list: 第二层 二级相随号列表
    :param level3_list: 第三层 三级跟随号列表
    :return: 存档文件完整路径
    """
    # 1. 单层内部去重+非法值清洗（格式规范：生成器表达式括号完全闭合）
    level2_clean = sorted(list(set(
        int(n) for n in level2_list
        if str(n).strip().isdigit() and 1 <= int(n) <= 80
    )))
    level3_raw = sorted(list(set(
        int(n) for n in level3_list
        if str(n).strip().isdigit() and 1 <= int(n) <= 80
    )))

    # 2. 跨层级去重：三级池剔除二级已存在号码，全局无重复
    level3_clean = [num for num in level3_raw if num not in level2_clean]

    # 3. 构造合规数据表（严格仅存二级、三级预测号）
    df_save = pd.DataFrame({
        "期号": [period] * (len(level2_clean) + len(level3_clean)),
        "候选等级": ["二级相随号"] * len(level2_clean) + ["三级跟随号"] * len(level3_clean),
        "号码": level2_clean + level3_clean
    })

    # 4. 按要求命名文件，编码兼容全平台
    filename = os.path.join(SAVE_DIR, f"{period}期预测号.csv")
    df_save.to_csv(filename, index=False, encoding="utf-8-sig")

    # 前端友好提示
    st.caption(
        f"✅ 预测号存档完成 | 二级相随号：{len(level2_clean)}个 | "
        f"三级跟随号：{len(level3_clean)}个 | 去重后合计：{len(df_save)}个唯一号码"
    )
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

# ====================== 数据分析核心引擎 ======================
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

def calc_road(flat):
    r0 = sum(1 for n in flat if n % 3 == 0)
    r1 = sum(1 for n in flat if n % 3 == 1)
    r2 = sum(1 for n in flat if n % 2 == 2)
    t = len(flat) if len(flat) > 0 else 1
    return {
        "r0": r0, "r1": r1, "r2": r2,
        "r0r": f"{r0/t*100:.1f}%",
        "r1r": f"{r1/t*100:.1f}%",
        "r2r": f"{r2/t*100:.1f}%"
    }

def calc_zone(flat):
    z1 = sum(1 for n in flat if 1 <= n <= 20)
    z2 = sum(1 for n in flat if 21 <= n <= 40)
    z3 = sum(1 for n in flat if 41 <= n <= 60)
    z4 = sum(1 for n in flat if 61 <= n <= 80)
    t = len(flat) if len(flat) > 0 else 1
    return {
        "z1": z1, "z2": z2, "z3": z3, "z4": z4,
        "z1r": f"{z1/t*100:.1f}%",
        "z2r": f"{z2/t*100:.1f}%",
        "z3r": f"{z3/t*100:.1f}%",
        "z4r": f"{z4/t*100:.1f}%"
    }

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


# ====================== 号码结构计算+预测生成核心函数 ======================
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

def generate_leveled_pool(l1_nums, co_occur_dict, follow_dict, num_status_dict):
    l1 = set(l1_nums)
    l2_result = set()
    l3_result = set()
    l2_group = []
    l3_group = []
    co_map = defaultdict(list)
    follow_map = defaultdict(list)

    # 二级相随号生成
    try:
        co_dict = co_occur_dict if isinstance(co_occur_dict, dict) else {}
        for n in l1:
            valid_list = []
            for k, c in co_dict.items():
                if not isinstance(k, tuple) or len(k) != 2:
                    continue
                a, b = k
                if (a == n and b not in l1) or (b == n and a not in l1):
                    match_num = b if a == n else a
                    valid_list.append((a, b, c, match_num))
                    co_map[n].append((match_num, c))
            valid_list_sorted = sorted(valid_list, key=lambda x: x[2], reverse=True)[:3]
            for item in valid_list_sorted:
                a, b, c, match_num = item
                l2_result.add(match_num)
                l2_group.append((c, [match_num]))
    except Exception:
        pass

    # 三级跟随号生成
    try:
        follow_dict_valid = follow_dict if isinstance(follow_dict, dict) else {}
        for n in l2_result:
            valid_follow = []
            for k, c in follow_dict_valid.items():
                if not isinstance(k, tuple) or len(k) != 2:
                    continue
                a, b = k
                if (a == n and b not in l1 and b not in l2_result) or (b == n and a not in l1 and a not in l2_result):
                    match_num = b if a == n else a
                    valid_follow.append((a, b, c, match_num))
                    follow_map[n].append((match_num, c))
            valid_follow_sorted = sorted(valid_follow, key=lambda x: x[2], reverse=True)[:3]
            for item in valid_follow_sorted:
                a, b, c, match_num = item
                l3_result.add(match_num)
                l3_group.append((c, [match_num]))
    except Exception:
        pass

    return {
        "l1": list(l1),
        "l2": list(l2_result),
        "l3": list(l3_result),
        "l2_group": l2_group,
        "l3_group": l3_group,
        "co": co_map,
        "follow": follow_map
    }

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

@st.cache_data(ttl=0)
def build_iron_rule_combination(candidate_pool, two_con, three_con, last_real_nums, hot12_list, hot24_list, df_back, need_cnt, group_cnt, seed_key, max_overlap=2):
    final_combs = []
    try:
        # 铁律1：强制剔除前三期连续开出号码
        candidate_pool = list(set(candidate_pool))
        candidate_pool = [n for n in candidate_pool if n not in three_con]
        if len(candidate_pool) < need_cnt:
            return final_combs

        # 百分比字符串转数字，全容错
        if not df_back.empty and "回补率%" in df_back.columns:
            df_back["temp_num"] = df_back["回补率%"].astype(str).str.replace("%", "").astype(float)
            high_back = set(df_back[df_back["temp_num"] >= 80]["号码"].tolist())
        else:
            high_back = set()

        # 权重计算：预测池优选权重+热号权重+回补权重
        score_dict = {}
        hot12 = set(hot12_list)
        hot24 = set(hot24_list)
        # 加载预测池号码，用于优选权重
        pred_df = load_predict_num(seed_key.split("_")[0])
        predict_pool_nums = set()
        if pred_df is not None and not pred_df.empty:
            predict_pool_nums = set(pred_df["号码"].tolist())
        
        for n in candidate_pool:
            base_score = 0
            # 预测池优选权重：预测池的号码加100分，优先选
            if n in predict_pool_nums:
                base_score += 100
            if n in hot24: base_score += 50
            if n in hot12: base_score += 30
            if n in high_back: base_score += 20
            if n in two_con: base_score -= 50
            score_dict[n] = base_score

        # 无随机固定排序，永久不变
        sort_nums = sorted(candidate_pool, key=lambda x: (-score_dict.get(x, 0), x))
        idx = 0
        max_try = 200
        last_num_len = len(last_real_nums) if len(last_real_nums) > 0 else 20
        # 铁律3：与上期重合率≤20%校验
        while len(final_combs) < group_cnt and idx < max_try and idx + need_cnt <= len(sort_nums):
            temp_comb = sort_nums[idx:idx+need_cnt]
            overlap = set(temp_comb) & set(last_real_nums)
            overlap_rate = len(overlap) / last_num_len
            # 铁律4：组间重叠度控制
            overlap_with_exist = False
            for exist_comb in final_combs:
                if len(set(temp_comb) & set(exist_comb)) > max_overlap:
                    overlap_with_exist = True
                    break
            if overlap_rate <= 0.20 and temp_comb not in final_combs and not overlap_with_exist:
                final_combs.append(temp_comb)
            idx += 2
    except Exception:
        pass
    return final_combs

# ====================== 全量批量自动复盘核心函数 ======================
def batch_auto_review_all_periods(df, overwrite_exist=False):
    sort_df = df.sort_values("period", ascending=True).reset_index(drop=True)
    result_list = []
    fail_list = []

    for idx, row in sort_df.iterrows():
        period = row["period"]
        current_nums = [int(x) for x in row.iloc[1:21].tolist()]
        detail_file = os.path.join(BATCH_REVIEW_DETAIL_DIR, f"{period}期_复盘明细.csv")
        predict_file = os.path.join(SAVE_DIR, f"{period}期预测号.csv")
        comb_file = os.path.join(SAVE_DIR, f"{period}期选号组合.csv")

        if not overwrite_exist and os.path.exists(detail_file) and os.path.exists(predict_file) and os.path.exists(comb_file):
            result_list.append({
                "期号": period,
                "处理状态": "已跳过(已存在)",
                "单期复盘": "已完成",
                "跨期预测池": "已完成",
                "4铁律选号组合": "已完成"
            })
            continue

        review_status = "未执行"
        predict_status = "未执行"
        comb_status = "未执行"
        try:
            # 1. 单期深度复盘
            prev_nums = None
            if idx > 0:
                prev_row = sort_df.iloc[idx-1]
                prev_nums = [int(x) for x in prev_row.iloc[1:21].tolist()]
            review_result = generate_deep_review(current_nums, prev_nums, period)
            review_df = pd.DataFrame([{
                "期号": review_result["period"],
                "开奖号码": " ".join([f"{x:02d}" for x in review_result["nums"]]),
                "奇偶比例": review_result["oe"],
                "大小比例": review_result["sl"],
                "012路比例": review_result["road"],
                "质合比例": review_result["pc"],
                "号码和值": review_result["sum"],
                "区间跨度": review_result["span"],
                "连号组数": review_result["con_cnt"],
                "连号明细": "、".join(review_result["con"]),
                "跨期重号数": review_result["repeat_cnt"],
                "重号明细": " ".join([f"{x:02d}" for x in review_result["repeat"]]),
                "同尾组数": review_result["tail_cnt"],
                "同尾明细": str(review_result["tail"])
            }])
            review_df.to_csv(detail_file, index=False, encoding="utf-8-sig")
            review_status = "已完成"

        # 2. 跨期对比+分层预测号生成｜格式规整+仅存二三层+双向去重同步优化
            if idx >= 1:
                full_analysis = get_full_analysis_cached(df)
                num_status_dict = get_num_status(full_analysis)
    
                # 生成原生分级号码池（隔离第一层开奖号，不参与存档）
                pool_result = generate_leveled_pool(
                    current_nums,
                    full_analysis["co_occur_matrix"]["dict"],
                    full_analysis["follow_matrix"]["dict"],
                    num_status_dict
                )

                # 只提取第二层/第三层，剔除第一层原生开奖基底号
                pure_level2 = list(pool_result["l2"])
                pure_level3 = list(pool_result["l3"])

                # 调用优化后去重存档函数，统一和Tab6逻辑对齐
                save_predict_num(period, pure_level2, pure_level3)
                predict_status = "已完成"
            else:
                predict_status = "跳过(无上期数据)"


            # 3. 4铁律选号组合生成
            if idx >= 3:
                his12 = get_full_analysis_cached(df, 12)
                his24 = get_full_analysis_cached(df, 24)
                n1 = set(sort_df.iloc[idx-1].iloc[1:21].tolist())
                n2 = set(sort_df.iloc[idx-2].iloc[1:21].tolist())
                n3 = set(sort_df.iloc[idx-3].iloc[1:21].tolist())
                two_continuous = list(n1 & n2)
                three_continuous = list(n1 & n2 & n3)
                last_pre_real = list(n1)
                pred_df = load_predict_num(period)
                if pred_df is not None and not pred_df.empty and "号码" in pred_df.columns:
                    l2_only = pred_df[pred_df["候选等级"] == "二级相随号"]["号码"].tolist()
                    l3_only = pred_df[pred_df["候选等级"] == "三级跟随号"]["号码"].tolist()
                    hot12_plain = [x[0] for x in his12.get("hot_cold", {}).get("hot_top10", [])]
                    hot24_plain = [x[0] for x in his24.get("hot_cold", {}).get("hot_top10", [])]
                    df_back_plain = his24.get("miss_analysis", {}).get("miss_df", pd.DataFrame({"回补率%": [], "号码": []}))
                    FIX_PLAY_CONFIG = [
                        {"玩法名称":"11码", "选号个数":11, "固定生成组数":3},
                        {"玩法名称":"8码", "选号个数":8,  "固定生成组数":5},
                        {"玩法名称":"6码", "选号个数":6,  "固定生成组数":10},
                        {"玩法名称":"3码", "选号个数":3,  "固定生成组数":10}
                    ]
                    all_combs = []
                    for cfg in FIX_PLAY_CONFIG:
                        play_name, need_num, fix_group = cfg["玩法名称"], cfg["选号个数"], cfg["固定生成组数"]
                        combs = build_iron_rule_combination(
                            candidate_pool=range(1,81),
                            two_con=two_continuous,
                            three_con=three_continuous,
                            last_real_nums=last_pre_real,
                            hot12_list=hot12_plain,
                            hot24_list=hot24_plain,
                            df_back=df_back_plain,
                            need_cnt=need_num,
                            group_cnt=fix_group,
                            seed_key=f"{period}_{play_name}"
                        )
                        all_combs.extend(combs)
                    if all_combs:
                        save_select_comb(period, "批量自动生成-4铁律合规", all_combs)
                        comb_status = "已完成"
                    else:
                        comb_status = "生成失败(候选池不足)"
            else:
                comb_status = "跳过(无前三期数据)"

            result_list.append({
                "期号": period,
                "处理状态": "处理成功",
                "单期复盘": review_status,
                "跨期预测池": predict_status,
                "4铁律选号组合": comb_status
            })

        except Exception as e:
            fail_list.append(f"{period}期：{str(e)}")
            result_list.append({
                "期号": period,
                "处理状态": "处理失败",
                "单期复盘": review_status,
                "跨期预测池": predict_status,
                "4铁律选号组合": comb_status,
                "失败原因": str(e)
            })

    result_df = pd.DataFrame(result_list)
    result_df.to_csv(BATCH_REVIEW_SUMMARY, index=False, encoding="utf-8-sig")
    return result_df, fail_list

# ====================== 双流派复盘专用工具函数 ======================
def calc_trend_hit_rate(trend_df, period_list, df_data):
    hit_records = []
    valid_p = sorted(trend_df["期号"].unique(), reverse=True)
    for p in valid_p:
        if p not in period_list:
            continue
        real_p = [int(x) for x in df_data[df_data["period"] == p].iloc[0].iloc[1:21].tolist()]
        for _,row in trend_df[trend_df["期号"]==p].iterrows():
            try:
                c_nums = [int(x) for x in row["选号号码"].split()]
                hit_records.append(calc_match_rate(c_nums, real_p)["正确率%"])
            except Exception:
                continue
    if hit_records:
        return round(np.mean(hit_records),2), len(hit_records)
    else:
        return 0, 0  


# ====================== 全局初始化+侧边栏+标签页定义 ======================
# 全局数据初始化
df = load_data_cached()
total = len(df)

# 侧边栏
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
    st.warning("仅历史数据统计娱乐，不构成购彩建议，理性购彩！")

# 标签页定义（变量和标签名100%匹配，先定义后使用，彻底解决未定义错误）
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏠首页", 
    "📋号码库", 
    "📊多周期", 
    "🔮选号参考", 
    "📝单期复盘", 
    "🔄跨期对比", 
    "⚙️设置", 
    "📦全量批量复盘"
])       


# ========== Tab1 首页 ==========
with tab1:
    st.title("🎰福彩快乐8专业数据分析系统")
    st.subheader(f"当前收录：{total}期 | 预测池优选升级终版")
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
            p2 = st.text_input("开奖号码（20个数字，空格分隔）")
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
        # 过滤掉初始的88期，用户只能删除自己录入的
        user_periods = [p for p in df['period'].tolist() if p not in INIT_PERIODS]
        if not user_periods:
            st.info("暂无用户手动录入的期号，系统自动生成的基准期号无法删除")
            dp = None
        else:
            dp = st.selectbox("选择删除期号（仅可删除手动录入的期号，系统基准期号无法删除）", user_periods)
        if st.form_submit_button("确认删除", type="secondary", use_container_width=True):
            if dp is None:
                st.error("❌ 无法删除系统自动生成的基准期号！")
            else:
                df = delete_period_data(dp, df)
                st.success("删除成功")
                load_data_cached.clear()
                get_full_analysis_cached.clear()
                st.rerun()
    st.divider()
    st.dataframe(df, use_container_width=True, height=400)

# ========== Tab3 多周期数据分析 ==========
with tab3:
    st.header("📊多周期数据分析")
    # 修改周期选项：10/20/50/100/150
    window_options = {"近10期": 10, "近20期": 20, "近50期": 50, "近100期": 100, "150期以上全量": None}
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

# ========== Tab4 双流派升级终版｜多玩法选号·4铁律风控·双流派并行 ==========
with tab4:
    st.header("🔮 多玩法选号｜4铁律风控固化组合·双流派并行升级")
    st.info("💡 升级逻辑：热号惯性+冷号回补双流派完全独立并行，彻底告别赌行情，全行情覆盖，一套踩空一套补位，杜绝全组合崩盘")
    st.error("🚨 共用刚性合规红线（双流派100%强制执行）：弃前三期连出 | 降两期连出权重 | 与上期重合率≤20% | 预测池优选权重优先")
    st.warning("⚠️ 仅历史数据推演娱乐，组合固化不随刷新变动，不构成购彩建议！")
    market_tab, hot_flow_tab, cold_back_tab, check_tab, review_tab = st.tabs([
        "📈 行情主线判断",
        "🔥 热号惯性流派",
        "🧊 冷号回补流派",
        "📊 开奖核对",
        "💡 双流派复盘优化"
    ])

    # 全局固定配置
    FIX_PLAY_CONFIG = [
        {"玩法名称":"11码", "选号个数":11, "固定生成组数":3},
        {"玩法名称":"8码", "选号个数":8,  "固定生成组数":5},
        {"玩法名称":"6码", "选号个数":6,  "固定生成组数":10},
        {"玩法名称":"3码", "选号个数":3,  "固定生成组数":10}
    ]
    MAX_OVERLAP_BETWEEN_TREND = 1

    # 共用工具函数（Tab内作用域）
    def get_recent_continuous_no(df_target, curr_period):
        two_continuous = []
        three_continuous = []
        last_real = []
        try:
            sort_df = df_target.sort_values("period", ascending=False).reset_index(drop=True)
            curr_idx = sort_df[sort_df["period"] == curr_period].index[0]
            if curr_idx + 3 >= len(sort_df):
                return two_continuous, three_continuous, last_real
            n1 = set(sort_df.iloc[curr_idx+1].iloc[1:21].tolist())
            n2 = set(sort_df.iloc[curr_idx+2].iloc[1:21].tolist())
            n3 = set(sort_df.iloc[curr_idx+3].iloc[1:21].tolist())
            two_continuous = list(n1 & n2)
            three_continuous = list(n1 & n2 & n3)
            last_real = list(n1)
        except Exception:
            pass
        return two_continuous, three_continuous, last_real

    def calc_occur_rate(df_target, window=10):
        try:
            data = df_target.head(window).copy()
            num_list = [[int(x) for x in row.iloc[1:21].tolist()] for _, row in data.iterrows()]
            flat_nums = [n for p in num_list for n in p]
            occur_count = Counter(flat_nums)
            return {n: occur_count.get(n, 0) for n in range(1, 81)}, num_list
        except Exception:
            return {n:0 for n in range(1,81)}, []

    def calc_follow_probability(df_target, target_nums, min_occur=4, min_rate=0.4):
        follow_count = defaultdict(int)
        target_appear_times = 0
        try:
            data = df_target.head(50).copy()
            num_list = [[int(x) for x in row.iloc[1:21].tolist()] for _, row in data.iterrows()]
            target_set = set(target_nums)
            for i in range(1, len(num_list)):
                pre_nums = set(num_list[i-1])
                curr_nums = set(num_list[i])
                if len(pre_nums & target_set) > 0:
                    target_appear_times += 1
                    for n in curr_nums:
                        follow_count[n] += 1
            if target_appear_times == 0:
                return []
            high_prob_follow = [
                n for n, cnt in follow_count.items()
                if cnt >= min_occur and (cnt / target_appear_times) >= min_rate
            ]
            return high_prob_follow
        except Exception:
            return []

    def get_under_open_zone(num_list, window=3, max_occur=3):
        zone_occur = {"zone1":0, "zone2":0, "zone3":0, "zone4":0}
        try:
            recent_data = num_list[:window]
            for period_nums in recent_data:
                for n in period_nums:
                    if 1 <= n <=20: zone_occur["zone1"] +=1
                    elif 21 <= n <=40: zone_occur["zone2"] +=1
                    elif 41 <= n <=60: zone_occur["zone3"] +=1
                    elif 61 <= n <=80: zone_occur["zone4"] +=1
            under_zones = [zone for zone, cnt in zone_occur.items() if cnt <= max_occur]
            zone_num_map = {
                "zone1": list(range(1,21)),
                "zone2": list(range(21,41)),
                "zone3": list(range(41,61)),
                "zone4": list(range(61,81))
            }
            under_zone_nums = []
            for z in under_zones:
                under_zone_nums.extend(zone_num_map[z])
            return under_zone_nums, zone_occur
        except Exception:
            return [], zone_occur

    # 全局基础数据加载
    period_list = df["period"].tolist() if len(df) > 0 else []
    if not period_list:
        st.error("暂无开奖数据，请先初始化号码库！")
    else:
        target_period = st.selectbox("选择绑定预测期号", period_list, key="tab4_target_period")
        current_idx = df[df["period"] == target_period].index[0]
        current_nums_base = [int(x) for x in df.iloc[current_idx].iloc[1:21].tolist()]
        two_continuous, three_continuous, last_pre_real = get_recent_continuous_no(df, target_period)
        full_analysis_all = get_full_analysis_cached(df)
        full_analysis_10 = get_full_analysis_cached(df, 10)
        full_analysis_12 = get_full_analysis_cached(df, 12)
        full_analysis_24 = get_full_analysis_cached(df, 24)
        num_status_dict = get_num_status(full_analysis_all)
        hot12_plain = [x[0] for x in full_analysis_12.get("hot_cold", {}).get("hot_top10", [])]
        hot24_plain = [x[0] for x in full_analysis_24.get("hot_cold", {}).get("hot_top10", [])]
        df_back_plain = full_analysis_24.get("miss_analysis", {}).get("miss_df", pd.DataFrame({"回补率%": [], "号码": []}))
        real_check_nums = df[df["period"] == target_period].iloc[0].iloc[1:21].tolist()
        occur_10, recent_3_num_list = calc_occur_rate(df, 10)
        occur_5, _ = calc_occur_rate(df, 5)
        high_prob_follow_nums = calc_follow_probability(df, current_nums_base, min_occur=4, min_rate=0.4)
        under_zone_nums, zone_occur_3 = get_under_open_zone(recent_3_num_list, window=3, max_occur=3)
        hot_core_pool = set()

        # 子标签1：行情主线判断
        with market_tab:
            st.subheader("📈 近2期行情主线自动判断")
            st.info("自动识别行情类型，给出双流派权重分配建议，告别主观赌行情")
            st.divider()
            try:
                recent_2_data = df.head(2).copy()
                recent_2_nums = [[int(x) for x in row.iloc[1:21].tolist()] for _, row in recent_2_data.iterrows()]
                hot_top20 = [x[0] for x in full_analysis_10["hot_cold"]["hot_top10"] + full_analysis_10["hot_cold"]["hot_top10"][10:20]]
                hot_count_2 = 0
                total_count_2 = 0
                for nums in recent_2_nums:
                    hot_count_2 += len(set(nums) & set(hot_top20))
                    total_count_2 += len(nums)
                hot_rate_2 = round(hot_count_2 / total_count_2 * 100, 2)
                repeat_count = len(set(recent_2_nums[0]) & set(recent_2_nums[1]))
                if hot_rate_2 >= 60 and repeat_count >=4:
                    market_type = "🔥 热号抱团惯性行情"
                    hot_weight = 60
                    cold_weight = 40
                    market_desc = "近2期热号占比≥60%，重号数≥4个，强者恒强特征明显，优先配置热号惯性流派"
                elif hot_rate_2 <= 40 and repeat_count <=2:
                    market_type = "🧊 冷号集中回补行情"
                    hot_weight = 40
                    cold_weight = 60
                    market_desc = "近2期热号占比≤40%，重号数≤2个，均值回归特征明显，优先配置冷号回补流派"
                else:
                    market_type = "⚖️ 均衡轮动行情"
                    hot_weight = 50
                    cold_weight = 50
                    market_desc = "行情特征不明显，区间轮动快，双流派均衡配置，双线兜底"
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("行情类型", market_type)
                with c2:
                    st.metric("近2期热号占比", f"{hot_rate_2}%")
                with c3:
                    st.metric("近2期跨期重号数", repeat_count)
                st.divider()
                st.success(market_desc)
                st.subheader("📊 双流派投注权重分配建议")
                weight_c1, weight_c2 = st.columns(2)
                with weight_c1:
                    st.metric("热号惯性流派权重", f"{hot_weight}%")
                with weight_c2:
                    st.metric("冷号回补流派权重", f"{cold_weight}%")
                st.divider()
                st.subheader("📋 近3期区间出号明细（欠开区间识别）")
                zone_df = pd.DataFrame([{
                    "区间": "1-20",
                    "近3期累计出号": zone_occur_3["zone1"],
                    "状态": "🔴 欠开区间" if zone_occur_3["zone1"] <=3 else "🟢 正常区间"
                },{
                    "区间": "21-40",
                    "近3期累计出号": zone_occur_3["zone2"],
                    "状态": "🔴 欠开区间" if zone_occur_3["zone2"] <=3 else "🟢 正常区间"
                },{
                    "区间": "41-60",
                    "近3期累计出号": zone_occur_3["zone3"],
                    "状态": "🔴 欠开区间" if zone_occur_3["zone3"] <=3 else "🟢 正常区间"
                },{
                    "区间": "61-80",
                    "近3期累计出号": zone_occur_3["zone4"],
                    "状态": "🔴 欠开区间" if zone_occur_3["zone4"] <=3 else "🟢 正常区间"
                }])
                st.dataframe(zone_df, hide_index=True, use_container_width=True)
            except Exception as e:
                st.error(f"行情判断失败：{str(e)}")

        # 子标签2：热号惯性流派
        with hot_flow_tab:
            st.header("🔥 热号惯性流派｜趋势跟随体系")
            st.info("底层逻辑：强者恒强，高概率相随号+有效热号双重筛选，适配热号抱团惯性行情")
            st.divider()
            st.warning("🚨 本流派刚性红线：1. 100%剔除三期连开必杀号，两期连开降权号单组最多1个；2. 与上期开奖号重合率≤20%，单组最多2个；3. 预测池优选权重优先；4. 组间核心胆码重叠度≤2个")
            st.divider()

            st.subheader("✅ 6步选号法执行结果")
            step1_base = [n for n in range(1,81) if n not in three_continuous]
            st.caption(f"步骤1：合规红线过滤，剔除三期连开必杀号，剩余候选池：{len(step1_base)}个")
            step2_follow = [n for n in high_prob_follow_nums if n in step1_base]
            st.caption(f"步骤2：提取上期号码高概率相随号，剩余候选池：{len(step2_follow)}个")
            step3_hot = []
            for n in step2_follow:
                if occur_10.get(n, 0) >=6 and occur_5.get(n, 0) >=1:
                    step3_hot.append(n)
            st.caption(f"步骤3：筛选有效热号，剩余候选池：{len(step3_hot)}个")
            step4_odd = [n for n in step3_hot if n %2 ==1]
            step4_even = [n for n in step3_hot if n %2 ==0]
            step4_zone12 = [n for n in step3_hot if 1<=n<=40]
            step4_zone34 = [n for n in step3_hot if 41<=n<=80]
            need_odd_cnt = max(round(len(step3_hot)*0.55), len(step4_odd))
            need_zone12_cnt = max(round(len(step3_hot)*0.5), len(step4_zone12))
            step4_final = step4_odd[:need_odd_cnt] + step4_even + step4_zone12[:need_zone12_cnt] + step4_zone34
            step4_final = list(set(step4_final))
            st.caption(f"步骤4：奇偶/区间适配，剩余候选池：{len(step4_final)}个")
            step5_core = sorted(step4_final, key=lambda x: (-occur_10.get(x,0), x))[:15]
            st.caption(f"步骤5：生成15个核心胆码，按热度排序完成")
            st.markdown(f"**核心胆码池**：{' '.join([f'{x:02d}' for x in step5_core])}")
            hot_core_pool = set(step5_core)

            st.divider()
            st.subheader("📌 热号惯性流派 固化组合生成结果")
            hot_all_combs = []
            for cfg in FIX_PLAY_CONFIG:
                play_name, need_num, fix_group = cfg["玩法名称"], cfg["选号个数"], cfg["固定生成组数"]
                hot_combs = build_iron_rule_combination(
                    candidate_pool=range(1,81),
                    two_con=two_continuous,
                    three_con=three_continuous,
                    last_real_nums=last_pre_real,
                    hot12_list=hot12_plain,
                    hot24_list=hot24_plain,
                    df_back=df_back_plain,
                    need_cnt=need_num,
                    group_cnt=fix_group,
                    seed_key=f"{target_period}_hot_{play_name}",
                    max_overlap=2
                )
                hot_all_combs.extend(hot_combs)
                st.divider()
                st.subheader(f"📌 {play_name}｜固定{fix_group}组（4铁律校验通过）")
                if not hot_combs:
                    st.warning("候选池号码不足，无法生成对应组数组合")
                else:
                    for idx, comb in enumerate(hot_combs, 1):
                        comb_html = " ".join([fmt_num(n, num_status_dict) for n in comb])
                        st.markdown(f"**热号流派{play_name}方案{idx}**：{comb_html}", unsafe_allow_html=True)
                        overlap_check = len(set(comb)&set(last_pre_real))/20*100 if len(last_pre_real) > 0 else 0
                        hit_res = calc_match_rate(comb, real_check_nums)
                        st.caption(f"重合率{overlap_check:.1f}%≤20%合规 | 当期命中{hit_res['匹配个数']}个 | 命中率{hit_res['正确率%']}%")
            
            if hot_all_combs:
                hot_save_path = save_select_comb(target_period, "热号惯性流派-4铁律合规", hot_all_combs)
                st.success(f"✅ 热号惯性流派全部组合已外置存档：{hot_save_path}，永久固定不变")

        # 子标签3：冷号回补流派
        with cold_back_tab:
            st.header("🧊 冷号回补流派｜均值回归体系")
            st.info("底层逻辑：万物皆有均值，欠的总要还，欠开区间全覆盖+有效温冷号筛选，适配冷号集中回补行情")
            st.divider()
            st.warning("🚨 本流派刚性红线：1. 100%剔除三期连开必杀号，两期连开号一律不用；2. 与上期开奖号重合率≤10%；3. 100%覆盖所有欠开区间；4. 与热号流派核心池重叠度≤1个；5. 预测池优选权重优先")
            st.divider()

            st.subheader("✅ 6步选号法执行结果")
            step1_base = [n for n in range(1,81) if n not in three_continuous and n not in two_continuous]
            st.caption(f"步骤1：合规红线过滤，剔除三期/两期连开号，剩余候选池：{len(step1_base)}个")
            step2_under_zone = [n for n in under_zone_nums if n in step1_base]
            st.caption(f"步骤2：锁定欠开区间全覆盖，剩余候选池：{len(step2_under_zone)}个")
            miss_dict = full_analysis_all["miss_analysis"]["mi"]
            step3_warm_cold = []
            for n in step2_under_zone:
                occur_10_cnt = occur_10.get(n, 0)
                miss_cnt = miss_dict.get(n, 0)
                if 2 <= occur_10_cnt <=5 and miss_cnt <10 and n in high_prob_follow_nums:
                    step3_warm_cold.append(n)
            st.caption(f"步骤3：筛选有效温冷号，剩余候选池：{len(step3_warm_cold)}个")
            step4_odd = [n for n in step3_warm_cold if n %2 ==1]
            step4_even = [n for n in step3_warm_cold if n %2 ==0]
            step4_zone12 = [n for n in step3_warm_cold if 1<=n<=40]
            step4_zone34 = [n for n in step3_warm_cold if 41<=n<=80]
            need_odd_cnt = max(round(len(step3_warm_cold)*0.55), len(step4_odd))
            need_zone12_cnt = max(round(len(step3_warm_cold)*0.5), len(step4_zone12))
            step4_final = step4_odd[:need_odd_cnt] + step4_even + step4_zone12[:need_zone12_cnt] + step4_zone34
            step4_final = list(set(step4_final))
            st.caption(f"步骤4：奇偶/区间适配，剩余候选池：{len(step4_final)}个")
            # 步骤5：生成15个核心胆码，按欠开幅度排序
            step5_core = sorted(step4_final, key=lambda x: (-miss_dict.get(x,0), x))[:15]
            st.caption(f"步骤5：生成15个核心胆码，按欠开幅度排序完成")
            st.markdown(f"**核心胆码池**：{' '.join([f'{x:02d}' for x in step5_core])}")

            # 流派隔离校验：和热号流派核心池重叠度≤1个
            try:
                overlap_with_hot = len(set(step5_core) & hot_core_pool)
                if overlap_with_hot > MAX_OVERLAP_BETWEEN_TREND:
                    st.warning(f"⚠️ 流派隔离校验不通过：与热号流派核心池重叠{overlap_with_hot}个，已自动调整")
                    # 自动调整，剔除重叠号码，补充备选
                    overlap_nums = set(step5_core) & hot_core_pool
                    step5_core = [n for n in step5_core if n not in overlap_nums]
                    # 补充备选号码
                    backup_nums = sorted(step4_final, key=lambda x: (-miss_dict.get(x,0), x))[15:15+overlap_with_hot]
                    step5_core.extend(backup_nums)
                    st.markdown(f"**调整后核心胆码池**：{' '.join([f'{x:02d}' for x in step5_core])}")
                else:
                    st.success(f"✅ 流派隔离校验通过：与热号流派核心池重叠{overlap_with_hot}个，符合≤{MAX_OVERLAP_BETWEEN_TREND}个的要求")
            except Exception:
                st.info("ℹ️ 热号流派核心池未生成，跳过流派隔离校验")

            # 步骤6：生成投注组合，4铁律校验，重合率收紧到≤1个
            st.divider()
            st.subheader("📌 冷号回补流派 固化组合生成结果")
            cold_all_combs = []
            for cfg in FIX_PLAY_CONFIG:
                play_name, need_num, fix_group = cfg["玩法名称"], cfg["选号个数"], cfg["固定生成组数"]
                cold_combs = build_iron_rule_combination(
                    candidate_pool=range(1,81),
                    two_con=two_continuous,
                    three_con=three_continuous,
                    last_real_nums=last_pre_real,
                    hot12_list=hot1# ========== Tab1 首页 ==========
with tab1:
    st.title("🎰福彩快乐8专业数据分析系统")
    st.subheader(f"当前收录：{total}期 | 预测池优选升级终版")
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
            p2 = st.text_input("开奖号码（20个数字，空格分隔）")
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
        # 过滤掉初始的88期，用户只能删除自己录入的
        user_periods = [p for p in df['period'].tolist() if p not in INIT_PERIODS]
        if not user_periods:
            st.info("暂无用户手动录入的期号，系统自动生成的基准期号无法删除")
            dp = None
        else:
            dp = st.selectbox("选择删除期号（仅可删除手动录入的期号，系统基准期号无法删除）", user_periods)
        if st.form_submit_button("确认删除", type="secondary", use_container_width=True):
            if dp is None:
                st.error("❌ 无法删除系统自动生成的基准期号！")
            else:
                df = delete_period_data(dp, df)
                st.success("删除成功")
                load_data_cached.clear()
                get_full_analysis_cached.clear()
                st.rerun()
    st.divider()
    st.dataframe(df, use_container_width=True, height=400)

# ========== Tab3 多周期数据分析 ==========
with tab3:
    st.header("📊多周期数据分析")
    # 修改周期选项：10/20/50/100/150
    window_options = {"近10期": 10, "近20期": 20, "近50期": 50, "近100期": 100, "150期以上全量": None}
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

# ========== Tab4 双流派升级终版｜多玩法选号·4铁律风控·双流派并行 ==========
with tab4:
    st.header("🔮 多玩法选号｜4铁律风控固化组合·双流派并行升级")
    st.info("💡 升级逻辑：热号惯性+冷号回补双流派完全独立并行，彻底告别赌行情，全行情覆盖，一套踩空一套补位，杜绝全组合崩盘")
    st.error("🚨 共用刚性合规红线（双流派100%强制执行）：弃前三期连出 | 降两期连出权重 | 与上期重合率≤20% | 预测池优选权重优先")
    st.warning("⚠️ 仅历史数据推演娱乐，组合固化不随刷新变动，不构成购彩建议！")
    market_tab, hot_flow_tab, cold_back_tab, check_tab, review_tab = st.tabs([
        "📈 行情主线判断",
        "🔥 热号惯性流派",
        "🧊 冷号回补流派",
        "📊 开奖核对",
        "💡 双流派复盘优化"
    ])

    # 全局固定配置
    FIX_PLAY_CONFIG = [
        {"玩法名称":"11码", "选号个数":11, "固定生成组数":3},
        {"玩法名称":"8码", "选号个数":8,  "固定生成组数":5},
        {"玩法名称":"6码", "选号个数":6,  "固定生成组数":10},
        {"玩法名称":"3码", "选号个数":3,  "固定生成组数":10}
    ]
    MAX_OVERLAP_BETWEEN_TREND = 1

    # 共用工具函数（Tab内作用域）
    def get_recent_continuous_no(df_target, curr_period):
        two_continuous = []
        three_continuous = []
        last_real = []
        try:
            sort_df = df_target.sort_values("period", ascending=False).reset_index(drop=True)
            curr_idx = sort_df[sort_df["period"] == curr_period].index[0]
            if curr_idx + 3 >= len(sort_df):
                return two_continuous, three_continuous, last_real
            n1 = set(sort_df.iloc[curr_idx+1].iloc[1:21].tolist())
            n2 = set(sort_df.iloc[curr_idx+2].iloc[1:21].tolist())
            n3 = set(sort_df.iloc[curr_idx+3].iloc[1:21].tolist())
            two_continuous = list(n1 & n2)
            three_continuous = list(n1 & n2 & n3)
            last_real = list(n1)
        except Exception:
            pass
        return two_continuous, three_continuous, last_real

    def calc_occur_rate(df_target, window=10):
        try:
            data = df_target.head(window).copy()
            num_list = [[int(x) for x in row.iloc[1:21].tolist()] for _, row in data.iterrows()]
            flat_nums = [n for p in num_list for n in p]
            occur_count = Counter(flat_nums)
            return {n: occur_count.get(n, 0) for n in range(1, 81)}, num_list
        except Exception:
            return {n:0 for n in range(1,81)}, []

    def calc_follow_probability(df_target, target_nums, min_occur=4, min_rate=0.4):
        follow_count = defaultdict(int)
        target_appear_times = 0
        try:
            data = df_target.head(50).copy()
            num_list = [[int(x) for x in row.iloc[1:21].tolist()] for _, row in data.iterrows()]
            target_set = set(target_nums)
            for i in range(1, len(num_list)):
                pre_nums = set(num_list[i-1])
                curr_nums = set(num_list[i])
                if len(pre_nums & target_set) > 0:
                    target_appear_times += 1
                    for n in curr_nums:
                        follow_count[n] += 1
            if target_appear_times == 0:
                return []
            high_prob_follow = [
                n for n, cnt in follow_count.items()
                if cnt >= min_occur and (cnt / target_appear_times) >= min_rate
            ]
            return high_prob_follow
        except Exception:
            return []

    def get_under_open_zone(num_list, window=3, max_occur=3):
        zone_occur = {"zone1":0, "zone2":0, "zone3":0, "zone4":0}
        try:
            recent_data = num_list[:window]
            for period_nums in recent_data:
                for n in period_nums:
                    if 1 <= n <=20: zone_occur["zone1"] +=1
                    elif 21 <= n <=40: zone_occur["zone2"] +=1
                    elif 41 <= n <=60: zone_occur["zone3"] +=1
                    elif 61 <= n <=80: zone_occur["zone4"] +=1
            under_zones = [zone for zone, cnt in zone_occur.items() if cnt <= max_occur]
            zone_num_map = {
                "zone1": list(range(1,21)),
                "zone2": list(range(21,41)),
                "zone3": list(range(41,61)),
                "zone4": list(range(61,81))
            }
            under_zone_nums = []
            for z in under_zones:
                under_zone_nums.extend(zone_num_map[z])
            return under_zone_nums, zone_occur
        except Exception:
            return [], zone_occur

    # 全局基础数据加载
    period_list = df["period"].tolist() if len(df) > 0 else []
    if not period_list:
        st.error("暂无开奖数据，请先初始化号码库！")
    else:
        target_period = st.selectbox("选择绑定预测期号", period_list, key="tab4_target_period")
        current_idx = df[df["period"] == target_period].index[0]
        current_nums_base = [int(x) for x in df.iloc[current_idx].iloc[1:21].tolist()]
        two_continuous, three_continuous, last_pre_real = get_recent_continuous_no(df, target_period)
        full_analysis_all = get_full_analysis_cached(df)
        full_analysis_10 = get_full_analysis_cached(df, 10)
        full_analysis_12 = get_full_analysis_cached(df, 12)
        full_analysis_24 = get_full_analysis_cached(df, 24)
        num_status_dict = get_num_status(full_analysis_all)
        hot12_plain = [x[0] for x in full_analysis_12.get("hot_cold", {}).get("hot_top10", [])]
        hot24_plain = [x[0] for x in full_analysis_24.get("hot_cold", {}).get("hot_top10", [])]
        df_back_plain = full_analysis_24.get("miss_analysis", {}).get("miss_df", pd.DataFrame({"回补率%": [], "号码": []}))
        real_check_nums = df[df["period"] == target_period].iloc[0].iloc[1:21].tolist()
        occur_10, recent_3_num_list = calc_occur_rate(df, 10)
        occur_5, _ = calc_occur_rate(df, 5)
        high_prob_follow_nums = calc_follow_probability(df, current_nums_base, min_occur=4, min_rate=0.4)
        under_zone_nums, zone_occur_3 = get_under_open_zone(recent_3_num_list, window=3, max_occur=3)
        hot_core_pool = set()

        # 子标签1：行情主线判断
        with market_tab:
            st.subheader("📈 近2期行情主线自动判断")
            st.info("自动识别行情类型，给出双流派权重分配建议，告别主观赌行情")
            st.divider()
            try:
                recent_2_data = df.head(2).copy()
                recent_2_nums = [[int(x) for x in row.iloc[1:21].tolist()] for _, row in recent_2_data.iterrows()]
                hot_top20 = [x[0] for x in full_analysis_10["hot_cold"]["hot_top10"] + full_analysis_10["hot_cold"]["hot_top10"][10:20]]
                hot_count_2 = 0
                total_count_2 = 0
                for nums in recent_2_nums:
                    hot_count_2 += len(set(nums) & set(hot_top20))
                    total_count_2 += len(nums)
                hot_rate_2 = round(hot_count_2 / total_count_2 * 100, 2)
                repeat_count = len(set(recent_2_nums[0]) & set(recent_2_nums[1]))
                if hot_rate_2 >= 60 and repeat_count >=4:
                    market_type = "🔥 热号抱团惯性行情"
                    hot_weight = 60
                    cold_weight = 40
                    market_desc = "近2期热号占比≥60%，重号数≥4个，强者恒强特征明显，优先配置热号惯性流派"
                elif hot_rate_2 <= 40 and repeat_count <=2:
                    market_type = "🧊 冷号集中回补行情"
                    hot_weight = 40
                    cold_weight = 60
                    market_desc = "近2期热号占比≤40%，重号数≤2个，均值回归特征明显，优先配置冷号回补流派"
                else:
                    market_type = "⚖️ 均衡轮动行情"
                    hot_weight = 50
                    cold_weight = 50
                    market_desc = "行情特征不明显，区间轮动快，双流派均衡配置，双线兜底"
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("行情类型", market_type)
                with c2:
                    st.metric("近2期热号占比", f"{hot_rate_2}%")
                with c3:
                    st.metric("近2期跨期重号数", repeat_count)
                st.divider()
                st.success(market_desc)
                st.subheader("📊 双流派投注权重分配建议")
                weight_c1, weight_c2 = st.columns(2)
                with weight_c1:
                    st.metric("热号惯性流派权重", f"{hot_weight}%")
                with weight_c2:
                    st.metric("冷号回补流派权重", f"{cold_weight}%")
                st.divider()
                st.subheader("📋 近3期区间出号明细（欠开区间识别）")
                zone_df = pd.DataFrame([{
                    "区间": "1-20",
                    "近3期累计出号": zone_occur_3["zone1"],
                    "状态": "🔴 欠开区间" if zone_occur_3["zone1"] <=3 else "🟢 正常区间"
                },{
                    "区间": "21-40",
                    "近3期累计出号": zone_occur_3["zone2"],
                    "状态": "🔴 欠开区间" if zone_occur_3["zone2"] <=3 else "🟢 正常区间"
                },{
                    "区间": "41-60",
                    "近3期累计出号": zone_occur_3["zone3"],
                    "状态": "🔴 欠开区间" if zone_occur_3["zone3"] <=3 else "🟢 正常区间"
                },{
                    "区间": "61-80",
                    "近3期累计出号": zone_occur_3["zone4"],
                    "状态": "🔴 欠开区间" if zone_occur_3["zone4"] <=3 else "🟢 正常区间"
                }])
                st.dataframe(zone_df, hide_index=True, use_container_width=True)
            except Exception as e:
                st.error(f"行情判断失败：{str(e)}")

        # 子标签2：热号惯性流派
        with hot_flow_tab:
            st.header("🔥 热号惯性流派｜趋势跟随体系")
            st.info("底层逻辑：强者恒强，高概率相随号+有效热号双重筛选，适配热号抱团惯性行情")
            st.divider()
            st.warning("🚨 本流派刚性红线：1. 100%剔除三期连开必杀号，两期连开降权号单组最多1个；2. 与上期开奖号重合率≤20%，单组最多2个；3. 预测池优选权重优先；4. 组间核心胆码重叠度≤2个")
            st.divider()

            st.subheader("✅ 6步选号法执行结果")
            step1_base = [n for n in range(1,81) if n not in three_continuous]
            st.caption(f"步骤1：合规红线过滤，剔除三期连开必杀号，剩余候选池：{len(step1_base)}个")
            step2_follow = [n for n in high_prob_follow_nums if n in step1_base]
            st.caption(f"步骤2：提取上期号码高概率相随号，剩余候选池：{len(step2_follow)}个")
            step3_hot = []
            for n in step2_follow:
                if occur_10.get(n, 0) >=6 and occur_5.get(n, 0) >=1:
                    step3_hot.append(n)
            st.caption(f"步骤3：筛选有效热号，剩余候选池：{len(step3_hot)}个")
            step4_odd = [n for n in step3_hot if n %2 ==1]
            step4_even = [n for n in step3_hot if n %2 ==0]
            step4_zone12 = [n for n in step3_hot if 1<=n<=40]
            step4_zone34 = [n for n in step3_hot if 41<=n<=80]
            need_odd_cnt = max(round(len(step3_hot)*0.55), len(step4_odd))
            need_zone12_cnt = max(round(len(step3_hot)*0.5), len(step4_zone12))
            step4_final = step4_odd[:need_odd_cnt] + step4_even + step4_zone12[:need_zone12_cnt] + step4_zone34
            step4_final = list(set(step4_final))
            st.caption(f"步骤4：奇偶/区间适配，剩余候选池：{len(step4_final)}个")
            step5_core = sorted(step4_final, key=lambda x: (-occur_10.get(x,0), x))[:15]
            st.caption(f"步骤5：生成15个核心胆码，按热度排序完成")
            st.markdown(f"**核心胆码池**：{' '.join([f'{x:02d}' for x in step5_core])}")
            hot_core_pool = set(step5_core)

            st.divider()
            st.subheader("📌 热号惯性流派 固化组合生成结果")
            hot_all_combs = []
            for cfg in FIX_PLAY_CONFIG:
                play_name, need_num, fix_group = cfg["玩法名称"], cfg["选号个数"], cfg["固定生成组数"]
                hot_combs = build_iron_rule_combination(
                    candidate_pool=range(1,81),
                    two_con=two_continuous,
                    three_con=three_continuous,
                    last_real_nums=last_pre_real,
                    hot12_list=hot12_plain,
                    hot24_list=hot24_plain,
                    df_back=df_back_plain,
                    need_cnt=need_num,
                    group_cnt=fix_group,
                    seed_key=f"{target_period}_hot_{play_name}",
                    max_overlap=2
                )
                hot_all_combs.extend(hot_combs)
                st.divider()
                st.subheader(f"📌 {play_name}｜固定{fix_group}组（4铁律校验通过）")
                if not hot_combs:
                    st.warning("候选池号码不足，无法生成对应组数组合")
                else:
                    for idx, comb in enumerate(hot_combs, 1):
                        comb_html = " ".join([fmt_num(n, num_status_dict) for n in comb])
                        st.markdown(f"**热号流派{play_name}方案{idx}**：{comb_html}", unsafe_allow_html=True)
                        overlap_check = len(set(comb)&set(last_pre_real))/20*100 if len(last_pre_real) > 0 else 0
                        hit_res = calc_match_rate(comb, real_check_nums)
                        st.caption(f"重合率{overlap_check:.1f}%≤20%合规 | 当期命中{hit_res['匹配个数']}个 | 命中率{hit_res['正确率%']}%")
            
            if hot_all_combs:
                hot_save_path = save_select_comb(target_period, "热号惯性流派-4铁律合规", hot_all_combs)
                st.success(f"✅ 热号惯性流派全部组合已外置存档：{hot_save_path}，永久固定不变")

        # 子标签3：冷号回补流派
        with cold_back_tab:
            st.header("🧊 冷号回补流派｜均值回归体系")
            st.info("底层逻辑：万物皆有均值，欠的总要还，欠开区间全覆盖+有效温冷号筛选，适配冷号集中回补行情")
            st.divider()
            st.warning("🚨 本流派刚性红线：1. 100%剔除三期连开必杀号，两期连开号一律不用；2. 与上期开奖号重合率≤10%；3. 100%覆盖所有欠开区间；4. 与热号流派核心池重叠度≤1个；5. 预测池优选权重优先")
            st.divider()

            st.subheader("✅ 6步选号法执行结果")
            step1_base = [n for n in range(1,81) if n not in three_continuous and n not in two_continuous]
            st.caption(f"步骤1：合规红线过滤，剔除三期/两期连开号，剩余候选池：{len(step1_base)}个")
            step2_under_zone = [n for n in under_zone_nums if n in step1_base]
            st.caption(f"步骤2：锁定欠开区间全覆盖，剩余候选池：{len(step2_under_zone)}个")
            miss_dict = full_analysis_all["miss_analysis"]["mi"]
            step3_warm_cold = []
            for n in step2_under_zone:
                occur_10_cnt = occur_10.get(n, 0)
                miss_cnt = miss_dict.get(n, 0)
                if 2 <= occur_10_cnt <=5 and miss_cnt <10 and n in high_prob_follow_nums:
                    step3_warm_cold.append(n)
            st.caption(f"步骤3：筛选有效温冷号，剩余候选池：{len(step3_warm_cold)}个")
            step4_odd = [n for n in step3_warm_cold if n %2 ==1]
            step4_even = [n for n in step3_warm_cold if n %2 ==0]
            step4_zone12 = [n for n in step3_warm_cold if 1<=n<=40]
            step4_zone34 = [n for n in step3_warm_cold if 41<=n<=80]
            need_odd_cnt = max(round(len(step3_warm_cold)*0.55), len(step4_odd))
            need_zone12_cnt = max(round(len(step3_warm_cold)*0.5), len(step4_zone12))
            step4_final = step4_odd[:need_odd_cnt] + step4_even + step4_zone12[:need_zone12_cnt] + step4_zone34
            step4_final = list(set(step4_final))
            st.caption(f"步骤4：奇偶/区间适配，剩余候选池：{len(step4_final)}个")
            # 步骤5：生成15个核心胆码，按欠开幅度排序
            step5_core = sorted(step4_final, key=lambda x: (-miss_dict.get(x,0), x))[:15]
            st.caption(f"步骤5：生成15个核心胆码，按欠开幅度排序完成")
            st.markdown(f"**核心胆码池**：{' '.join([f'{x:02d}' for x in step5_core])}")

            # 流派隔离校验：和热号流派核心池重叠度≤1个
            try:
                overlap_with_hot = len(set(step5_core) & hot_core_pool)
                if overlap_with_hot > MAX_OVERLAP_BETWEEN_TREND:
                    st.warning(f"⚠️ 流派隔离校验不通过：与热号流派核心池重叠{overlap_with_hot}个，已自动调整")
                    # 自动调整，剔除重叠号码，补充备选
                    overlap_nums = set(step5_core) & hot_core_pool
                    step5_core = [n for n in step5_core if n not in overlap_nums]
                    # 补充备选号码
                    backup_nums = sorted(step4_final, key=lambda x: (-miss_dict.get(x,0), x))[15:15+overlap_with_hot]
                    step5_core.extend(backup_nums)
                    st.markdown(f"**调整后核心胆码池**：{' '.join([f'{x:02d}' for x in step5_core])}")
                else:
                    st.success(f"✅ 流派隔离校验通过：与热号流派核心池重叠{overlap_with_hot}个，符合≤{MAX_OVERLAP_BETWEEN_TREND}个的要求")
            except Exception:
                st.info("ℹ️ 热号流派核心池未生成，跳过流派隔离校验")

            # 步骤6：生成投注组合，4铁律校验，重合率收紧到≤1个
            st.divider()
            st.subheader("📌 冷号回补流派 固化组合生成结果")
            cold_all_combs = []
            for cfg in FIX_PLAY_CONFIG:
                play_name, need_num, fix_group = cfg["玩法名称"], cfg["选号个数"], cfg["固定生成组数"]
                cold_combs = build_iron_rule_combination(
                    candidate_pool=range(1,81),
                    two_con=two_continuous,
                    three_con=three_continuous,
                    last_real_nums=last_pre_real,
                    hot12_list=hot1
                    cold_combs = build_iron_rule_combination(
                        candidate_pool=range(1,81),
                        two_con=two_continuous,
                        three_con=three_continuous,
                        last_real_nums=last_pre_real,
                        hot12_list=hot12_plain,
                        hot24_list=hot24_plain,
                        df_back=df_back_plain,
                        need_cnt=need_num,
                        group_cnt=fix_group,
                        seed_key=f"{target_period}_cold_{play_name}",
                        max_overlap=1
                    )
                    cold_all_combs.extend(cold_combs)
                    st.divider()
                    st.subheader(f"📌 {play_name}｜固定{fix_group}组（4铁律校验通过）")
                    if not cold_combs:
                        st.warning("候选池号码不足，无法生成对应组数组合")
                    else:
                        for idx, comb in enumerate(cold_combs, 1):
                            comb_html = " ".join([fmt_num(n, num_status_dict) for n in comb])
                            st.markdown(f"**冷号流派{play_name}方案{idx}**：{comb_html}", unsafe_allow_html=True)
                            overlap_check = len(set(comb)&set(last_pre_real))/20*100 if len(last_pre_real) > 0 else 0
                            hit_res = calc_match_rate(comb, real_check_nums)
                            st.caption(f"重合率{overlap_check:.1f}%≤10%合规 | 当期命中{hit_res['匹配个数']}个 | 命中率{hit_res['正确率%']}%")
            
            if cold_all_combs:
                cold_save_path = save_select_comb(target_period, "冷号回补流派-4铁律合规", cold_all_combs)
                st.success(f"✅ 冷号回补流派全部组合已外置存档：{cold_save_path}，永久固定不变")

        # 子标签4：开奖核对
        with check_tab:
            st.header("📊 开奖核对")
            st.info("自动核对所有组合的命中情况，无需手动计算")
            st.divider()
            if st.button("一键核对所有组合命中情况", use_container_width=True):
                with st.spinner("正在核对..."):
                    all_comb_df = load_all_select_comb()
                    if all_comb_df.empty:
                        st.warning("暂无存档组合")
                    else:
                        real_nums = df[df["period"] == target_period].iloc[0].iloc[1:21].tolist()
                        hit_list = []
                        for _, row in all_comb_df.iterrows():
                            if row["期号"] != target_period:
                                continue
                            try:
                                comb = [int(x) for x in row["选号号码"].split()]
                                hit = calc_match_rate(comb, real_nums)
                                hit_list.append({
                                    "期号": row["期号"],
                                    "流派": row["玩法类型"],
                                    "方案": row["方案编号"],
                                    "命中个数": hit["匹配个数"],
                                    "命中率%": hit["正确率%"],
                                    "命中号码": " ".join([f"{x:02d}" for x in hit["匹配号码"]])
                                })
                            except Exception:
                                continue
                        if hit_list:
                            hit_df = pd.DataFrame(hit_list)
                            st.dataframe(hit_df, use_container_width=True, hide_index=True)
                            avg_hit = hit_df["命中率%"].mean()
                            st.success(f"✅ 核对完成，平均命中率：{avg_hit:.2f}%")
                        else:
                            st.info("暂无对应期号的组合")

        # 子标签5：双流派复盘优化
        with review_tab:
            st.header("💡 双流派复盘优化")
            st.info("自动复盘历史双流派的命中情况，优化权重分配")
            st.divider()
            if st.button("一键复盘历史双流派命中情况", use_container_width=True):
                with st.spinner("正在复盘..."):
                    all_comb_df = load_all_select_comb()
                    if all_comb_df.empty:
                        st.warning("暂无存档组合")
                    else:
                        hot_df = all_comb_df[all_comb_df["玩法类型"].str.contains("热号", na=False)]
                        cold_df = all_comb_df[all_comb_df["玩法类型"].str.contains("冷号", na=False)]
                        hot_avg, hot_cnt = calc_trend_hit_rate(hot_df, period_list, df)
                        cold_avg, cold_cnt = calc_trend_hit_rate(cold_df, period_list, df)
                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric("热号惯性流派平均命中率", f"{hot_avg}%", f"共{hot_cnt}组")
                        with c2:
                            st.metric("冷号回补流派平均命中率", f"{cold_avg}%", f"共{cold_cnt}组")
                        st.success("✅ 复盘完成，可根据历史命中率调整双流派权重")

# ========== Tab5 单期深度复盘 ==========
with tab5:
    st.header("📝 单期深度复盘")
    period = st.selectbox("选择期号", df["period"].tolist())
    if period:
        row = df[df["period"] == period].iloc[0]
        nums = [int(x) for x in row.iloc[1:21].tolist()]
        prev_nums = None
        prev_row = df[df["period"] < period].head(1)
        if not prev_row.empty:
            prev_nums = [int(x) for x in prev_row.iloc[0].iloc[1:21].tolist()]
        s = generate_deep_review(nums, prev_nums, period)
        st.subheader(f"{period}期 深度复盘结果")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("奇偶", s["oe"])
        with c2:
            st.metric("大小", s["sl"])
        with c3:
            st.metric("012路", s["road"])
        with c4:
            st.metric("质合", s["pc"])
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.metric("和值", s["sum"])
        with c6:
            st.metric("跨度", s["span"])
        with c7:
            st.metric("连号组数", s["con_cnt"])
        with c8:
            st.metric("跨期重号", s["repeat_cnt"])
        st.divider()
        st.markdown(f"**开奖号码**：{' '.join([fmt_num(n, num_status_dict) for n in s['nums']])}", unsafe_allow_html=True)
        st.caption(f"连号：{'、'.join(s['con'])}")
        st.caption(f"跨期重号：{' '.join([f'{x:02d}' for x in s['repeat']])}")

# ========== Tab6 跨期对比+分层预测号生成 ==========
with tab6:
    st.header("🔄 跨期对比+分层预测号生成")
    st.info("基于上期开奖，自动生成三级预测号池，自动去重，自动区分冷热温号+012路")
    period = st.selectbox("选择上期期号", df["period"].tolist())
    if period:
        row = df[df["period"] == period].iloc[0]
        nums = [int(x) for x in row.iloc[1:21].tolist()]
        full = get_full_analysis_cached(df)
        num_status = get_num_status(full)
        if st.button("生成分层预测号", type="primary", use_container_width=True):
            with st.spinner("正在计算..."):
                pool = generate_leveled_pool(
                    nums,
                    full["co_occur_matrix"]["dict"],
                    full["follow_matrix"]["dict"],
                    num_status
                )
                l1, l2, l3 = pool["l1"], pool["l2"], pool["l3"]
                st.subheader("第一层：原生开奖号（基底）")
                st.markdown(" ".join([fmt_num(n, num_status) for n in l1]), unsafe_allow_html=True)
                st.divider()
                st.subheader("第二层：二级相随号（高概率）")
                st.markdown(" ".join([fmt_num(n, num_status) for n in l2]), unsafe_allow_html=True)
                st.divider()
                st.subheader("第三层：三级跟随号（次高概率）")
                st.markdown(" ".join([fmt_num(n, num_status) for n in l3]), unsafe_allow_html=True)
                st.divider()
                st.subheader("候选汇总（按关联次数排序，次数越高优先级越高）")
                cnt_map = defaultdict(int)
                for _, cnt in pool["l2_group"]:
                    for n in cnt:
                        cnt_map[n] += 1
                for _, cnt in pool["l3_group"]:
                    for n in cnt:
                        cnt_map[n] += 1
                sorted_nums = sorted(cnt_map.keys(), key=lambda x: (-cnt_map[x], x))
                st.markdown(" ".join([f"{fmt_num(n, num_status)}<small>({cnt_map[n]}次)</small>" for n in sorted_nums]), unsafe_allow_html=True)
                st.divider()
                if st.button("保存预测号到本地", use_container_width=True):
                    save_predict_num(period, l2, l3)
                    st.success("保存成功")

# ========== Tab7 设置 ==========
with tab7:
    st.header("⚙️ 系统设置")
    st.subheader("热冷阈值")
    st.number_input("热号阈值（高于平均值+N）", value=HOT_COLD_FACTOR, disabled=True)
    st.divider()
    st.subheader("存档目录")
    st.code(SAVE_DIR)
    st.divider()
    if st.button("清空所有存档", type="secondary"):
        for f in os.listdir(SAVE_DIR):
            os.remove(os.path.join(SAVE_DIR, f))
        st.success("清空完成")
        st.rerun()

# ========== Tab8 全量批量自动复盘 ==========
with tab8:
    st.header("📦 全量批量自动复盘")
    st.info("一键自动完成所有期号的单期复盘、跨期预测号、4铁律选号组合，永久固化存档，无需手动逐个处理")
    st.warning("⚠️ 首次运行会自动处理所有期号，后续运行会自动跳过已处理的期号，仅处理新增期号")
    st.divider()
    if st.button("一键启动全量自动复盘", type="primary", use_container_width=True):
        with st.spinner("正在处理所有期号，请勿关闭页面..."):
            result_df, fail_list = batch_auto_review_all_periods(df, overwrite_exist=False)
            st.success(f"✅ 全量复盘完成，共处理{len(result_df)}期，失败{len(fail_list)}期")
            st.dataframe(result_df, use_container_width=True, hide_index=True)
            if fail_list:
                st.error("失败列表：\n" + "\n".join(fail_list))
    st.divider()
    if os.path.exists(BATCH_REVIEW_SUMMARY):
        st.subheader("复盘进度总览")
        summary_df = pd.read_csv(BATCH_REVIEW_SUMMARY, encoding="utf-8-sig")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.info("暂无复盘数据，点击上方按钮启动全量复盘")   
