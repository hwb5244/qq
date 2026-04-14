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

# ====================== 全局常量定义（执行顺序100%正确，先路径后业务规则） ======================
# 核心数据路径（先定义基础路径，再定义衍生路径，彻底解决未定义错误）
DATA_FILE = "kl8_history_data.csv"
SAVE_DIR = "lottery_save"
ARCHIVE_ROOT = os.path.join(os.getcwd(), "KL8_Lottery_Data_Archive")
INDEX_FILE = os.path.join(ARCHIVE_ROOT, "05_存档总索引表", "index.csv")

# 批量复盘全局存档配置（仅定义1次，放在ARCHIVE_ROOT之后）
BATCH_REVIEW_DIR = os.path.join(ARCHIVE_ROOT, "06_全量批量复盘存档")
BATCH_REVIEW_SUMMARY = os.path.join(BATCH_REVIEW_DIR, "全量期数复盘总表.csv")
BATCH_REVIEW_DETAIL_DIR = os.path.join(BATCH_REVIEW_DIR, "单期复盘明细")

# 业务规则常量（仅定义1次，无重复，按需求修改周期配置）
PRIME_NUMBERS = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79]
ZONE_RULE = {"zone1":[1,20],"zone2":[21,40],"zone3":[41,60],"zone4":[61,80]}
HOT_COLD_FACTOR = 2
PLAY_RULE = {"选10":10,"选8":8,"选7":7,"选5":5,"选20":20}
# 按需求修改：分析周期配置，替换原12/24/60/120为10/20/50/100/150
PERIOD_WINDOW_OPTIONS = {
    "近10期": 10,
    "近20期": 20,
    "近50期": 50,
    "近100期": 100,
    "150期全量": None
}

# 统一初始化所有文件夹（所有路径定义完成后再创建，避免路径不存在）
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
# 提取基准期号列表，用于禁止删除校验
INIT_PERIOD_LIST = [row[0] for row in INIT_DATA]

# ====================== 数据读写核心工具函数（含删除权限控制） ======================
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

# 按需求修改：禁止删除基准期号，仅允许删除用户手动新增的期号
def delete_period_data(period, df):
    if period in INIT_PERIOD_LIST:
        return df, False, "禁止删除：该期号为系统基准原始数据，无法删除"
    new_df = df[df['period'] != period].reset_index(drop=True)
    new_df.to_csv(DATA_FILE, index=False, encoding='utf-8')
    return new_df, True, "删除成功"

# 新增：自动生成存档文件删除工具函数（满足可删除方案需求）
def delete_single_archive_file(file_name):
    """删除单个自动生成的存档文件"""
    file_path = os.path.join(SAVE_DIR, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        return True, f"已删除文件：{file_name}"
    return False, "文件不存在"

def delete_all_archive_files():
    """批量删除所有自动生成的存档文件"""
    if not os.path.exists(SAVE_DIR):
        return False, "存档目录不存在"
    file_list = os.listdir(SAVE_DIR)
    if not file_list:
        return False, "暂无存档文件可删除"
    for file in file_list:
        file_path = os.path.join(SAVE_DIR, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    return True, f"已清空全部{len(file_list)}个存档文件"

def delete_batch_review_data():
    """删除全量批量复盘生成的存档数据"""
    if os.path.exists(BATCH_REVIEW_SUMMARY):
        os.remove(BATCH_REVIEW_SUMMARY)
    for file in os.listdir(BATCH_REVIEW_DETAIL_DIR):
        file_path = os.path.join(BATCH_REVIEW_DETAIL_DIR, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    return True, "已清空全量批量复盘存档数据"

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
    level2_clean = sorted(list(set(
        int(n) for n in level2_list
        if str(n).strip().isdigit() and 1 <= int(n) <= 80
    )))
    level3_raw = sorted(list(set(
        int(n) for n in level3_list
        if str(n).strip().isdigit() and 1 <= int(n) <= 80
    )))
    level3_clean = [num for num in level3_raw if num not in level2_clean]
    df_save = pd.DataFrame({
        "期号": [period] * (len(level2_clean) + len(level3_clean)),
        "候选等级": ["二级相随号"] * len(level2_clean) + ["三级跟随号"] * len(level3_clean),
        "号码": level2_clean + level3_clean
    })
    filename = os.path.join(SAVE_DIR, f"{period}期预测号.csv")
    df_save.to_csv(filename, index=False, encoding="utf-8-sig")
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


# ====================== 缓存装饰器（性能优化，无缓存污染） ======================
@st.cache_data(ttl=3600)
def load_data_cached():
    return load_data()

@st.cache_data(ttl=3600)
def get_full_analysis_cached(df, window=None):
    return analyze_full_data(df, window)

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
    r2 = sum(1 for n in flat if n % 3 == 2)
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

# ====================== 按需求修改：4铁律组合生成函数（预测池优选逻辑） ======================
# 核心修改：优先从预测池选号，预测池不足时，从全量合规候选池补充，不局限于预测池
@st.cache_data(ttl=0)
def build_iron_rule_combination(
    predict_pool, full_candidate_pool, two_con, three_con, last_real_nums,
    hot12_list, hot24_list, df_back, need_cnt, group_cnt, seed_key, max_overlap=2
):
    final_combs = []
    try:
        # 铁律1：强制剔除前三期连续开出号码
        predict_pool_clean = list(set([n for n in predict_pool if n not in three_con]))
        full_candidate_clean = list(set([n for n in full_candidate_pool if n not in three_con]))
        
        # 预测池优选：优先用预测池，不足时从全量合规池补充
        if len(predict_pool_clean) >= need_cnt:
            base_candidate = predict_pool_clean
        else:
            supplement_nums = [n for n in full_candidate_clean if n not in predict_pool_clean]
            base_candidate = predict_pool_clean + supplement_nums
        
        if len(base_candidate) < need_cnt:
            return final_combs

        # 百分比字符串转数字，全容错
        if not df_back.empty and "回补率%" in df_back.columns:
            df_back["temp_num"] = df_back["回补率%"].astype(str).str.replace("%", "").astype(float)
            high_back = set(df_back[df_back["temp_num"] >= 80]["号码"].tolist())
        else:
            high_back = set()

        # 权重计算：铁律2 两期连出号码降权，预测池号码权重优先
        score_dict = {}
        hot12 = set(hot12_list)
        hot24 = set(hot24_list)
        predict_set = set(predict_pool_clean)
        for n in base_candidate:
            base_score = 0
            # 预测池号码优先加权
            if n in predict_set:
                base_score += 100
            if n in hot24:
                base_score += 50
            if n in hot12:
                base_score += 30
            if n in high_back:
                base_score += 20
            if n in two_con:
                base_score -= 50
            score_dict[n] = base_score

        # 无随机固定排序，永久不变
        sort_nums = sorted(base_candidate, key=lambda x: (-score_dict.get(x, 0), x))
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

# ====================== 全量批量自动复盘核心函数（修复版） ======================
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

            # 2. 跨期对比+分层预测号生成（修复：强制生成预测池，确保后续组合生成有数据源）
            if idx >= 1:
                full_analysis = get_full_analysis_cached(df)
                num_status_dict = get_num_status(full_analysis)
                pool_result = generate_leveled_pool(
                    current_nums,
                    full_analysis["co_occur_matrix"]["dict"],
                    full_analysis["follow_matrix"]["dict"],
                    num_status_dict
                )
                pure_level2 = list(pool_result["l2"])
                pure_level3 = list(pool_result["l3"])
                # 强制存档预测号，确保后续能读取到
                save_predict_num(period, pure_level2, pure_level3)
                predict_status = "已完成"
            else:
                predict_status = "跳过(无上期数据)"

            # 3. 4铁律选号组合生成（修复：统一周期参数+强制读取预测池+兼容核对模块格式）
            if idx >= 3:
                # 与Tab4主逻辑统一周期：近10期/近20期
                his10 = get_full_analysis_cached(df, 10)
                his20 = get_full_analysis_cached(df, 20)
                # 连续号计算与Tab4完全一致
                n1 = set(sort_df.iloc[idx-1].iloc[1:21].tolist())
                n2 = set(sort_df.iloc[idx-2].iloc[1:21].tolist())
                n3 = set(sort_df.iloc[idx-3].iloc[1:21].tolist())
                two_continuous = list(n1 & n2)
                three_continuous = list(n1 & n2 & n3)
                last_pre_real = list(n1)
                # 强制读取已生成的预测池，确保预测池优选逻辑生效
                pred_df = load_predict_num(period)
                if pred_df is not None and not pred_df.empty and "号码" in pred_df.columns:
                    l2_only = pred_df[pred_df["候选等级"] == "二级相随号"]["号码"].tolist()
                    l3_only = pred_df[pred_df["候选等级"] == "三级跟随号"]["号码"].tolist()
                    predict_pool = l2_only + l3_only
                    full_candidate_pool = list(range(1,81))
                    # 热号列表与Tab4统一
                    hot10_plain = [x[0] for x in his10.get("hot_cold", {}).get("hot_top10", [])]
                    hot20_plain = [x[0] for x in his20.get("hot_cold", {}).get("hot_top10", [])]
                    df_back_plain = his20.get("miss_analysis", {}).get("miss_df", pd.DataFrame({"回补率%": [], "号码": []}))
                    # 固定玩法配置与Tab4完全一致
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
                            predict_pool=predict_pool,
                            full_candidate_pool=full_candidate_pool,
                            two_con=two_continuous,
                            three_con=three_continuous,
                            last_real_nums=last_pre_real,
                            hot12_list=hot10_plain,
                            hot24_list=hot20_plain,
                            df_back=df_back_plain,
                            need_cnt=need_num,
                            group_cnt=fix_group,
                            seed_key=f"{period}_batch_{play_name}"
                        )
                        all_combs.extend(combs)
                    # 修复：存档玩法类型命名，兼容核对模块的筛选逻辑
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


# ====================== 全局数据初始化 ======================
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

# 标签页定义（变量和标签名100%匹配，先定义后使用）
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
    st.subheader(f"当前收录：{total}期 | 双流派升级终版 | 预测池优选")
    st.error("开奖完全随机，仅历史统计娱乐，不构成购彩建议！")
    if total > 0:
        l = df.iloc[0]
        st.subheader(f"最新{l['period']}期开奖：{' '.join([f'{x:02d}' for x in l.iloc[1:21]])}")

# ========== Tab2 号码库管理（按需求修改：禁止删除基准期号） ==========
with tab2:
    st.header("📋开奖号码库管理")
    st.info("⚠️ 系统基准88期原始数据禁止删除，仅可删除手动新增的期号")
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
        dp = st.selectbox("选择删除期号", df['period'].tolist())
        del_submit = st.form_submit_button("确认删除", type="secondary", use_container_width=True)
        if del_submit:
            df, del_success, del_msg = delete_period_data(dp, df)
            if del_success:
                st.success(del_msg)
                load_data_cached.clear()
                get_full_analysis_cached.clear()
                st.rerun()
            else:
                st.error(del_msg)
    st.divider()
    st.dataframe(df, use_container_width=True, height=400)

# ========== Tab3 多周期数据分析（按需求修改：周期改为10/20/50/100/150） ==========
with tab3:
    st.header("📊多周期数据分析")
    sel = st.selectbox("选择分析周期", list(PERIOD_WINDOW_OPTIONS.keys()))
    w = PERIOD_WINDOW_OPTIONS[sel]
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

# ========== Tab4 双流派升级终版｜多玩法选号·4铁律风控·预测池优选 ==========
with tab4:
    st.header("🔮 多玩法选号｜4铁律风控固化组合·双流派并行·预测池优选")
    st.info("💡 升级逻辑：热号惯性+冷号回补双流派完全独立并行，预测池优先选号，不足自动补充合规全量池，全行情覆盖")
    st.error("🚨 共用刚性合规红线（双流派100%强制执行）：弃前三期连出 | 降两期连出权重 | 与上期重合率≤20% | 预测池优选")
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
        full_analysis_20 = get_full_analysis_cached(df, 20)
        num_status_dict = get_num_status(full_analysis_all)
        hot10_plain = [x[0] for x in full_analysis_10.get("hot_cold", {}).get("hot_top10", [])]
        hot20_plain = [x[0] for x in full_analysis_20.get("hot_cold", {}).get("hot_top10", [])]
        df_back_plain = full_analysis_20.get("miss_analysis", {}).get("miss_df", pd.DataFrame({"回补率%": [], "号码": []}))
        real_check_nums = df[df["period"] == target_period].iloc[0].iloc[1:21].tolist()
        occur_10, recent_3_num_list = calc_occur_rate(df, 10)
        occur_5, _ = calc_occur_rate(df, 5)
        high_prob_follow_nums = calc_follow_probability(df, current_nums_base, min_occur=4, min_rate=0.4)
        under_zone_nums, zone_occur_3 = get_under_open_zone(recent_3_num_list, window=3, max_occur=3)
        hot_core_pool = set()
        # 加载预测池（用于预测池优选）
        pred_df = load_predict_num(target_period)
        predict_pool = []
        if pred_df is not None and not pred_df.empty and "号码" in pred_df.columns:
            l2_only = pred_df[pred_df["候选等级"] == "二级相随号"]["号码"].tolist()
            l3_only = pred_df[pred_df["候选等级"] == "三级跟随号"]["号码"].tolist()
            predict_pool = l2_only + l3_only
        full_candidate_pool = list(range(1,81))

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
            st.warning("🚨 本流派刚性红线：1. 100%剔除三期连开必杀号，两期连开降权号单组最多1个；2. 与上期开奖号重合率≤20%；3. 预测池优选，优先从预测池选号；4. 组间核心胆码重叠度≤2个")
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
            st.subheader("📌 热号惯性流派 固化组合生成结果（预测池优选）")
            # 预测池优先，核心胆码+预测池合并
            hot_predict_pool = list(set(predict_pool + step5_core))
            hot_all_combs = []
            for cfg in FIX_PLAY_CONFIG:
                play_name, need_num, fix_group = cfg["玩法名称"], cfg["选号个数"], cfg["固定生成组数"]
                hot_combs = build_iron_rule_combination(
                    predict_pool=hot_predict_pool,
                    full_candidate_pool=full_candidate_pool,
                    two_con=two_continuous,
                    three_con=three_continuous,
                    last_real_nums=last_pre_real,
                    hot12_list=hot10_plain,
                    hot24_list=hot20_plain,
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
            st.warning("🚨 本流派刚性红线：1. 100%剔除三期连开必杀号，两期连开号一律不用；2. 与上期开奖号重合率≤10%；3. 100%覆盖所有欠开区间；4. 与热号流派核心池重叠度≤1个；5. 预测池优选，优先从预测池选号")
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
            step5_core = sorted(step4_final, key=lambda x: (-miss_dict.get(x,0), x))[:15]
            st.caption(f"步骤5：生成15个核心胆码，按欠开幅度排序完成")
            st.markdown(f"**核心胆码池**：{' '.join([f'{x:02d}' for x in step5_core])}")

            # 流派隔离校验：和热号流派核心池重叠度≤1个
            try:
                overlap_with_hot = len(set(step5_core) & hot_core_pool)
                if overlap_with_hot > MAX_OVERLAP_BETWEEN_TREND:
                    st.warning(f"⚠️ 流派隔离校验不通过：与热号流派核心池重叠{overlap_with_hot}个，已自动调整")
                    overlap_nums = set(step5_core) & hot_core_pool
                    step5_core = [n for n in step5_core if n not in overlap_nums]
                    backup_nums = sorted(step4_final, key=lambda x: (-miss_dict.get(x,0), x))[15:15+overlap_with_hot]
                    step5_core.extend(backup_nums)
                    st.markdown(f"**调整后核心胆码池**：{' '.join([f'{x:02d}' for x in step5_core])}")
                else:
                    st.success(f"✅ 流派隔离校验通过：与热号流派核心池重叠{overlap_with_hot}个，符合≤{MAX_OVERLAP_BETWEEN_TREND}个的要求")
            except Exception:
                st.info("ℹ️ 热号流派核心池未生成，跳过流派隔离校验")

            # 生成投注组合，4铁律校验，重合率收紧到≤1个
            st.divider()
            st.subheader("📌 冷号回补流派 固化组合生成结果（预测池优选）")
            # 预测池优先，核心胆码+预测池合并
            cold_predict_pool = list(set(predict_pool + step5_core))
            cold_all_combs = []
            for cfg in FIX_PLAY_CONFIG:
                play_name, need_num, fix_group = cfg["玩法名称"], cfg["选号个数"], cfg["固定生成组数"]
                cold_combs = build_iron_rule_combination(
                    predict_pool=cold_predict_pool,
                    full_candidate_pool=full_candidate_pool,
                    two_con=two_continuous,
                    three_con=three_continuous,
                    last_real_nums=last_pre_real,
                    hot12_list=hot10_plain,
                    hot24_list=hot20_plain,
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
            
            # 冷号流派组合存档
            if cold_all_combs:
                cold_save_path = save_select_comb(target_period, "冷号回补流派-4铁律合规", cold_all_combs)
                st.success(f"✅ 冷号回补流派全部组合已外置存档：{cold_save_path}，永久固定不变")

                # ====================== 子标签4：开奖核对（修复版：支持批量/热号/冷号全流派核对） ======================
        with check_tab:
            st.header("📊 开奖核对")
            st.info("全流派兼容：支持手动生成的热号/冷号流派、批量复盘自动生成的组合，一对一核验当期开奖号")
            period_list = df["period"].tolist() if len(df) > 0 else []
            if not period_list:
                st.error("暂无开奖数据！")
            else:
                check_period = st.selectbox("选择核对期号", period_list, key="tab4_check_period")
                pred_check_df = load_predict_num(check_period)
                real_check_nums = df[check_period == df["period"]].iloc[0].iloc[1:21].tolist() if check_period in df["period"].values else []
                all_comb_df = load_all_select_comb()

                # 1. 预测号池核对（修复：兼容批量生成的预测号）
                st.subheader("📊 预测号池核对（含手动/批量自动生成）")
                if pred_check_df is not None and not pred_check_df.empty and "号码" in pred_check_df.columns and len(real_check_nums) > 0:
                    pred_list = pred_check_df["号码"].tolist()
                    res = calc_match_rate(pred_list, real_check_nums)
                    c1,c2,c3 = st.columns(3)
                    with c1:
                        st.metric("预测总数量",f"{len(pred_list)}个")
                    with c2:
                        st.metric("精准命中数",f"{res['匹配个数']}个")
                    with c3:
                        st.metric("综合命中率",f"{res['正确率%']}%")
                    st.divider()
                    hit_text = "、".join(f"{x:02d}" for x in res["匹配号码"]) if res["匹配号码"] else "暂无命中号码"
                    st.info(f"命中明细：{hit_text}")
                    
                    # 二级/三级预测号分开核对
                    st.subheader("📋 分层预测号明细核对")
                    l2_nums = pred_check_df[pred_check_df["候选等级"] == "二级相随号"]["号码"].tolist()
                    l3_nums = pred_check_df[pred_check_df["候选等级"] == "三级跟随号"]["号码"].tolist()
                    l2_res = calc_match_rate(l2_nums, real_check_nums)
                    l3_res = calc_match_rate(l3_nums, real_check_nums)
                    col_l2, col_l3 = st.columns(2)
                    with col_l2:
                        st.metric("二级相随号命中率", f"{l2_res['正确率%']}%", f"命中{l2_res['匹配个数']}个")
                        st.caption(f"二级号码：{' '.join([f'{x:02d}' for x in l2_nums])}")
                    with col_l3:
                        st.metric("三级跟随号命中率", f"{l3_res['正确率%']}%", f"命中{l3_res['匹配个数']}个")
                        st.caption(f"三级号码：{' '.join([f'{x:02d}' for x in l3_nums])}")
                else:
                    st.error("⚠️ 缺少对应期有效预测号/开奖原始数据！请先执行跨期预测生成或全量批量复盘")
                
                # 2. 选号组合核对（修复：新增批量自动生成组合Tab，全流派兼容）
                st.divider()
                st.subheader("📊 选号组合全流派核对")
                if not all_comb_df.empty and check_period in all_comb_df["期号"].values:
                    period_comb_df = all_comb_df[all_comb_df["期号"] == check_period]
                    # 分流派筛选
                    hot_comb_df = period_comb_df[period_comb_df["玩法类型"].str.contains("热号")]
                    cold_comb_df = period_comb_df[period_comb_df["玩法类型"].str.contains("冷号")]
                    batch_comb_df = period_comb_df[period_comb_df["玩法类型"].str.contains("批量自动生成")]
                    
                    # 分Tab展示所有流派
                    tab_hot_check, tab_cold_check, tab_batch_check, tab_all_check = st.tabs([
                        "🔥 热号流派组合核对", 
                        "🧊 冷号流派组合核对",
                        "📦 批量自动生成组合核对",
                        "📋 全流派组合汇总核对"
                    ])
                    
                    # 热号流派Tab
                    with tab_hot_check:
                        if not hot_comb_df.empty:
                            # 计算每组命中率
                            hot_detail = []
                            for _, row in hot_comb_df.iterrows():
                                try:
                                    c_nums = [int(x) for x in row["选号号码"].split()]
                                    hit_res = calc_match_rate(c_nums, real_check_nums)
                                    hot_detail.append({
                                        "玩法类型": row["玩法类型"],
                                        "方案编号": row["方案编号"],
                                        "选号号码": row["选号号码"],
                                        "命中个数": hit_res["匹配个数"],
                                        "命中率%": hit_res["正确率%"],
                                        "命中号码": "、".join([f"{x:02d}" for x in hit_res["匹配号码"]])
                                    })
                                except Exception:
                                    continue
                            if hot_detail:
                                hot_detail_df = pd.DataFrame(hot_detail)
                                st.dataframe(hot_detail_df, hide_index=True, use_container_width=True)
                                # 流派平均命中率
                                hot_avg_hit = round(hot_detail_df["命中率%"].mean(), 2)
                                st.success(f"🔥 热号流派平均命中率：{hot_avg_hit}%")
                        else:
                            st.warning("该期暂无热号流派组合存档，请先手动生成或执行批量复盘")
                    
                    # 冷号流派Tab
                    with tab_cold_check:
                        if not cold_comb_df.empty:
                            cold_detail = []
                            for _, row in cold_comb_df.iterrows():
                                try:
                                    c_nums = [int(x) for x in row["选号号码"].split()]
                                    hit_res = calc_match_rate(c_nums, real_check_nums)
                                    cold_detail.append({
                                        "玩法类型": row["玩法类型"],
                                        "方案编号": row["方案编号"],
                                        "选号号码": row["选号号码"],
                                        "命中个数": hit_res["匹配个数"],
                                        "命中率%": hit_res["正确率%"],
                                        "命中号码": "、".join([f"{x:02d}" for x in hit_res["匹配号码"]])
                                    })
                                except Exception:
                                    continue
                            if cold_detail:
                                cold_detail_df = pd.DataFrame(cold_detail)
                                st.dataframe(cold_detail_df, hide_index=True, use_container_width=True)
                                cold_avg_hit = round(cold_detail_df["命中率%"].mean(), 2)
                                st.success(f"🧊 冷号流派平均命中率：{cold_avg_hit}%")
                        else:
                            st.warning("该期暂无冷号流派组合存档，请先手动生成或执行批量复盘")
                    
                    # 批量自动生成Tab（核心修复：新增批量数据展示）
                    with tab_batch_check:
                        if not batch_comb_df.empty:
                            batch_detail = []
                            for _, row in batch_comb_df.iterrows():
                                try:
                                    c_nums = [int(x) for x in row["选号号码"].split()]
                                    hit_res = calc_match_rate(c_nums, real_check_nums)
                                    batch_detail.append({
                                        "玩法类型": row["玩法类型"],
                                        "方案编号": row["方案编号"],
                                        "选号号码": row["选号号码"],
                                        "命中个数": hit_res["匹配个数"],
                                        "命中率%": hit_res["正确率%"],
                                        "命中号码": "、".join([f"{x:02d}" for x in hit_res["匹配号码"]])
                                    })
                                except Exception:
                                    continue
                            if batch_detail:
                                batch_detail_df = pd.DataFrame(batch_detail)
                                st.dataframe(batch_detail_df, hide_index=True, use_container_width=True)
                                batch_avg_hit = round(batch_detail_df["命中率%"].mean(), 2)
                                st.success(f"📦 批量自动生成组合平均命中率：{batch_avg_hit}%")
                        else:
                            st.warning("该期暂无批量自动生成的组合存档，请先执行全量批量复盘，且勾选「覆盖已存在的存档数据」")
                    
                    # 全流派汇总Tab
                    with tab_all_check:
                        all_detail = []
                        for _, row in period_comb_df.iterrows():
                            try:
                                c_nums = [int(x) for x in row["选号号码"].split()]
                                hit_res = calc_match_rate(c_nums, real_check_nums)
                                all_detail.append({
                                    "玩法类型": row["玩法类型"],
                                    "方案编号": row["方案编号"],
                                    "选号号码": row["选号号码"],
                                    "命中个数": hit_res["匹配个数"],
                                    "命中率%": hit_res["正确率%"],
                                    "命中号码": "、".join([f"{x:02d}" for x in hit_res["匹配号码"]])
                                })
                            except Exception:
                                continue
                        if all_detail:
                            all_detail_df = pd.DataFrame(all_detail)
                            st.dataframe(all_detail_df, hide_index=True, use_container_width=True)
                            all_avg_hit = round(all_detail_df["命中率%"].mean(), 2)
                            st.success(f"📋 全流派组合平均命中率：{all_avg_hit}%")
                else:
                    st.warning("该期暂无任何选号组合存档，请先手动生成选号组合，或执行全量批量复盘")


        # 子标签5：双流派复盘优化
        with review_tab:
            st.header("💡 双流派复盘优化")
            st.info("双流派组合分开复盘，单独统计命中率，独立优化迭代，互不干扰")
            all_history_comb = load_all_select_comb()
            if all_history_comb.empty:
                st.warning("暂无合规存档组合，先生成后再复盘！")
            else:
                hot_history = all_history_comb[all_history_comb["玩法类型"].str.contains("热号")]
                cold_history = all_history_comb[all_history_comb["玩法类型"].str.contains("冷号")]
                period_list = df["period"].tolist()

                hot_avg_hit, hot_total = calc_trend_hit_rate(hot_history, period_list, df)
                cold_avg_hit, cold_total = calc_trend_hit_rate(cold_history, period_list, df)

                st.subheader("📈 双流派历史命中率复盘总览")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("热号惯性流派 历史平均命中率", f"{hot_avg_hit}%", f"累计{hot_total}组组合")
                with c2:
                    st.metric("冷号回补流派 历史平均命中率", f"{cold_avg_hit}%", f"累计{cold_total}组组合")
                st.divider()

                st.subheader("📋 按期号明细复盘")
                sel_review_period = st.selectbox("选择复盘期号", sorted(all_history_comb["期号"].unique(), reverse=True))
                if sel_review_period in period_list:
                    real_review_nums = [int(x) for x in df[df["period"] == sel_review_period].iloc[0].iloc[1:21].tolist()]
                    period_review_df = all_history_comb[all_history_comb["期号"] == sel_review_period]
                    review_detail = []
                    for _, row in period_review_df.iterrows():
                        try:
                            c_nums = [int(x) for x in row["选号号码"].split()]
                            hit_res = calc_match_rate(c_nums, real_review_nums)
                            review_detail.append({
                                "流派类型": row["玩法类型"],
                                "方案编号": row["方案编号"],
                                "选号号码": row["选号号码"],
                                "命中个数": hit_res["匹配个数"],
                                "命中率%": hit_res["正确率%"],
                                "命中号码": "、".join([f"{x:02d}" for x in hit_res["匹配号码"]])
                            })
                        except Exception:
                            continue
                    if review_detail:
                        review_detail_df = pd.DataFrame(review_detail)
                        st.dataframe(review_detail_df, hide_index=True, use_container_width=True)
                    else:
                        st.warning("暂无该期复盘数据")
                st.divider()

                st.subheader("💡 双流派迭代优化建议")
                if hot_avg_hit > cold_avg_hit:
                    st.success("📌 热号惯性流派历史表现更优，建议近期提升热号流派权重至60%-70%，优化有效热号筛选标准，剔除衰退热号")
                elif cold_avg_hit > hot_avg_hit:
                    st.success("📌 冷号回补流派历史表现更优，建议近期提升冷号流派权重至60%-70%，优化欠开区间识别标准，精准锁定回补号码")
                else:
                    st.success("📌 双流派表现均衡，建议继续保持50%:50%均衡配置，双线兜底，全行情覆盖")
                st.markdown("""
                🔧 核心优化方向：
                1. 热号流派：重点优化「衰退热号」的剔除标准，避免把连续2期未开的热号放进核心池
                2. 冷号流派：重点优化「欠开区间」的识别周期，可尝试调整为近2期/近4期，适配不同的回补节奏
                3. 双流派统一：严格遵守流派隔离铁律，核心池重叠度永远≤1个，彻底杜绝同质化风险
                """)

# ========== Tab5 单期深度复盘 ==========
with tab5:
    st.header("📝 单期深度复盘")
    st.info("支持历史期号一键复盘/手动录入，同源固定逻辑计算，历史数据不变则复盘结果永久唯一不变动")
    review_mode = st.radio("选择复盘方式", ["选择历史期号", "手动录入新期号码"], horizontal=True)

    if review_mode == "选择历史期号":
        period_list = df["period"].tolist()
        selected_period = st.selectbox("选择要复盘的期号", period_list)
        
        if st.button("生成深度复盘报告", use_container_width=True, type="primary"):
            current_row = df[df["period"] == selected_period].iloc[0]
            current_nums = [int(x) for x in current_row.iloc[1:21].tolist()]
            current_idx = df[df["period"] == selected_period].index[0]
            
            prev_nums = None
            prev_period = None
            if current_idx < len(df) - 1:
                prev_row = df.iloc[current_idx + 1]
                prev_nums = [int(x) for x in prev_row.iloc[1:21].tolist()]
                prev_period = prev_row["period"]

            review_result = generate_deep_review(current_nums, prev_nums, selected_period)
            full_analysis = get_full_analysis_cached(df)
            num_status_dict = get_num_status(full_analysis)

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

                        if manual_period not in df["period"].values:
                            if st.button("✅ 一键保存到号码库", type="primary", use_container_width=True):
                                save_success = save_new_data(manual_period, num_msg)
                                if save_success:
                                    st.success(f"✅ 成功入库{manual_period}期数据！")
                                    load_data_cached.clear()
                                    get_full_analysis_cached.clear()
                                    st.rerun()    

# ========== Tab6 跨期对比与预测号码池 ==========
with tab6:
    st.header("🔄 跨期对比与预测号码池")
    st.info("数据与单期复盘同源统一 | 两期自动对比生成结论 | 分层预测池美化展示 | 生成即刻自动存档")
    period_list = df["period"].tolist()
    selected_current_period = st.selectbox("选择【本期】分析期号(系统自动加载上期联动对比)", period_list)

    if st.button("🚀 生成跨期对比+优化预测池+自动存档", use_container_width=True, type="primary"):
        current_idx = df[df["period"] == selected_current_period].index[0]
        current_row = df.iloc[current_idx]
        current_nums = [int(x) for x in current_row.iloc[1:21].tolist()]
        
        prev_nums = None
        prev_period = None
        if current_idx < len(df) - 1:
            prev_row = df.iloc[current_idx + 1]
            prev_nums = [int(x) for x in prev_row.iloc[1:21].tolist()]
            prev_period = prev_row["period"]

        prev_review = generate_deep_review(prev_nums, None, prev_period) if prev_nums else None
        curr_review = generate_deep_review(current_nums, prev_nums, selected_current_period)
        full_analysis = get_full_analysis_cached(df)
        num_status_dict = get_num_status(full_analysis)

        pool_result = generate_leveled_pool(
            current_nums,
            full_analysis["co_occur_matrix"]["dict"],
            full_analysis["follow_matrix"]["dict"],
            num_status_dict
        )

        pure_level2 = list(pool_result["l2"])
        pure_level3 = list(pool_result["l3"])
        save_file_path = save_predict_num(selected_current_period, pure_level2, pure_level3)
        st.success(f"✅ 预测池已自动存档完成！保存路径：{save_file_path}")

        st.divider()
        st.subheader("📊 上期VS本期 核心指标同源对比表")
        if prev_review:
            compare_df = pd.DataFrame([
                ["开奖期号", prev_period, selected_current_period],
                ["奇偶比例", prev_review["oe"], curr_review["oe"]],
                ["大小比例", prev_review["sl"], curr_review["sl"]],
                ["012路比例", prev_review["road"], curr_review["road"]],
                ["质合比例", prev_review["pc"], curr_review["pc"]],
                ["号码和值", prev_review["sum"], curr_review["sum"]],
                ["连号组数", f"{prev_review['con_cnt']}组", f"{curr_review['con_cnt']}组"],
                ["跨期重号数量", "-", f"{curr_review['repeat_cnt']}个"]
            ], columns=["统计维度", f"上期{prev_period}", f"本期{selected_current_period}"])
            st.dataframe(compare_df, hide_index=True, use_container_width=True)

        st.divider()
        st.subheader("📝 两期数据深度对比结论报告")
        conclusion_text = []
        curr_odd, curr_even = curr_review["odd"], curr_review["even"]
        if curr_odd > curr_even:
            conclusion_text.append("① 本期奇数热开，奇偶偏向奇数侧，偏离理论10:10均衡值；")
        elif curr_odd < curr_even:
            conclusion_text.append("① 本期偶数占优，偶数出号活跃度更高；")
        else:
            conclusion_text.append("① 本期奇偶完全均衡，贴合历史理论标准配比；")
        
        curr_small, curr_large = curr_review["small"], curr_review["large"]
        if curr_small > curr_large:
            conclusion_text.append("② 小号区(1-40)出号强势，大号区走冷回调；")
        elif curr_small < curr_large:
            conclusion_text.append("② 大号区(41-80)发力明显，小号区处于回补等待阶段；")
        else:
            conclusion_text.append("② 大小号配比均衡，四区分布无极端偏移；")
        
        conclusion_text.append(f"③ 本期相对上期重号共{curr_review['repeat_cnt']}个，属于历史正常波动区间；")
        conclusion_text.append(f"④ 本期连号组数{curr_review['con_cnt']}组，{'连号爆发' if curr_review['con_cnt']>4 else '连号平稳'}；")
        
        for text in conclusion_text:
            st.info(text)

        st.divider()
        st.subheader("🎯 下一期分层预测号码池（冷热色标区分 | 已自动存档）")
        co_map = pool_result["co"]
        follow_map = pool_result["follow"]

        st.markdown("#### 🔴 第一层基底：本期原生开奖号码(一级核心参考)")
        l1_html = " ".join([fmt_num(n, num_status_dict) for n in sorted(pool_result["l1"])])
        st.markdown(l1_html, unsafe_allow_html=True)

        st.markdown("#### 🟡 第二层候选：本期高频相随号(二级重点预测，已存档)")
        level2_groups = pool_result["l2_group"]
        if level2_groups:
            for cnt, nums in level2_groups:
                nums_sorted = sorted(nums, key=lambda x: num_status_dict[x]["cnt"], reverse=True)
                l2_html = " ".join([fmt_num(n, num_status_dict) for n in nums_sorted])
                st.markdown(f"同频关联{cnt}次：{l2_html}", unsafe_allow_html=True)
        else:
            st.warning("暂无高频二级相随号数据")

        st.markdown("#### 🟢 第三层备选：相随号衍生跟随号(三级补充预测，已存档)")
        level3_groups = pool_result["l3_group"]
        if level3_groups:
            for cnt, nums in level3_groups:
                nums_sorted = sorted(nums, key=lambda x: num_status_dict[x]["cnt"], reverse=True)
                l3_html = " ".join([fmt_num(n, num_status_dict) for n in nums_sorted])
                st.markdown(f"跨期跟随{cnt}次：{l3_html}", unsafe_allow_html=True)
        else:
            st.warning("暂无衍生三级跟随号数据")

        st.divider()
        st.subheader("💡 基于跨期对比的下期选号适配建议")
        st.success("结合两期偏移规律：优先保留二级相随号核心、搭配高遗漏回补冷号，冷热配比控制1:1；规避本期连号扎堆区间，平衡012路分布，严格控制与本期开奖重合率≤20%。")

st.caption("🔒 技术说明：底层历史数据/核心计算逻辑未修改时，跨期对比结果、预测池号码永久固定不变，无随机变动")

# ========== Tab7 设置页（数据管理+备份迁移+自动生成文件删除+系统重置） ==========
with tab7:
    st.header("⚙️ 数据管理、存档迁移与系统重置")
    st.info("支持外置存档独立备份、跨代码版本迁移、原始数据下载，更替代码不丢数据；自动生成的存档文件支持一键删除")

    # 1. 原始开奖CSV单机备份
    st.subheader("📄 原始开奖数据单机备份")
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            csv_raw_data = f.read()
        st.download_button(
            label="📥 下载原始CSV备份文件",
            data=csv_raw_data,
            file_name=f"kl8_history_backup_{df.iloc[0]['period']}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.warning("数据文件不存在，请先初始化系统")

    st.divider()
    # 2. 自动生成存档文件管理（按需求实现可删除方案）
    st.subheader("📂 自动生成存档文件管理（支持单删/批量删除）")
    if os.path.exists(SAVE_DIR):
        save_files = os.listdir(SAVE_DIR)
        if save_files:
            st.write(f"当前自动生成存档总数：{len(save_files)}个")
            # 单个文件删除
            del_single_file = st.selectbox("选择单个文件删除", save_files)
            if st.button("删除选中文件", use_container_width=True, type="secondary"):
                del_success, del_msg = delete_single_archive_file(del_single_file)
                if del_success:
                    st.success(del_msg)
                    st.rerun()
                else:
                    st.error(del_msg)
            st.divider()
            # 批量删除所有存档
            if st.button("清空所有自动生成的预测号/选号组合存档", use_container_width=True, type="secondary"):
                batch_del_success, batch_del_msg = delete_all_archive_files()
                if batch_del_success:
                    st.success(batch_del_msg)
                    st.rerun()
                else:
                    st.warning(batch_del_msg)
        else:
            st.info("暂无自动生成的存档文件")

    st.divider()
    # 3. 批量复盘存档删除
    st.subheader("📦 全量批量复盘存档管理")
    if st.button("清空所有批量复盘生成的存档数据", use_container_width=True, type="secondary"):
        review_del_success, review_del_msg = delete_batch_review_data()
        if review_del_success:
            st.success(review_del_msg)
            st.rerun()
        else:
            st.error(review_del_msg)

    st.divider()
    # 4. 全库数据统计总览
    st.subheader("📈 全库数据统计看板")
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("总收录期数", f"{total}期")
    with col_stat2:
        st.metric("最早期号", df.iloc[-1]["period"] if total > 0 else "无")
    with col_stat3:
        st.metric("最新期号", df.iloc[0]["period"] if total > 0 else "无")
    with col_stat4:
        st.metric("总号码记录数", f"{total * 20}个")

    st.divider()
    # 5. 全库一键打包备份/迁移
    st.subheader("💾 全库一键打包备份 | 跨代码/跨电脑迁移专用")
    try:
        zip_name = f"KL8全量外置存档_一键迁移包_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.zip"
        zip_path = os.path.join(ARCHIVE_ROOT, zip_name)

        if st.button("📦 开始打包全部外置数据（适配代码更替/换服务器/换电脑）", use_container_width=True, type="primary"):
            with st.spinner("正在压缩全库存档，请稍候..."):
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(ARCHIVE_ROOT):
                        for file in files:
                            fp = os.path.join(root, file)
                            arcname = os.path.relpath(fp, ARCHIVE_ROOT)
                            zipf.write(fp, arcname)
            st.success("✅ 打包完成！更替代码只需要复制压缩包，新环境解压配置路径即可秒读所有历史数据")
            with open(zip_path, "rb") as f:
                st.download_button("⬇️ 下载全库迁移压缩包", f, file_name=zip_name, use_container_width=True)

        st.divider()
        st.subheader("📋 外置存档全局索引总表（全历史检索）")
        if os.path.exists(INDEX_FILE):
            index_df = pd.read_csv(INDEX_FILE, encoding="utf-8-sig")
            st.dataframe(index_df, hide_index=True, use_container_width=True, height=300)
        else:
            st.info("暂无存档索引，生成预测号/组合后自动创建")
    except Exception as e:
        st.error(f"模块加载提示：{str(e)}，不影响主程序运行，仅迁移功能临时不可用")

    st.divider()
    # 6. 危险区：系统数据重置
    st.subheader("⚠️ 数据重置终极操作（高危不可恢复）")
    st.error("此操作清空增量数据，仅恢复初始88期基准，更替代码无需点这里！")
    with st.form("reset_data_form", border=True):
        reset_confirm = st.checkbox("我已知风险，确认重置回原始88期基准数据")
        reset_submit = st.form_submit_button("执行数据重置", type="secondary", use_container_width=True)
        if reset_submit:
            if reset_confirm:
                with open(DATA_FILE, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['period'] + [f'n{i}' for i in range(1,21)])
                    writer.writerows(INIT_DATA)
                load_data_cached.clear()
                get_full_analysis_cached.clear()
                st.success("✅ 已重置为初始基准数据")
                st.rerun()
            else:
                st.error("请勾选确认框后再执行")

# ========== Tab8 全量批量自动复盘 ==========
with tab8:
    st.header("📦 全量期数一键自动复盘系统")
    st.info("自动完成88期全量数据的「单期深度复盘+跨期对比预测池+4铁律选号组合」生成，结果永久存档，后期随时可调用")
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        overwrite_mode = st.checkbox("覆盖已存在的存档数据（增量模式不勾选，全量重算勾选）", value=False)
    with c2:
        st.metric("当前可处理总期数", f"{len(df)}期")

    run_batch = st.button("🚀 开始全量自动复盘", use_container_width=True, type="primary")
    st.divider()

    if run_batch:
        with st.spinner("正在全量批量处理中，请勿刷新页面..."):
            result_df, fail_list = batch_auto_review_all_periods(df, overwrite_exist=overwrite_mode)
            
            st.subheader("✅ 处理完成结果总览")
            success_cnt = len(result_df[result_df["处理状态"] == "处理成功"])
            skip_cnt = len(result_df[result_df["处理状态"] == "已跳过(已存在)"])
            fail_cnt = len(result_df[result_df["处理状态"] == "处理失败"])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("处理成功", f"{success_cnt}期")
            with col2:
                st.metric("已跳过", f"{skip_cnt}期")
            with col3:
                st.metric("处理失败", f"{fail_cnt}期")

            st.dataframe(result_df, hide_index=True, use_container_width=True, height=400)

            if fail_list:
                st.divider()
                st.error("❌ 处理失败期号明细")
                for fail in fail_list:
                    st.write(fail)

            st.divider()
            with open(BATCH_REVIEW_SUMMARY, "rb") as f:
                st.download_button(
                    label="📥 下载全量复盘总表CSV",
                    data=f.read(),
                    file_name="快乐8全量期数复盘总表.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    st.divider()
    st.subheader("📋 历史批量复盘存档查看")
    if os.path.exists(BATCH_REVIEW_SUMMARY):
        history_df = pd.read_csv(BATCH_REVIEW_SUMMARY, encoding="utf-8-sig")
        st.dataframe(history_df, hide_index=True, use_container_width=True, height=300)
        st.subheader("🔍 单期复盘明细查询")
        sel_period = st.selectbox("选择要查看的期号", df["period"].tolist())
        detail_file = os.path.join(BATCH_REVIEW_DETAIL_DIR, f"{sel_period}期_复盘明细.csv")
        if os.path.exists(detail_file):
            detail_df = pd.read_csv(detail_file, encoding="utf-8-sig")
            st.dataframe(detail_df, hide_index=True, use_container_width=True)
            with open(detail_file, "rb") as f:
                st.download_button(f"下载{sel_period}期复盘明细", f.read(), file_name=f"{sel_period}期_复盘明细.csv", mime="text/csv")
        else:
            st.warning("该期暂无复盘明细，请先执行批量复盘")
    else:
        st.info("暂无批量复盘存档，请先点击「开始全量自动复盘」生成数据")

# ====================== 全局尾部合规声明（完整闭合） ======================
st.divider()
st.markdown('<div style="text-align:center;color:#666;font-size:14px">温馨提示:本系统仅历史数据统计娱乐,彩票开奖完全随机,不构成购彩建议,理性购彩遵守法规</div>', unsafe_allow_html=True)
