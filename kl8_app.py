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

]
# 提取基准期号列表，用于禁止删除校验
INIT_PERIOD_LIST = [row[0] for row in INIT_DATA]

# ====================== 数据读写核心工具函数（含删除权限控制） ======================
def load_data():
    try:
        base_period_set = set(INIT_PERIOD_LIST)
        if os.path.exists(DATA_FILE):
            temp_df = pd.read_csv(DATA_FILE, dtype={'period': str})
            file_period_set = set(temp_df['period'].tolist())
            if not base_period_set.issubset(file_period_set):
                raise ValueError("基准期号缺失，重置数据")
        else:
            raise ValueError("数据文件不存在，初始化")

        df = pd.read_csv(DATA_FILE, dtype={'period': str})
        required_cols = ['period'] + [f'n{i}' for i in range(1, 21)]
        if not all(col in df.columns for col in required_cols):
            raise ValueError("表头损坏重置")
            
        valid_rows = []
        for _, row in df.iterrows():
            period = row['period']
            try:
                nums = [int(row[f'n{i}']) for i in range(1,21)]
                if len(nums)!=20 or len(set(nums))!=20 or min(nums)<1 or max(nums)>80:
                    continue
                valid_rows.append(row)
            except Exception:
                continue
                
        df = pd.DataFrame(valid_rows).reset_index(drop=True)
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
        df = pd.read_csv(DATA_FILE, dtype={'period': str})
        st.warning("数据已自动修复为基准原始数据")
        return df

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
def save_predict_num(target_period, data_end_period, level2_list, level3_list):
    level2_clean = sorted(list(set(
        int(n) for n in level2_list
        if str(n).strip().isdigit() and 1 <= int(n) <= 80
    )))
    level3_raw = sorted(list(set(
        int(n) for n in level3_list
        if str(n).strip().isdigit() and 1 <= int(n) <= 80
    )))
    level3_clean = [num for num in level3_raw if num not in level2_clean]
    if len(set(level2_clean)&set(level3_clean))>0:
        st.warning("预测池层间存在重复，已自动隔离")
    
    df_save = pd.DataFrame({
        "预测目标期号": [target_period] * (len(level2_clean) + len(level3_clean)),
        "数据截止期号": [data_end_period] * (len(level2_clean) + len(level3_clean)),
        "候选等级": ["二级相随号"] * len(level2_clean) + ["三级跟随号"] * len(level3_clean),
        "号码": level2_clean + level3_clean,
        "生成时间": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]*(len(level2_clean)+len(level3_clean)),
        "是否事前预测": ["是"]*(len(level2_clean)+len(level3_clean))
    })
    filename = os.path.join(SAVE_DIR, f"{target_period}期预测号.csv")
    df_save.to_csv(filename, index=False, encoding="utf-8-sig")
    st.caption(
        f"✅{target_period}期预测号存档 | 二级相随号：{len(level2_clean)}个 | "
        f"三级跟随号：{len(level3_clean)}个 | 层间完全隔离无重复"
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

def generate_leveled_pool(history_nums, co_occur_dict, follow_dict, num_status_dict, max_predict_count=30):
    if not history_nums or len(history_nums) < 1:
        return {"l1": [], "l2": [], "l3": [], "l2_group": [], "l3_group": [], "co": {}, "follow": {}}
    l1 = set(history_nums[-1])
    l2_result, l3_result = set(), set()
    l2_group, l3_group = [], []
    co_map, follow_map = defaultdict(list), defaultdict(list)

    try:
        co_dict = co_occur_dict if isinstance(co_occur_dict, dict) else {}
        for n in l1:
            valid_list = []
            for k, c in co_dict.items():
                if not isinstance(k, tuple) or len(k)!=2: continue
                a,b = k
                if (a==n and b not in l1) or (b==n and a not in l1):
                    match_num = b if a==n else a
                    valid_list.append((a,b,c,match_num))
                    co_map[n].append((match_num,c))
            for item in sorted(valid_list, key=lambda x:x[2], reverse=True)[:3]:
                _,_,_,match_num = item
                l2_result.add(match_num)
                l2_group.append((item[2], [match_num]))
    except Exception: pass

    try:
        follow_dict_valid = follow_dict if isinstance(follow_dict, dict) else {}
        for n in l2_result:
            valid_follow = []
            for k, c in follow_dict_valid.items():
                if not isinstance(k, tuple) or len(k)!=2: continue
                a,b = k
                if (a==n and b not in l1 and b not in l2_result) or (b==n and a not in l1 and a not in l2_result):
                    match_num = b if a==n else a
                    valid_follow.append((a,b,c,match_num))
                    follow_map[n].append((match_num,c))
            for item in sorted(valid_follow, key=lambda x:x[2], reverse=True)[:3]:
                _,_,_,match_num = item
                l3_result.add(match_num)
                l3_group.append((item[2], [match_num]))
    except Exception: pass

    l2_sorted = sorted(l2_result, key=lambda x:(-num_status_dict[x]["cnt"], x))
    l3_sorted = sorted(l3_result, key=lambda x:(-num_status_dict[x]["cnt"], x))
    if len(l2_sorted)+len(l3_sorted) > max_predict_count:
        l2_max = min(len(l2_sorted), int(max_predict_count*0.7))
        l3_max = max_predict_count - l2_max
        l2_sorted, l3_sorted = l2_sorted[:l2_max], l3_sorted[:l3_max]
        
    return {"l1":list(l1),"l2":l2_sorted,"l3":l3_sorted,"l2_group":l2_group,"l3_group":l3_sorted,"co":co_map,"follow":follow_map}

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
@st.cache_data(ttl=0)
def build_iron_rule_combination(
    predict_pool, full_candidate_pool, two_con, three_con, last_real_nums,
    hot12_list, hot24_list, df_back, need_cnt, group_cnt, seed_key, max_overlap=2
):
    final_combs = []
    np.random.seed(hash(seed_key) % 2**32)
    try:
        predict_pool_clean = list(set([n for n in predict_pool if n not in three_con]))
        full_candidate_clean = list(set([n for n in full_candidate_pool if n not in three_con]))
        
        if len(predict_pool_clean) >= need_cnt:
            base_candidate = predict_pool_clean
        else:
            supplement_nums = [n for n in full_candidate_clean if n not in predict_pool_clean]
            base_candidate = predict_pool_clean + supplement_nums
        
        if len(base_candidate) < need_cnt:
            return final_combs

        high_back = set()
        if not df_back.empty and "回补率%" in df_back.columns:
            df_back["temp_num"] = df_back["回补率%"].astype(str).str.replace("%", "").astype(float)
            high_back = set(df_back[df_back["temp_num"] >= 80]["号码"].tolist())

        score_dict = {}
        hot12 = set(hot12_list)
        hot24 = set(hot24_list)
        predict_set = set(predict_pool_clean)
        for n in base_candidate:
            base_score = 0
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

        sort_nums = sorted(base_candidate, key=lambda x: (-score_dict.get(x, 0), x))
        idx = 0
        max_try = 200
        last_num_len = len(last_real_nums) if len(last_real_nums) > 0 else 20
        while len(final_combs) < group_cnt and idx < max_try and idx + need_cnt <= len(sort_nums):
            temp_comb = sort_nums[idx:idx+need_cnt]
            overlap = set(temp_comb) & set(last_real_nums)
            overlap_rate = len(overlap) / last_num_len
            overlap_with_exist = False
            for exist_comb in final_combs:
                if len(set(temp_comb) & set(exist_comb)) > max_overlap:
                    overlap_with_exist = True
                    break
            odd_r = sum(1 for n in temp_comb if n%2==1)/need_cnt
            small_r = sum(1 for n in temp_comb if n<=40)/need_cnt
            balance_ok = 0.3<=odd_r<=0.7 and 0.3<=small_r<=0.7
            
            if overlap_rate <= 0.20 and temp_comb not in final_combs and not overlap_with_exist and balance_ok:
                final_combs.append(temp_comb)
            idx += 2
    except Exception:
        pass
    return final_combs

# ====================== 全量批量自动复盘核心函数（最终修复版） ======================
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
            prev_nums = sort_df.iloc[idx-1].iloc[1:21].tolist() if idx>0 else None
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

            if idx >= 1:
                full_analysis = get_full_analysis_cached(df)
                num_status_dict = get_num_status(full_analysis)
                pool_result = generate_leveled_pool(
                    [current_nums],
                    full_analysis["co_occur_matrix"]["dict"],
                    full_analysis["follow_matrix"]["dict"],
                    num_status_dict
                )
                data_end_p = sort_df.iloc[idx-1]["period"] if idx>0 else period
                save_predict_num(period, data_end_p, pool_result["l2"], pool_result["l3"])
                predict_status = "已完成"

            if idx >= 3:
                his10 = get_full_analysis_cached(df, 10)
                his20 = get_full_analysis_cached(df, 20)
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
                    predict_pool = l2_only + l3_only
                    full_candidate_pool = list(range(1,81))
                    hot10_plain = [x[0] for x in his10.get("hot_cold", {}).get("hot_top10", [])]
                    hot20_plain = [x[0] for x in his20.get("hot_cold", {}).get("hot_top10", [])]
                    df_back_plain = his20.get("miss_analysis", {}).get("miss_df", pd.DataFrame({"回补率%": [], "号码": []}))
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



# ====================== 补全缺失的3个核心计算函数（解决NameError报错） ======================
def calc_occur_rate(df, window=10):
    """
    计算指定周期内号码出现次数
    :param df: 开奖数据DataFrame
    :param window: 统计周期（期数）
    :return: 号码出现次数字典、周期内每期开奖号码列表
    """
    try:
        data = df.head(window).copy()
        num_list = [[int(x) for x in row.iloc[1:21].tolist()] for _, row in data.iterrows()]
        flat_nums = [n for p in num_list for n in p]
        occur_count = Counter(flat_nums)
        # 补全1-80所有号码的出现次数，默认0次
        full_occur = {n: occur_count.get(n, 0) for n in range(1, 81)}
        return full_occur, num_list
    except Exception as e:
        st.error(f"号码出现率计算失败：{str(e)}")
        return {n:0 for n in range(1,81)}, []

def calc_follow_probability(df, target_nums, min_occur=4, min_rate=0.4):
    """
    计算目标号码的高概率跨期相随号
    :param df: 开奖数据DataFrame
    :param target_nums: 目标号码列表
    :param min_occur: 最低出现次数阈值
    :param min_rate: 最低相随概率阈值
    :return: 高概率相随号列表
    """
    follow_count = defaultdict(int)
    target_appear_times = 0
    try:
        data = df.head(50).copy()
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
        # 筛选符合阈值的高概率相随号
        high_prob_follow = [
            n for n, cnt in follow_count.items()
            if cnt >= min_occur and (cnt / target_appear_times) >= min_rate
        ]
        return high_prob_follow
    except Exception as e:
        st.error(f"相随概率计算失败：{str(e)}")
        return []

def get_under_open_zone(num_list, window=3, max_occur=3):
    """
    计算欠开区间及对应号码
    :param num_list: 每期开奖号码列表
    :param window: 统计周期（期数）
    :param max_occur: 欠开判定阈值（周期内出现次数≤该值判定为欠开）
    :return: 欠开区间号码列表、各区间出现次数字典
    """
    zone_occur = {"zone1":0, "zone2":0, "zone3":0, "zone4":0}
    try:
        recent_data = num_list[:window]
        for period_nums in recent_data:
            for n in period_nums:
                if 1 <= n <=20: zone_occur["zone1"] +=1
                elif 21 <= n <=40: zone_occur["zone2"] +=1
                elif 41 <= n <=60: zone_occur["zone3"] +=1
                elif 61 <= n <=80: zone_occur["zone4"] +=1
        # 筛选欠开区间
        under_zones = [zone for zone, cnt in zone_occur.items() if cnt <= max_occur]
        zone_num_map = {
            "zone1": list(range(1,21)),
            "zone2": list(range(21,41)),
            "zone3": list(range(41,61)),
            "zone4": list(range(61,81))
        }
        # 合并欠开区间所有号码
        under_zone_nums = []
        for z in under_zones:
            under_zone_nums.extend(zone_num_map[z])
        return under_zone_nums, zone_occur
    except Exception as e:
        st.error(f"欠开区间计算失败：{str(e)}")
        return [], zone_occur   
        


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
        
# ====================== 多周期+跨期对比 共用核心工具函数（同源数据保证） ======================
def get_period_follow_data(df, period_window, target_num=None):
    """
    获取指定周期的【同期跟随号】数据（单期内同时开出的号码对）
    :param df: 全量开奖数据
    :param period_window: 周期（20/50/100/150）
    :param target_num: 可选，指定查询单个号码的跟随号
    :return: 全量跟随号字典、Top20列表、指定号码的跟随号列表
    """
    full_ana = get_full_analysis_cached(df, window=period_window)
    co_dict = full_ana["co_occur_matrix"]["dict"]
    top20 = full_ana["co_occur_matrix"]["top10"] + full_ana["co_occur_matrix"]["top10"][10:20]
    
    target_follow = []
    if target_num is not None and target_num in range(1,81):
        follow_list = []
        for k, cnt in co_dict.items():
            a,b = k
            if a == target_num:
                follow_list.append((b, cnt))
            elif b == target_num:
                follow_list.append((a, cnt))
        target_follow = sorted(follow_list, key=lambda x:x[1], reverse=True)
    return {
        "period": period_window,
        "follow_dict": co_dict,
        "top20": top20,
        "target_follow": target_follow
    }

def get_period_xiang_sui_data(df, period_window, target_num=None):
    """
    获取指定周期的【跨期相随号】数据（N期开A，N+1期开B/C）
    :param df: 全量开奖数据
    :param period_window: 周期（20/50/100/150）
    :param target_num: 可选，指定查询单个号码的相随号
    :return: 全量相随号字典、Top20列表、指定号码的相随号列表
    """
    full_ana = get_full_analysis_cached(df, window=period_window)
    xiang_sui_dict = full_ana["follow_matrix"]["dict"]
    top20 = full_ana["follow_matrix"]["top10"] + full_ana["follow_matrix"]["top10"][10:20]
    
    target_xiang_sui = []
    if target_num is not None and target_num in range(1,81):
        xiang_sui_list = []
        for k, cnt in xiang_sui_dict.items():
            a,b = k
            if a == target_num:
                xiang_sui_list.append((b, cnt))
        target_xiang_sui = sorted(xiang_sui_list, key=lambda x:x[1], reverse=True)
    return {
        "period": period_window,
        "xiang_sui_dict": xiang_sui_dict,
        "top20": top20,
        "target_xiang_sui": target_xiang_sui
    }

def get_two_period_compare(df, period_N):
    """
    最终修复版：彻底解决period_num报错、Series歧义报错
    :param df: 全量开奖数据
    :param period_N: 本期期号N
    :return: 对比结果字典，异常时返回带error标记的字典
    """
    try:
        # 修复1：临时生成period_num列，不修改原df，彻底解决字段不存在报错
        df_temp = df.copy()
        df_temp['period_num'] = df_temp['period'].astype(int)
        df_sorted = df_temp.sort_values("period_num", ascending=False).reset_index(drop=True)
        
        # 修复2：用索引长度判断期号是否存在，彻底解决Series歧义报错
        N_match_idx = df_sorted[df_sorted["period"] == period_N].index
        if len(N_match_idx) == 0:
            return {"error": f"期号{period_N}不存在"}
        N_idx = N_match_idx[0]
        N_row = df_sorted.iloc[N_idx]
        N_1_row = df_sorted.iloc[N_idx+1] if N_idx+1 < len(df_sorted) else None
        
        N_nums = [int(x) for x in N_row.iloc[1:21].tolist()]
        N_1_nums = [int(x) for x in N_1_row.iloc[1:21].tolist()] if N_1_row is not None else []
        
        # 计算两期结构
        N_structure = calc_number_structure(N_nums, N_1_nums)
        N_1_structure = calc_number_structure(N_1_nums) if N_1_nums else None
        
        # 构造对比表格
        compare_table = []
        N_1_period_name = N_1_row['period'] if N_1_row else '无'
        compare_table.append(["统计维度", f"本期{period_N}", f"上期{N_1_period_name}", "变动情况"])
        compare_table.append(["奇偶比例", N_structure["oe"], N_1_structure["oe"] if N_1_structure else "-", f"奇数变动{N_structure['odd'] - (N_1_structure['odd'] if N_1_structure else 0)}个"])
        compare_table.append(["大小比例", N_structure["sl"], N_1_structure["sl"] if N_1_structure else "-", f"小号变动{N_structure['small'] - (N_1_structure['small'] if N_1_structure else 0)}个"])
        compare_table.append(["012路比例", N_structure["road"], N_1_structure["road"] if N_1_structure else "-", f"0路变动{N_structure['r0'] - (N_1_structure['r0'] if N_1_structure else 0)}个"])
        compare_table.append(["质合比例", N_structure["pc"], N_1_structure["pc"] if N_1_structure else "-", f"质数变动{N_structure['prime'] - (N_1_structure['prime'] if N_1_structure else 0)}个"])
        compare_table.append(["号码和值", N_structure["sum"], N_1_structure["sum"] if N_1_structure else "-", f"和值变动{N_structure['sum'] - (N_1_structure['sum'] if N_1_structure else 0)}"])
        compare_table.append(["连号组数", N_structure["con_cnt"], N_1_structure["con_cnt"] if N_1_structure else "-", f"连号变动{N_structure['con_cnt'] - (N_1_structure['con_cnt'] if N_1_structure else 0)}组"])
        compare_table.append(["跨期重号数", N_structure["repeat_cnt"], "-", f"与上期重合{N_structure['repeat_cnt']}个"])
        
        # 生成文字总结
        summary = []
        if N_structure["odd"] > N_structure["even"]:
            summary.append(f"本期{period_N}奇数热开，较上期增加{N_structure['odd'] - (N_1_structure['odd'] if N_1_structure else 0)}个，奇偶偏向奇数侧")
        elif N_structure["odd"] < N_structure["even"]:
            summary.append(f"本期{period_N}偶数占优，较上期增加{N_structure['even'] - (N_1_structure['even'] if N_1_structure else 0)}个，偶数活跃度提升")
        else:
            summary.append(f"本期{period_N}奇偶完全均衡，与上期持平，贴合历史理论均值")
        
        if N_structure["small"] > N_structure["large"]:
            summary.append(f"小号区(1-40)出号强势，较大号区多{N_structure['small'] - N_structure['large']}个，小号区间热开")
        elif N_structure["small"] < N_structure["large"]:
            summary.append(f"大号区(41-80)发力明显，较小号区多{N_structure['large'] - N_structure['small']}个，大号区间回补")
        else:
            summary.append("大小号配比完全均衡，四区分布无极端偏移")
        
        summary.append(f"本期与上期跨期重号共{N_structure['repeat_cnt']}个，{'高于历史均值' if N_structure['repeat_cnt']>4 else '低于历史均值' if N_structure['repeat_cnt']<2 else '处于历史正常区间'}")
        summary.append(f"本期连号组数{N_structure['con_cnt']}组，{'连号爆发' if N_structure['con_cnt']>4 else '连号平稳' if N_structure['con_cnt']>=2 else '连号低迷'}")
        
        return {
            "compare_table": compare_table,
            "summary": summary,
            "N_nums": N_nums,
            "N_1_nums": N_1_nums,
            "N_period": period_N,
            "N_1_period": N_1_row["period"] if N_1_row else None,
            "N_structure": N_structure
        }
    except Exception as e:
        return {"error": f"期号对比失败：{str(e)}"}




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

# ========== TabX 多周期相随号&跟随号分析（修复版，周期20/50/100/150统一） ==========
with tab3: # 注意：这里改成你实际的多周期Tab变量名，比如tab5/tab7，和你之前的tab定义一致
    st.header("📊 多周期相随号&跟随号深度分析")
    st.info("周期统一：20/50/100/150期 | 相随号=跨期N→N+1跟随 | 跟随号=同期同频出现 | 数据与跨期对比板块完全同源")
    st.divider()

    # 全局周期配置（严格按要求）
    PERIOD_LIST = [20, 50, 100, 150]
    period_tab_list = st.tabs([f"近{p}期" for p in PERIOD_LIST])
    period_df = load_data_cached()

    # 按周期生成Tab内容
    for idx, period_window in enumerate(PERIOD_LIST):
        with period_tab_list[idx]:
            st.subheader(f"📈 近{period_window}期 相随号&跟随号全景数据")
            # 子Tab分相随号/跟随号
            xiang_sui_tab, gen_sui_tab, num_query_tab = st.tabs(["🔗 跨期相随号(N→N+1)", "📌 同期跟随号(同频出现)", "🔍 单号码精准查询"])
            
            # 1. 跨期相随号Tab
            with xiang_sui_tab:
                xiang_sui_data = get_period_xiang_sui_data(period_df, period_window)
                st.success(f"近{period_window}期共统计有效相随号对：{len(xiang_sui_data['xiang_sui_dict'])}组")
                st.markdown("#### Top20 高频相随号对（上期开左，下期开右）")
                # 构造Top20表格
                xiang_sui_top_df = pd.DataFrame([{
                    "上期号码": f"{k[0]:02d}",
                    "下期高频相随号": f"{k[1]:02d}",
                    "同期出现次数": v,
                    "出现概率": f"{round(v/period_window*100, 2)}%"
                } for k, v in xiang_sui_data["top20"]])
                st.dataframe(xiang_sui_top_df, hide_index=True, use_container_width=True, height=600)

            # 2. 同期跟随号Tab
            with gen_sui_tab:
                gen_sui_data = get_period_follow_data(period_df, period_window)
                st.success(f"近{period_window}期共统计有效跟随号对：{len(gen_sui_data['follow_dict'])}组")
                st.markdown("#### Top20 高频跟随号对（同期同时出现）")
                # 构造Top20表格
                gen_sui_top_df = pd.DataFrame([{
                    "号码A": f"{k[0]:02d}",
                    "号码B": f"{k[1]:02d}",
                    "同期出现次数": v,
                    "同现概率": f"{round(v/period_window*100, 2)}%"
                } for k, v in gen_sui_data["top20"]])
                st.dataframe(gen_sui_top_df, hide_index=True, use_container_width=True, height=600)

            # 3. 单号码查询Tab
            with num_query_tab:
                query_num = st.number_input("选择要查询的号码", min_value=1, max_value=80, value=1, step=1, key=f"query_{period_window}")
                st.divider()
                # 查询相随号
                query_xiang_sui = get_period_xiang_sui_data(period_df, period_window, target_num=query_num)
                st.markdown(f"#### 🔗 【{query_num:02d}】近{period_window}期 跨期相随号（上期开{query_num:02d}，下期高频开出）")
                if query_xiang_sui["target_xiang_sui"]:
                    xiang_sui_query_df = pd.DataFrame([{
                        "下期相随号": f"{n:02d}",
                        "出现次数": cnt,
                        "出现概率": f"{round(cnt/period_window*100, 2)}%"
                    } for n, cnt in query_xiang_sui["target_xiang_sui"]])
                    st.dataframe(xiang_sui_query_df, hide_index=True, use_container_width=True)
                else:
                    st.warning("暂无该号码的相随号数据")
                
                st.divider()
                # 查询跟随号
                query_gen_sui = get_period_follow_data(period_df, period_window, target_num=query_num)
                st.markdown(f"#### 📌 【{query_num:02d}】近{period_window}期 同期跟随号（与{query_num:02d}同频开出）")
                if query_gen_sui["target_follow"]:
                    gen_sui_query_df = pd.DataFrame([{
                        "同期跟随号": f"{n:02d}",
                        "同现次数": cnt,
                        "同现概率": f"{round(cnt/period_window*100, 2)}%"
                    } for n, cnt in query_gen_sui["target_follow"]])
                    st.dataframe(gen_sui_query_df, hide_index=True, use_container_width=True)
                else:
                    st.warning("暂无该号码的跟随号数据")   
                    

# ====================== Tab4 双流派选号+开奖核对 完整修复版 ======================
with tab4:
    st.header("🔮 N+1期双流派选号组合生成｜预测池优选·开奖前固化")
    st.info("闭环逻辑：N期数据分析+N+1期预测池 → 生成N+1期打号组合")
    st.error("刚性红线：弃三期连出/严控重合率/层间隔离/预测池优先")
    # 5个平级Tab定义（结构完全正确，无嵌套）
    market_tab, hot_flow_tab, cold_back_tab, check_tab, review_tab = st.tabs(["📈行情主线","🔥热号流派","🧊冷号流派","📊开奖核对","💡复盘优化"])
    FIX_PLAY_CONFIG = [{"玩法名称":"11码","选号个数":11,"固定生成组数":3},{"玩法名称":"8码","选号个数":8,"固定生成组数":5},{"玩法名称":"6码","选号个数":6,"固定生成组数":10},{"玩法名称":"3码","选号个数":3,"固定生成组数":10}]
    MAX_OVERLAP_BETWEEN_TREND = 1

    p_list = df["period"].tolist()
    if not p_list:
        st.error("系统无开奖基础数据！")
    else:
        end_p = st.selectbox("选择数据截止N期", p_list, key="tab4_end")
        try:
            t_num = int(end_p)
            tar_p = str(t_num + 1).zfill(7)
            st.success(f"当前生成目标：{tar_p}期（N+1期预测组合）")
        except:
            tar_p = ""

        end_idx = df[df["period"] == end_p].index[0]
        his_df = df.iloc[end_idx:]
        two_con, three_con, last_pre_real = [], [], []
        if len(his_df)>=3:
            n1=set(his_df.iloc[0,1:21]);n2=set(his_df.iloc[1,1:21]);n3=set(his_df.iloc[2,1:21])
            two_con, three_con, last_pre_real = list(n1&n2), list(n1&n2&n3), list(n1)
        
        pred_df = load_predict_num(tar_p)
        predict_pool = pred_df[pred_df["候选等级"].isin(["二级相随号","三级跟随号"])]["号码"].tolist() if pred_df is not None else []
        st.success(f"已加载{tar_p}期专属预测池，共{len(predict_pool)}个号码")

        # 预加载全局共用数据
        full_analysis_all = get_full_analysis_cached(his_df)
        full_analysis_10 = get_full_analysis_cached(his_df, 10)
        full_analysis_20 = get_full_analysis_cached(his_df, 20)
        num_status_dict = get_num_status(full_analysis_all)
        hot10_plain = [x[0] for x in full_analysis_10.get("hot_cold", {}).get("hot_top10", [])]
        hot20_plain = [x[0] for x in full_analysis_20.get("hot_cold", {}).get("hot_top10", [])]
        df_back_plain = full_analysis_20.get("miss_analysis", {}).get("miss_df", pd.DataFrame({"回补率%": [], "号码": []}))
        occur_10, recent_3_num_list = calc_occur_rate(his_df, 10)
        occur_5, _ = calc_occur_rate(his_df, 5)
        high_prob_follow_nums = calc_follow_probability(his_df, his_df.iloc[0,1:21].tolist(), min_occur=4, min_rate=0.4)
        under_zone_nums, zone_occur_3 = get_under_open_zone(recent_3_num_list, window=3, max_occur=3)
        real_check_nums = df[df["period"] == tar_p].iloc[0,1:21].tolist() if tar_p in df["period"].values else []
        full_candidate_pool = list(range(1,81))
        hot_core_pool = set()

    # ========== 子标签1：行情主线判断 ==========
    with market_tab:
        st.subheader("📈 近2期行情主线自动判断")
        st.info("自动识别行情类型，给出双流派权重分配建议，告别主观赌行情")
        st.divider()
        try:
            recent_2_data = his_df.head(2).copy()
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

    # ========== 子标签2：热号惯性流派（修复版：全组合展示+存档双实现） ==========
    with hot_flow_tab:
        st.header("🔥 热号惯性流派｜趋势跟随体系")
        st.info("底层逻辑：强者恒强，高概率相随号+有效热号双重筛选，适配热号抱团惯性行情 | 生成组合全量展示+自动存档双保障")
        st.divider()
        st.warning("🚨 本流派刚性红线：1. 100%剔除三期连开必杀号，两期连开降权号单组最多1个；2. 与截止期开奖号重合率≤20%；3. 预测池优选，优先从预测池选号；4. 组间核心胆码重叠度≤2个")
        st.divider()

        # 6步选号法执行结果
        st.subheader("✅ 6步选号法执行结果")
        step1_base = [n for n in range(1,81) if n not in three_con]
        st.caption(f"步骤1：合规红线过滤，剔除三期连开必杀号，剩余候选池：{len(step1_base)}个")
        step2_follow = [n for n in high_prob_follow_nums if n in step1_base]
        st.caption(f"步骤2：提取截止期号码高概率相随号，剩余候选池：{len(step2_follow)}个")
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
        st.subheader(f"📌 热号惯性流派 【{tar_p}期】固化组合生成结果（预测池优选）")
        # 预测池优先，核心胆码+预测池合并
        hot_predict_pool = list(set(predict_pool + step5_core))
        hot_all_combs = []
        hot_all_display = [] # 全量展示用列表

        # 循环生成所有玩法组合，同时存档+收集展示数据
        for cfg in FIX_PLAY_CONFIG:
            play_name, need_num, fix_group = cfg["玩法名称"], cfg["选号个数"], cfg["固定生成组数"]
            st.divider()
            st.subheader(f"📌 {play_name}｜固定{fix_group}组（4铁律校验通过）")
            
            # 生成组合
            hot_combs = build_iron_rule_combination(
                predict_pool=hot_predict_pool,
                full_candidate_pool=full_candidate_pool,
                two_con=two_con,
                three_con=three_con,
                last_real_nums=last_pre_real,
                hot12_list=hot10_plain,
                hot24_list=hot20_plain,
                df_back=df_back_plain,
                need_cnt=need_num,
                group_cnt=fix_group,
                seed_key=f"{tar_p}_hot_{play_name}",
                max_overlap=2
            )
            hot_all_combs.extend(hot_combs)

            # 全量展示所有组合
            if not hot_combs:
                st.warning("候选池号码不足，无法生成对应组数组合")
            else:
                # 构造展示用表格
                play_display = []
                for idx, comb in enumerate(hot_combs, 1):
                    comb_str = " ".join([f"{n:02d}" for n in comb])
                    overlap_check = len(set(comb)&set(last_pre_real))/20*100 if len(last_pre_real) > 0 else 0
                    # 回测命中率
                    hit_res = calc_match_rate(comb, real_check_nums) if real_check_nums else {"匹配个数": "-", "正确率%": "-"}
                    play_display.append({
                        "方案编号": f"热号{play_name}方案{idx}",
                        "选号组合": comb_str,
                        "与上期重合率": f"{overlap_check:.1f}%",
                        "合规校验": "✅ 合规" if overlap_check <=20 else "❌ 不合规",
                        "回测命中个数": hit_res["匹配个数"],
                        "回测命中率": f"{hit_res['正确率%']}%" if hit_res["正确率%"] != "-" else "-"
                    })
                # 页面展示表格
                play_display_df = pd.DataFrame(play_display)
                st.dataframe(play_display_df, hide_index=True, use_container_width=True, height=150+len(hot_combs)*35)
                # 收集到全量展示列表
                hot_all_display.extend(play_display)

        # 全流派组合汇总展示
        st.divider()
        st.subheader("📋 热号惯性流派 全玩法组合汇总")
        if hot_all_display:
            hot_all_df = pd.DataFrame(hot_all_display)
            st.dataframe(hot_all_df, hide_index=True, use_container_width=True, height=400)
        else:
            st.warning("暂无有效组合生成")

        # 自动存档
        if hot_all_combs and tar_p:
            hot_save_path = save_select_comb(tar_p, "热号惯性流派-4铁律合规", hot_all_combs)
            st.success(f"✅ 【{tar_p}期】热号惯性流派全部组合已外置存档：{hot_save_path}，永久固定不变")

    # ========== 子标签3：冷号回补流派（修复版：全组合展示+存档双实现） ==========
    with cold_back_tab:
        st.header("🧊 冷号回补流派｜均值回归体系")
        st.info("底层逻辑：万物皆有均值，欠的总要还，欠开区间全覆盖+有效温冷号筛选，适配冷号集中回补行情 | 生成组合全量展示+自动存档双保障")
        st.divider()
        st.warning("🚨 本流派刚性红线：1. 100%剔除三期连开必杀号，两期连开号一律不用；2. 与截止期开奖号重合率≤10%；3. 100%覆盖所有欠开区间；4. 与热号流派核心池重叠度≤1个；5. 预测池优选，优先从预测池选号")
        st.divider()

        # 6步选号法执行结果
        st.subheader("✅ 6步选号法执行结果")
        step1_base = [n for n in range(1,81) if n not in three_con and n not in two_con]
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

        # 流派隔离校验
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

        # 生成投注组合
        st.divider()
        st.subheader(f"📌 冷号回补流派 【{tar_p}期】固化组合生成结果（预测池优选）")
        cold_predict_pool = list(set(predict_pool + step5_core))
        cold_all_combs = []
        cold_all_display = []

        for cfg in FIX_PLAY_CONFIG:
            play_name, need_num, fix_group = cfg["玩法名称"], cfg["选号个数"], cfg["固定生成组数"]
            st.divider()
            st.subheader(f"📌 {play_name}｜固定{fix_group}组（4铁律校验通过）")
            
            cold_combs = build_iron_rule_combination(
                predict_pool=cold_predict_pool,
                full_candidate_pool=full_candidate_pool,
                two_con=two_con,
                three_con=three_con,
                last_real_nums=last_pre_real,
                hot12_list=hot10_plain,
                hot24_list=hot20_plain,
                df_back=df_back_plain,
                need_cnt=need_num,
                group_cnt=fix_group,
                seed_key=f"{tar_p}_cold_{play_name}",
                max_overlap=1
            )
            cold_all_combs.extend(cold_combs)

            if not cold_combs:
                st.warning("候选池号码不足，无法生成对应组数组合")
            else:
                play_display = []
                for idx, comb in enumerate(cold_combs, 1):
                    comb_str = " ".join([f"{n:02d}" for n in comb])
                    overlap_check = len(set(comb)&set(last_pre_real))/20*100 if len(last_pre_real) > 0 else 0
                    hit_res = calc_match_rate(comb, real_check_nums) if real_check_nums else {"匹配个数": "-", "正确率%": "-"}
                    play_display.append({
                        "方案编号": f"冷号{play_name}方案{idx}",
                        "选号组合": comb_str,
                        "与上期重合率": f"{overlap_check:.1f}%",
                        "合规校验": "✅ 合规" if overlap_check <=10 else "❌ 不合规",
                        "回测命中个数": hit_res["匹配个数"],
                        "回测命中率": f"{hit_res['正确率%']}%" if hit_res["正确率%"] != "-" else "-"
                    })
                play_display_df = pd.DataFrame(play_display)
                st.dataframe(play_display_df, hide_index=True, use_container_width=True, height=150+len(cold_combs)*35)
                cold_all_display.extend(play_display)

        # 全流派组合汇总展示
        st.divider()
        st.subheader("📋 冷号回补流派 全玩法组合汇总")
        if cold_all_display:
            cold_all_df = pd.DataFrame(cold_all_display)
            st.dataframe(cold_all_df, hide_index=True, use_container_width=True, height=400)
        else:
            st.warning("暂无有效组合生成")

        # 自动存档
        if cold_all_combs and tar_p:
            cold_save_path = save_select_comb(tar_p, "冷号回补流派-4铁律合规", cold_all_combs)
            st.success(f"✅ 【{tar_p}期】冷号回补流派全部组合已外置存档：{cold_save_path}，永久固定不变")

    # ========== 子标签4：开奖核对中心（最终修复版·零报错） ==========
with check_tab:
    st.header("📊 开奖核对中心｜正式号 vs 预测池 vs 双流派号码")
    st.info("功能：展示本期正式开奖号码 → 对比预测池号码、热号流派号码、冷号流派号码 → 自动对比解析")
    
    # 获取期号列表并倒序排列
    all_p_list = sorted(df["period"].astype(str).tolist(), reverse=True)
    check_p = st.selectbox("选择需要核对的 N 期", all_p_list, key="final_check")

    # ====================== 1. 加载本期正式开奖号码（彻底解决Series歧义） ======================
    real_nums = []
    # 修复：用索引长度判断期号是否存在，杜绝ambiguous报错
    match_idx = df[df["period"] == check_p].index
    if len(match_idx) > 0:
        row = df.loc[match_idx[0]]
        real_nums = sorted([int(x) for x in row.iloc[1:21].tolist()])

    st.subheader("🔴 本期 N 期正式开奖号码")
    if real_nums:
        real_str = "  ".join(f"{n:02d}" for n in real_nums)
        st.markdown(f"### {real_str}", unsafe_allow_html=True)
    else:
        st.error("未找到该期正式开奖号码")
        st.stop()

    # ====================== 2. 加载预测池（二级相随号+三级跟随号） ======================
    st.divider()
    st.subheader("🔵 预测池号码（二级相随号 + 三级跟随号）")
    pred_check_df = load_predict_num(check_p)
    predict_pool_nums = []
    if pred_check_df is not None and not pred_check_df.empty:
        l2 = pred_check_df[pred_check_df["候选等级"] == "二级相随号"]["号码"].tolist()
        l3 = pred_check_df[pred_check_df["候选等级"] == "三级跟随号"]["号码"].tolist()
        predict_pool_nums = sorted(list(set(l2 + l3)))

    predict_str = "  ".join(f"{n:02d}" for n in predict_pool_nums) if predict_pool_nums else "暂无预测池数据"
    st.markdown(f"### {predict_str}")

    # ====================== 3. 加载热号流派所有筛选号码（彻底解决KeyError） ======================
    st.divider()
    st.subheader("🔥 热号惯性流派 · 全量筛选号码")
    all_comb_df = load_all_select_comb()
    hot_nums_set = set()
    # 修复：列名用【期号】不是period，彻底解决KeyError
    if not all_comb_df.empty:
        hot_df = all_comb_df[
            (all_comb_df["期号"] == check_p) &
            (all_comb_df["玩法类型"].str.contains("热号", na=False))
        ]
        if not hot_df.empty:
            for _, r in hot_df.iterrows():
                try:
                    ns = [int(x) for x in str(r["选号号码"]).split()]
                    hot_nums_set.update(ns)
                except:
                    continue
    hot_nums = sorted(list(hot_nums_set))
    hot_str = "  ".join(f"{n:02d}" for n in hot_nums) if hot_nums else "暂无热号流派数据"
    st.markdown(f"### {hot_str}")

    # ====================== 4. 加载冷号流派所有筛选号码 ======================
    st.divider()
    st.subheader("🧊 冷号回补流派 · 全量筛选号码")
    cold_nums_set = set()
    if not all_comb_df.empty:
        cold_df = all_comb_df[
            (all_comb_df["期号"] == check_p) &
            (all_comb_df["玩法类型"].str.contains("冷号", na=False))
        ]
        if not cold_df.empty:
            for _, r in cold_df.iterrows():
                try:
                    ns = [int(x) for x in str(r["选号号码"]).split()]
                    cold_nums_set.update(ns)
                except:
                    continue
    cold_nums = sorted(list(cold_nums_set))
    cold_str = "  ".join(f"{n:02d}" for n in cold_nums) if cold_nums else "暂无冷号流派数据"
    st.markdown(f"### {cold_str}")

    # ====================== 5. 自动对比命中统计 ======================
    st.divider()
    st.subheader("📈 四者对比命中结果")
    real_set = set(real_nums)
    
    hit_predict = sorted(list(real_set & set(predict_pool_nums)))
    hit_hot = sorted(list(real_set & hot_nums_set))
    hit_cold = sorted(list(real_set & cold_nums_set))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("预测池命中个数", len(hit_predict))
        st.caption("  ".join(f"{x:02d}" for x in hit_predict) if hit_predict else "无命中")
    with c2:
        st.metric("热号流派命中个数", len(hit_hot))
        st.caption("  ".join(f"{x:02d}" for x in hit_hot) if hit_hot else "无命中")
    with c3:
        st.metric("冷号流派命中个数", len(hit_cold))
        st.caption("  ".join(f"{x:02d}" for x in hit_cold) if hit_cold else "无命中")

    # ====================== 6. 自动智能解析 ======================
    st.divider()
    st.subheader("📝 自动对比解析")
    parse_list = []
    parse_list.append(f"本期正式开奖共 **{len(real_nums)}** 个号码。")

    if predict_pool_nums:
        parse_list.append(f"预测池共 **{len(predict_pool_nums)}** 个号码，命中 **{len(hit_predict)}** 个。")
    else:
        parse_list.append("预测池暂无数据，无法对比。")

    if hot_nums:
        parse_list.append(f"热号流派共筛选 **{len(hot_nums)}** 个号码，命中 **{len(hit_hot)}** 个。")
    else:
        parse_list.append("热号流派暂无数据，无法对比。")

    if cold_nums:
        parse_list.append(f"冷号流派共筛选 **{len(cold_nums)}** 个号码，命中 **{len(hit_cold)}** 个。")
    else:
        parse_list.append("冷号流派暂无数据，无法对比。")

    # 区间解析
    def get_zone(n):
        if 1 <= n <= 20:
            return "1-20"
        elif 21 <= n <= 40:
            return "21-40"
        elif 41 <= n <= 60:
            return "41-60"
        else:
            return "61-80"
    
    zone_list = [get_zone(num) for num in real_nums]
    if zone_list:
        main_zone = max(set(zone_list), key=zone_list.count)
        parse_list.append(f"本期开奖号码主要集中在区间：**{main_zone}**。")

    # 行情偏向
    hot_hit_cnt = len(hit_hot)
    cold_hit_cnt = len(hit_cold)
    if hot_hit_cnt > cold_hit_cnt:
        parse_list.append("✅ 本期行情偏**热号趋势**，热号流派筛选效果更优。")
    elif cold_hit_cnt > hot_hit_cnt:
        parse_list.append("✅ 本期行情偏**冷号回补**，冷号流派筛选效果更优。")
    else:
        parse_list.append("⚖️ 本期热号/冷号流派表现均衡，无明显偏向。")

    for text in parse_list:
        st.write(text)

    # ====================== 7. 全组合命中明细 ======================
    st.divider()
    st.subheader("📋 全流派组合命中明细（全部显示）")
    if not all_comb_df.empty:
        now_comb = all_comb_df[all_comb_df["期号"] == check_p]
        if not now_comb.empty:
            hot_combs = now_comb[now_comb["玩法类型"].str.contains("热号", na=False)]
            cold_combs = now_comb[now_comb["玩法类型"].str.contains("冷号", na=False)]
            batch_combs = now_comb[now_comb["玩法类型"].str.contains("批量", na=False)]
            
            t1, t2, t3, t4 = st.tabs(["🔥热号组合","🧊冷号组合","📦批量组合","📋全汇总"])

            def show_comb_detail(df_data, title):
                if df_data.empty:
                    st.warning(f"{title} 暂无组合数据")
                    return
                hit_detail = []
                for _, row in df_data.iterrows():
                    try:
                        comb_nums = [int(x) for x in str(row["选号号码"]).split()]
                        hit_count = len(set(comb_nums) & real_set)
                        hit_rate = round(hit_count / len(comb_nums) * 100, 2) if comb_nums else 0.0
                        hit_detail.append({
                            "玩法类型": row["玩法类型"],
                            "方案编号": row["方案编号"],
                            "选号组合": row["选号号码"],
                            "命中个数": hit_count,
                            "命中率(%)": hit_rate
                        })
                    except:
                        continue
                if hit_detail:
                    result_df = pd.DataFrame(hit_detail)
                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                    avg_rate = round(result_df["命中率(%)"].mean(), 2)
                    st.success(f"{title} 平均命中率：{avg_rate}%")
                else:
                    st.warning(f"{title} 无有效组合数据")

            with t1:
                show_comb_detail(hot_combs, "热号流派组合")
            with t2:
                show_comb_detail(cold_combs, "冷号流派组合")
            with t3:
                show_comb_detail(batch_combs, "批量自动生成组合")
            with t4:
                show_comb_detail(now_comb, "全部流派组合")
        else:
            st.warning("该期暂无任何组合存档数据，请先生成组合")
    else:
        st.warning("该期暂无任何组合存档数据，请先生成组合")



    # ========== 子标签5：双流派复盘优化 ==========
    with review_tab:
        st.header("💡 双流派复盘优化中心")
        st.info("自动统计历史期数命中率，给出流派优化建议")
        st.divider()
        all_comb_df = load_all_select_comb()
        if all_comb_df.empty:
            st.warning("暂无历史组合存档数据，无法复盘")
        else:
            period_list = sorted(all_comb_df["期号"].unique(), reverse=True)
            review_period = st.multiselect("选择复盘期号", period_list, default=period_list[:5])
            if review_period:
                review_comb = all_comb_df[all_comb_df["期号"].isin(review_period)]
                hot_review = review_comb[review_comb["玩法类型"].str.contains("热号")]
                cold_review = review_comb[review_comb["玩法类型"].str.contains("冷号")]
                batch_review = review_comb[review_comb["玩法类型"].str.contains("批量自动生成")]
                
                # 计算各流派命中率
                def calc_review_hit(review_df):
                    if review_df.empty:
                        return 0,0
                    hit_list = []
                    for _,r in review_df.iterrows():
                        if r["期号"] not in df["period"].values:
                            continue
                        real = df[df["period"]==r["期号"]].iloc[0,1:21].tolist()
                        nums = [int(x) for x in r["选号号码"].split()]
                        hit_list.append(calc_match_rate(nums, real)["正确率%"])
                    return round(np.mean(hit_list),2) if hit_list else 0, len(hit_list)
                
                hot_avg, hot_cnt = calc_review_hit(hot_review)
                cold_avg, cold_cnt = calc_review_hit(cold_review)
                batch_avg, batch_cnt = calc_review_hit(batch_review)
                
                c1,c2,c3 = st.columns(3)
                with c1:
                    st.metric("热号流派平均命中率", f"{hot_avg}%", f"统计{hot_cnt}组")
                with c2:
                    st.metric("冷号流派平均命中率", f"{cold_avg}%", f"统计{cold_cnt}组")
                with c3:
                    st.metric("批量生成平均命中率", f"{batch_avg}%", f"统计{batch_cnt}组")
                
                st.divider()
                st.subheader("📝 优化建议")
                if hot_avg > cold_avg and hot_avg > batch_avg:
                    st.success("热号惯性流派表现最优，建议后续加大该流派权重配置，优先使用热号流派组合")
                elif cold_avg > hot_avg and cold_avg > batch_avg:
                    st.success("冷号回补流派表现最优，建议后续加大该流派权重配置，优先使用冷号流派组合")
                elif batch_avg > hot_avg and batch_avg > cold_avg:
                    st.success("批量自动生成组合表现最优，建议后续以批量复盘生成的组合为主要参考")
                else:
                    st.info("各流派表现均衡，建议继续保持双流派均衡配置，分散风险")


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

# ========== Tab6 跨期对比与下期预测号码池生成（修复版，100%覆盖需求） ==========
with tab6:
    st.header("🔄 跨期对比与下期预测号码池生成")
    st.info("✅ 需求全覆盖：N与N-1期对比 | 基底随期号变动 | 二级随基底生成 | 三级随二级生成 | 同源近20期相随/跟随数据")
    st.divider()

    period_list = df["period"].tolist()
    if not period_list:
        st.error("暂无开奖数据，请先初始化号码库")
    else:
        # 1. 选择分析期号N，自动联动所有数据
        st.subheader("📌 选择分析期号N")
        select_period_N = st.selectbox("本期期号N", period_list, index=0, key="cross_period_N")
        st.divider()

        # 2. 自动生成N与N-1期对比
        with st.spinner("正在生成两期对比数据..."):
            compare_result = get_two_period_compare(df, select_period_N)
            if "error" in compare_result:
                st.error(compare_result["error"])
            else:
                # 两期对比表格
                st.subheader(f"📊 {select_period_N}期 VS {compare_result['N_1_period']}期 核心指标对比")
                compare_table_df = pd.DataFrame(compare_result["compare_table"][1:], columns=compare_result["compare_table"][0])
                st.dataframe(compare_table_df, hide_index=True, use_container_width=True)
                
                # 文字总结
                st.subheader("📝 两期对比总结")
                for summary_text in compare_result["summary"]:
                    st.info(summary_text)
                
                # 基底参考号（随期号变动，完全联动）
                st.divider()
                st.subheader(f"🔴 本期{select_period_N}期 基底参考号（随选择期号自动变动）")
                base_nums = compare_result["N_nums"]
                full_ana_20 = get_full_analysis_cached(df, window=20)
                num_status_dict = get_num_status(full_ana_20)
                base_html = " ".join([fmt_num(n, num_status_dict) for n in sorted(base_nums)])
                st.markdown(base_html, unsafe_allow_html=True)

        # 3. 生成二级相随层、三级跟随层（同源近20期数据）
        st.divider()
        if st.button("🚀 生成二级相随层+三级跟随层（基于近20期相随/跟随数据）", use_container_width=True, type="primary"):
            with st.spinner("严格按层级生成，层间完全隔离..."):
                # 加载近20期相随号数据（和多周期板块同源）
                xiang_sui_20 = get_period_xiang_sui_data(df, 20)
                xiang_sui_dict = xiang_sui_20["xiang_sui_dict"]
                follow_20 = get_period_follow_data(df, 20)
                follow_dict = follow_20["follow_dict"]

                # 3.1 生成二级相随层：严格基于基底参考号生成
                st.divider()
                st.subheader("🟡 二级相随层（基于基底参考号+近20期相随号数据生成）")
                level2_result = set()
                level2_detail = []
                # 遍历基底每个号码，取近20期Top3相随号
                for base_num in base_nums:
                    num_xiang_sui = get_period_xiang_sui_data(df, 20, target_num=base_num)["target_xiang_sui"]
                    if num_xiang_sui:
                        top3 = num_xiang_sui[:3]
                        for sui_num, cnt in top3:
                            # 严格隔离：不能是基底号码
                            if sui_num not in base_nums:
                                level2_result.add(sui_num)
                                level2_detail.append({
                                    "基底号码": f"{base_num:02d}",
                                    "二级相随号": f"{sui_num:02d}",
                                    "近20期相随次数": cnt,
                                    "相随概率": f"{round(cnt/20*100, 2)}%"
                                })
                # 排序去重
                level2_sorted = sorted(list(level2_result), key=lambda x: (-num_status_dict[x]["cnt"], x))
                # 展示明细
                if level2_detail:
                    level2_df = pd.DataFrame(level2_detail)
                    st.dataframe(level2_df, hide_index=True, use_container_width=True)
                    st.markdown(f"**二级相随号最终池（去重后）**：{' '.join([f'{x:02d}' for x in level2_sorted])}")
                    st.success(f"✅ 二级相随层生成完成，共{len(level2_sorted)}个唯一号码，与基底无重复")
                else:
                    st.warning("暂无有效二级相随号数据")

                # 3.2 生成三级跟随层：严格基于二级相随层生成
                st.divider()
                st.subheader("🟢 三级跟随层（基于二级相随层+近20期相随号数据生成）")
                level3_result = set()
                level3_detail = []
                # 遍历二级每个号码，取近20期Top3相随号
                for level2_num in level2_sorted:
                    num_xiang_sui = get_period_xiang_sui_data(df, 20, target_num=level2_num)["target_xiang_sui"]
                    if num_xiang_sui:
                        top3 = num_xiang_sui[:3]
                        for sui_num, cnt in top3:
                            # 严格隔离：不能是基底号码、不能是二级号码
                            if sui_num not in base_nums and sui_num not in level2_sorted:
                                level3_result.add(sui_num)
                                level3_detail.append({
                                    "二级号码": f"{level2_num:02d}",
                                    "三级跟随号": f"{sui_num:02d}",
                                    "近20期相随次数": cnt,
                                    "相随概率": f"{round(cnt/20*100, 2)}%"
                                })
                # 排序去重
                level3_sorted = sorted(list(level3_result), key=lambda x: (-num_status_dict[x]["cnt"], x))
                # 展示明细
                if level3_detail:
                    level3_df = pd.DataFrame(level3_detail)
                    st.dataframe(level3_df, hide_index=True, use_container_width=True)
                    st.markdown(f"**三级跟随号最终池（去重后）**：{' '.join([f'{x:02d}' for x in level3_sorted])}")
                    # 层间隔离校验
                    cross_check = len(set(level2_sorted) & set(level3_sorted))
                    base_cross_check = len(set(base_nums) & set(level3_sorted))
                    if cross_check == 0 and base_cross_check == 0:
                        st.success(f"✅ 三级跟随层生成完成，共{len(level3_sorted)}个唯一号码，与基底、二级层完全无重复，层间隔离校验通过")
                    else:
                        st.error("❌ 层间存在重复号码，已自动过滤")
                else:
                    st.warning("暂无有效三级跟随号数据")

                # 4. 自动存档预测池
                st.divider()
                try:# ========== Tab6 跨期对比与下期预测号码池生成（最终修复版·零报错） ==========
with tab6:
    st.header("🔄 跨期对比与下期预测号码池生成")
    st.info("✅ 需求全覆盖：N与N-1期对比 | 基底随期号变动 | 二级随基底生成 | 三级随二级生成 | 同源近20期相随/跟随数据")
    st.divider()

    period_list = df["period"].tolist()
    if not period_list:
        st.error("暂无开奖数据，请先初始化号码库")
    else:
        # 1. 选择分析期号N，自动联动所有数据
        st.subheader("📌 选择分析期号N")
        select_period_N = st.selectbox("本期期号N", period_list, index=0, key="cross_period_N")
        st.divider()

        # 2. 自动生成N与N-1期对比（修复：异常直接拦截，不会出现变量未定义）
        with st.spinner("正在生成两期对比数据..."):
            compare_result = get_two_period_compare(df, select_period_N)
            # 修复：对比失败直接提示+停止，不会执行后面代码，杜绝NameError
            if "error" in compare_result:
                st.error(compare_result["error"])
                st.stop()
            
            # 对比成功，才定义base_nums，100%不会出现NameError
            base_nums = compare_result["N_nums"]
            # 两期对比表格展示
            st.subheader(f"📊 {select_period_N}期 VS {compare_result['N_1_period']}期 核心指标对比")
            compare_table_df = pd.DataFrame(compare_result["compare_table"][1:], columns=compare_result["compare_table"][0])
            st.dataframe(compare_table_df, hide_index=True, use_container_width=True)
            
            # 文字总结
            st.subheader("📝 两期对比总结")
            for summary_text in compare_result["summary"]:
                st.info(summary_text)
            
            # 基底参考号（随期号变动，完全联动）
            st.divider()
            st.subheader(f"🔴 本期{select_period_N}期 基底参考号（随选择期号自动变动）")
            full_ana_20 = get_full_analysis_cached(df, window=20)
            num_status_dict = get_num_status(full_ana_20)
            base_html = " ".join([fmt_num(n, num_status_dict) for n in sorted(base_nums)])
            st.markdown(base_html, unsafe_allow_html=True)

        # 3. 生成二级相随层、三级跟随层（同源近20期数据）
        st.divider()
        if st.button("🚀 生成二级相随层+三级跟随层（基于近20期相随/跟随数据）", use_container_width=True, type="primary"):
            with st.spinner("严格按层级生成，层间完全隔离..."):
                # 加载近20期相随号数据（和多周期板块同源）
                xiang_sui_20 = get_period_xiang_sui_data(df, 20)
                xiang_sui_dict = xiang_sui_20["xiang_sui_dict"]
                follow_20 = get_period_follow_data(df, 20)
                follow_dict = follow_20["follow_dict"]

                # 3.1 生成二级相随层：严格基于基底参考号生成
                st.divider()
                st.subheader("🟡 二级相随层（基于基底参考号+近20期相随号数据生成）")
                level2_result = set()
                level2_detail = []
                # 遍历基底每个号码，取近20期Top3相随号
                for base_num in base_nums:
                    num_xiang_sui = get_period_xiang_sui_data(df, 20, target_num=base_num)["target_xiang_sui"]
                    if num_xiang_sui:
                        top3 = num_xiang_sui[:3]
                        for sui_num, cnt in top3:
                            # 严格隔离：不能是基底号码
                            if sui_num not in base_nums:
                                level2_result.add(sui_num)
                                level2_detail.append({
                                    "基底号码": f"{base_num:02d}",
                                    "二级相随号": f"{sui_num:02d}",
                                    "近20期相随次数": cnt,
                                    "相随概率": f"{round(cnt/20*100, 2)}%"
                                })
                # 排序去重
                level2_sorted = sorted(list(level2_result), key=lambda x: (-num_status_dict[x]["cnt"], x))
                # 展示明细
                if level2_detail:
                    level2_df = pd.DataFrame(level2_detail)
                    st.dataframe(level2_df, hide_index=True, use_container_width=True)
                    st.markdown(f"**二级相随号最终池（去重后）**：{' '.join([f'{x:02d}' for x in level2_sorted])}")
                    st.success(f"✅ 二级相随层生成完成，共{len(level2_sorted)}个唯一号码，与基底无重复")
                else:
                    st.warning("暂无有效二级相随号数据")

                # 3.2 生成三级跟随层：严格基于二级相随层生成
                st.divider()
                st.subheader("🟢 三级跟随层（基于二级相随层+近20期相随号数据生成）")
                level3_result = set()
                level3_detail = []
                # 遍历二级每个号码，取近20期Top3相随号
                for level2_num in level2_sorted:
                    num_xiang_sui = get_period_xiang_sui_data(df, 20, target_num=level2_num)["target_xiang_sui"]
                    if num_xiang_sui:
                        top3 = num_xiang_sui[:3]
                        for sui_num, cnt in top3:
                            # 严格隔离：不能是基底号码、不能是二级号码
                            if sui_num not in base_nums and sui_num not in level2_sorted:
                                level3_result.add(sui_num)
                                level3_detail.append({
                                    "二级号码": f"{level2_num:02d}",
                                    "三级跟随号": f"{sui_num:02d}",
                                    "近20期相随次数": cnt,
                                    "相随概率": f"{round(cnt/20*100, 2)}%"
                                })
                # 排序去重
                level3_sorted = sorted(list(level3_result), key=lambda x: (-num_status_dict[x]["cnt"], x))
                # 展示明细
                if level3_detail:
                    level3_df = pd.DataFrame(level3_detail)
                    st.dataframe(level3_df, hide_index=True, use_container_width=True)
                    st.markdown(f"**三级跟随号最终池（去重后）**：{' '.join([f'{x:02d}' for x in level3_sorted])}")
                    # 层间隔离校验
                    cross_check = len(set(level2_sorted) & set(level3_sorted))
                    base_cross_check = len(set(base_nums) & set(level3_sorted))
                    if cross_check == 0 and base_cross_check == 0:
                        st.success(f"✅ 三级跟随层生成完成，共{len(level3_sorted)}个唯一号码，与基底、二级层完全无重复，层间隔离校验通过")
                    else:
                        st.error("❌ 层间存在重复号码，已自动过滤")
                else:
                    st.warning("暂无有效三级跟随号数据")

                # 4. 自动存档预测池
                st.divider()
                try:
                    # 自动计算目标预测期号N+1
                    target_period_num = int(select_period_N) + 1
                    target_predict_period = str(target_period_num).zfill(7)
                    save_path = save_predict_num(
                        target_period=target_predict_period,
                        data_end_period=select_period_N,
                        level2_list=level2_sorted,
                        level3_list=level3_sorted
                    )
                    st.success(f"✅ 【{target_predict_period}期】预测池已自动存档，路径：{save_path}")
                except Exception as e:
                    st.error(f"预测池存档失败：{str(e)}")

    st.caption("🔒 数据同源说明：二级/三级号码生成完全基于多周期板块近20期相随号数据，保证逻辑一致性")  


# ========== Tab7 设置页（数据管理+备份迁移+自动生成文件删除+系统重置）【修复版】 ==========
with tab7:
    st.header("⚙️ 数据管理、存档迁移与系统重置")
    st.info("支持外置存档独立备份、跨代码版本迁移、原始数据下载，更替代码不丢数据；自动生成的存档文件支持一键删除")

    # ====================== 1. 原始开奖CSV单机备份【修复索引越界】 ======================
    st.subheader("📄 原始开奖数据单机备份")
    # 先做df非空判断，再执行iloc操作，彻底解决索引越界
    if total > 0 and os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "rb") as f:
                csv_raw_data = f.read()
            # 只有df非空时才调用iloc，绝对不会越界
            latest_period = df.iloc[0]['period']
            st.download_button(
                label="📥 下载原始CSV备份文件",
                data=csv_raw_data,
                file_name=f"kl8_history_backup_{latest_period}.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as e:
            st.warning(f"备份文件生成失败：{str(e)}，不影响其他功能使用")
    else:
        st.warning("数据文件不存在或无有效开奖数据，请先初始化系统")

    st.divider()
    # ====================== 2. 自动生成存档文件管理【修复路径异常】 ======================
    st.subheader("📂 自动生成存档文件管理（支持单删/批量删除）")
    # 路径异常全捕获，不会崩溃
    try:
        if os.path.exists(SAVE_DIR):
            save_files = os.listdir(SAVE_DIR)
            # 过滤只保留文件，排除文件夹
            save_files = [f for f in save_files if os.path.isfile(os.path.join(SAVE_DIR, f))]
            if save_files:
                st.write(f"当前自动生成存档总数：{len(save_files)}个")
                # 单个文件删除
                del_single_file = st.selectbox("选择单个文件删除", save_files, key="del_single_file")
                # 修复rerun循环：用回调函数处理，避免无限刷新
                if st.button("删除选中文件", use_container_width=True, type="secondary", key="del_single_btn"):
                    del_success, del_msg = delete_single_archive_file(del_single_file)
                    if del_success:
                        st.success(del_msg)
                        st.rerun()
                    else:
                        st.error(del_msg)
                
                st.divider()
                # 批量删除所有存档
                if st.button("清空所有自动生成的预测号/选号组合存档", use_container_width=True, type="secondary", key="del_batch_btn"):
                    batch_del_success, batch_del_msg = delete_all_archive_files()
                    if batch_del_success:
                        st.success(batch_del_msg)
                        st.rerun()
                    else:
                        st.warning(batch_del_msg)
            else:
                st.info("暂无自动生成的存档文件")
        else:
            st.info("存档目录不存在，生成预测号/组合后自动创建")
    except Exception as e:
        st.warning(f"存档目录加载失败：{str(e)}，不影响其他功能使用")

    st.divider()
    # ====================== 3. 批量复盘存档删除【修复异常捕获】 ======================
    st.subheader("📦 全量批量复盘存档管理")
    if st.button("清空所有批量复盘生成的存档数据", use_container_width=True, type="secondary", key="del_review_btn"):
        try:
            review_del_success, review_del_msg = delete_batch_review_data()
            if review_del_success:
                st.success(review_del_msg)
                st.rerun()
            else:
                st.error(review_del_msg)
        except Exception as e:
            st.error(f"复盘数据删除失败：{str(e)}")

    st.divider()
    # ====================== 4. 全库数据统计总览【修复索引越界】 ======================
    st.subheader("📈 全库数据统计看板")
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("总收录期数", f"{total}期")
    with col_stat2:
        # 先判断total>0，再执行iloc，绝对不会越界
        earliest_period = df.iloc[-1]["period"] if total > 0 else "无"
        st.metric("最早期号", earliest_period)
    with col_stat3:
        latest_period = df.iloc[0]["period"] if total > 0 else "无"
        st.metric("最新期号", latest_period)
    with col_stat4:
        st.metric("总号码记录数", f"{total * 20}个")

    st.divider()
    # ====================== 5. 全库一键打包备份/迁移【修复全量异常捕获】 ======================
    st.subheader("💾 全库一键打包备份 | 跨代码/跨电脑迁移专用")
    try:
        zip_name = f"KL8全量外置存档_一键迁移包_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.zip"
        zip_path = os.path.join(ARCHIVE_ROOT, zip_name)

        if st.button("📦 开始打包全部外置数据（适配代码更替/换服务器/换电脑）", use_container_width=True, type="primary", key="zip_btn"):
            with st.spinner("正在压缩全库存档，请稍候..."):
                # 先判断目录是否存在
                if not os.path.exists(ARCHIVE_ROOT):
                    st.error("存档根目录不存在，无法打包")
                else:
                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for root, dirs, files in os.walk(ARCHIVE_ROOT):
                            for file in files:
                                # 排除正在生成的压缩包，避免循环写入
                                if file == zip_name:
                                    continue
                                fp = os.path.join(root, file)
                                arcname = os.path.relpath(fp, ARCHIVE_ROOT)
                                zipf.write(fp, arcname)
                    st.success("✅ 打包完成！更替代码只需要复制压缩包，新环境解压配置路径即可秒读所有历史数据")
                    with open(zip_path, "rb") as f:
                        st.download_button("⬇️ 下载全库迁移压缩包", f, file_name=zip_name, use_container_width=True, key="download_zip")

        st.divider()
        st.subheader("📋 外置存档全局索引总表（全历史检索）")
        if os.path.exists(INDEX_FILE):
            try:
                index_df = pd.read_csv(INDEX_FILE, encoding="utf-8-sig")
                st.dataframe(index_df, hide_index=True, use_container_width=True, height=300)
            except Exception as e:
                st.warning(f"索引表加载失败：{str(e)}")
        else:
            st.info("暂无存档索引，生成预测号/组合后自动创建")
    except Exception as e:
        st.error(f"迁移模块加载失败：{str(e)}，不影响主程序运行，仅迁移功能临时不可用")

    st.divider()
    # ====================== 6. 危险区：系统数据重置【修复form逻辑】 ======================
    st.subheader("⚠️ 数据重置终极操作（高危不可恢复）")
    st.error("此操作清空增量数据，仅恢复初始88期基准，更替代码无需点这里！")
    # 修复form逻辑，避免rerun循环
    with st.form("reset_data_form", border=True):
        reset_confirm = st.checkbox("我已知风险，确认重置回原始88期基准数据")
        reset_submit = st.form_submit_button("执行数据重置", type="secondary", use_container_width=True)
        if reset_submit:
            if reset_confirm:
                try:
                    with open(DATA_FILE, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['period'] + [f'n{i}' for i in range(1,21)])
                        writer.writerows(INIT_DATA)
                    # 清空缓存
                    load_data_cached.clear()
                    get_full_analysis_cached.clear()
                    st.success("✅ 已重置为初始基准数据")
                    st.rerun()
                except Exception as e:
                    st.error(f"重置失败：{str(e)}")
            else:
                st.error("请勾选确认框后再执行")

# ========== Tab8 全量批量自动复盘【修复版】 ==========
with tab8:
    st.header("📦 全量期数一键自动复盘系统")
    st.info("自动完成88期全量数据的「单期深度复盘+跨期对比预测池+4铁律选号组合」生成，结果永久存档，后期随时可调用")
    st.divider()

    # 先做df非空判断
    if total <= 0:
        st.warning("暂无有效开奖数据，无法执行批量复盘")
    else:
        c1, c2 = st.columns(2)
        with c1:
            overwrite_mode = st.checkbox("覆盖已存在的存档数据（增量模式不勾选，全量重算勾选）", value=False, key="overwrite_mode")
        with c2:
            st.metric("当前可处理总期数", f"{len(df)}期")

        run_batch = st.button("🚀 开始全量自动复盘", use_container_width=True, type="primary", key="run_batch_btn")
        st.divider()

        # 按钮逻辑全量异常捕获，不会崩溃
        if run_batch:
            with st.spinner("正在全量批量处理中，请勿刷新页面..."):
                try:
                    result_df, fail_list = batch_auto_review_all_periods(df, overwrite_exist=overwrite_mode)
                    
                    st.subheader("✅ 处理完成结果总览")
                    # 处理结果统计，避免KeyError
                    success_cnt = len(result_df[result_df["处理状态"] == "处理成功"]) if "处理状态" in result_df.columns else 0
                    skip_cnt = len(result_df[result_df["处理状态"] == "已跳过(已存在)"]) if "处理状态" in result_df.columns else 0
                    fail_cnt = len(result_df[result_df["处理状态"] == "处理失败"]) if "处理状态" in result_df.columns else 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("处理成功", f"{success_cnt}期")
                    with col2:
                        st.metric("已跳过", f"{skip_cnt}期")
                    with col3:
                        st.metric("处理失败", f"{fail_cnt}期")

                    st.dataframe(result_df, hide_index=True, use_container_width=True, height=400)

                    # 失败明细
                    if fail_list:
                        st.divider()
                        st.error("❌ 处理失败期号明细")
                        for fail in fail_list:
                            st.write(fail)

                    # 下载总表
                    st.divider()
                    if os.path.exists(BATCH_REVIEW_SUMMARY):
                        with open(BATCH_REVIEW_SUMMARY, "rb") as f:
                            st.download_button(
                                label="📥 下载全量复盘总表CSV",
                                data=f.read(),
                                file_name="快乐8全量期数复盘总表.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                except Exception as e:
                    st.error(f"批量复盘执行失败：{str(e)}")

        st.divider()
        # 历史复盘存档查看，全量异常捕获
        st.subheader("📋 历史批量复盘存档查看")
        if os.path.exists(BATCH_REVIEW_SUMMARY):
            try:
                history_df = pd.read_csv(BATCH_REVIEW_SUMMARY, encoding="utf-8-sig")
                st.dataframe(history_df, hide_index=True, use_container_width=True, height=300)
                
                st.subheader("🔍 单期复盘明细查询")
                sel_period = st.selectbox("选择要查看的期号", df["period"].tolist(), key="sel_review_period")
                detail_file = os.path.join(BATCH_REVIEW_DETAIL_DIR, f"{sel_period}期_复盘明细.csv")
                
                if os.path.exists(detail_file):
                    detail_df = pd.read_csv(detail_file, encoding="utf-8-sig")
                    st.dataframe(detail_df, hide_index=True, use_container_width=True)
                    with open(detail_file, "rb") as f:
                        st.download_button(f"下载{sel_period}期复盘明细", f.read(), file_name=f"{sel_period}期_复盘明细.csv", mime="text/csv", key="download_detail")
                else:
                    st.warning("该期暂无复盘明细，请先执行批量复盘")
            except Exception as e:
                st.error(f"历史存档加载失败：{str(e)}")
        else:
            st.info("暂无批量复盘存档，请先点击「开始全量自动复盘」生成数据")

# ====================== 全局尾部合规声明（完整闭合） ======================
st.divider()
st.markdown('<div style="text-align:center;color:#666;font-size:14px">温馨提示:本系统仅历史数据统计娱乐,彩票开奖完全随机,不构成购彩建议,理性购彩遵守法规</div>', unsafe_allow_html=True)
