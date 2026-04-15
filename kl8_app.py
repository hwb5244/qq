# ====================== 核心依赖库（无冗余导入） ======================
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import os
import csv
import datetime
import shutil
import zipfile

# ====================== 页面基础配置（必须首行执行） ======================
st.set_page_config(
    page_title="快乐8专业数据分析系统",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== 全局常量定义（统一管理无硬编码） ======================
DATA_FILE = "kl8_history_data.csv"
SAVE_DIR = "lottery_save"      # 预测号/选号组合存档
ARCHIVE_ROOT = "lottery_archive"  # 全局备份根目录
INDEX_FILE = os.path.join(ARCHIVE_ROOT, "global_archive_index.csv") # 全局索引

# 初始化所有文件夹，避免首次运行报错
for dir_path in [SAVE_DIR, ARCHIVE_ROOT]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# 1-80质数固定列表
PRIME_NUMBERS = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79]
ZONE_RULE = {"zone1":[1,20],"zone2":[21,40],"zone3":[41,60],"zone4":[61,80]}
HOT_COLD_FACTOR = 2
PLAY_RULE = {"选10":10,"选8":8,"选7":7,"选5":5,"选20":20}

# ====================== 【75期原始开奖基准数据 - 2026001至20260075】 ======================
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
["2026075",1,4,17,18,21,22,23,24,25,30,41,47,48,50,54,55,56,57,62,78]
]

# ====================== 缓存装饰器（确保结果唯一） ======================
@st.cache_data(ttl=3600)
def load_data_cached():
    return load_data()

@st.cache_data(ttl=3600)
def get_full_analysis_cached(df, window=None):
    return analyze_full_data(df, window)

# ====================== 存档工具函数 ======================
def save_predict_num(period, level2_list, level3_list):
    filename = os.path.join(SAVE_DIR, f"{period}期预测号.csv")
    df_save = pd.DataFrame({
        "期号": [period] * len(level2_list + level3_list),
        "候选等级": ["相随号"] * len(level2_list) + ["跟随号"] * len(level3_list),
        "号码": level2_list + level3_list
    })
    df_save.to_csv(filename, index=False, encoding="utf-8-sig")
    return filename

def save_genre_comb(period, genre, comb_list):
    filename = os.path.join(SAVE_DIR, f"{period}期{genre}组合.csv")
    rows = []
    for idx, nums in enumerate(comb_list):
        rows.append([period, genre, f"方案{idx+1}", " ".join([f"{n:02d}" for n in nums])])
    df_save = pd.DataFrame(rows, columns=["期号", "流派", "方案编号", "选号号码"])
    df_save.to_csv(filename, index=False, encoding="utf-8-sig")
    return filename

def load_predict_num(period):
    filename = os.path.join(SAVE_DIR, f"{period}期预测号.csv")
    if os.path.exists(filename):
        return pd.read_csv(filename, encoding="utf-8-sig")
    return None

def load_genre_comb(period, genre):
    filename = os.path.join(SAVE_DIR, f"{period}期{genre}组合.csv")
    if os.path.exists(filename):
        return pd.read_csv(filename, encoding="utf-8-sig")
    return None

def calc_match_rate(predict_nums, real_nums):
    predict_set = set([int(x) for x in predict_nums])
    real_set = set([int(x) for x in real_nums])
    match = predict_set & real_set
    match_cnt = len(match)
    rate = round(match_cnt / len(predict_set) * 100, 2) if predict_set else 0
    return {"匹配号码": sorted(list(match)), "匹配个数": match_cnt, "正确率%": rate}

# ====================== 底层模块1：数据读写/校验（仅保留2026001-20260075） ======================
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
        
        # 核心修复：仅保留2026001至20260075期数据
        df['period_num'] = df['period'].astype(int)
        df = df[(df['period_num'] >= 2026001) & (df['period_num'] <= 20260075)]
        df = df.drop(columns=['period_num'])
        
        # 去重：保留最后一期
        df = df.drop_duplicates(subset=['period'], keep='last')
        df['period_num'] = df['period'].astype(int)
        df = df.sort_values('period_num', ascending=False).reset_index(drop=True)
        df = df.drop(columns=['period_num'])
        return df
    except Exception:
        with open(DATA_FILE, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['period'] + [f'n{i}' for i in range(1, 21)])
            w.writerows(INIT_DATA)
        df = pd.read_csv(DATA_FILE, dtype={'period': str})
        df['period_num'] = df['period'].astype(int)
        df = df[(df['period_num'] >= 2026001) & (df['period_num'] <= 20260075)]
        df = df.drop(columns=['period_num'])
        return df

def save_new_data(period, numbers):
    try:
        period_num = int(period)
        if period_num < 2026001 or period_num > 20260075:
            return False, "期号需在2026001-20260075范围内"
        
        with open(DATA_FILE, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([period] + sorted(numbers))
        return True, "录入成功"
    except Exception as e:
        return False, f"录入失败：{str(e)}"

def update_period_data(period, numbers, df):
    try:
        df.loc[df['period'] == period, [f'n{i}' for i in range(1, 21)]] = sorted(numbers)
        df.to_csv(DATA_FILE, index=False, encoding='utf-8-sig')
        return True, "修改成功"
    except Exception as e:
        return False, f"修改失败：{str(e)}"

def delete_period_data(period, df):
    new_df = df[df['period'] != period].reset_index(drop=True)
    new_df.to_csv(DATA_FILE, index=False, encoding='utf-8-sig')
    return new_df

def validate_period_unique(period, df):
    if not period or not period.isdigit():
        return False, "期号纯数字不能为空"
    if period in df['period'].values:
        return False, "期号重复"
    period_num = int(period)
    if period_num < 2026001 or period_num > 20260075:
        return False, "期号需在2026001-20260075范围内"
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

# ====================== 底层模块2：数据分析引擎 ======================
def analyze_full_data(df, window=None):
    data = df.head(window).copy() if window else df.copy()
    num_list = [[int(x) for x in row.iloc[1:21].tolist()] for _, row in data.iterrows()]
    flat_nums = [n for p in num_list for n in p]
    total = len(num_list)
    avg = len(flat_nums) / 80 if total > 0 else 0
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
    t = len(flat) or 1
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
    t = len(flat) or 1
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

# ====================== 同源核心函数 ======================
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

# ====================== 标签页创建（按要求重命名） ======================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🏠首页", "📋号码库", "📊多周期数据", "🎯选号组合", "📈多期对比", "🔮选号池生成"])

# ========== Tab1 首页 ==========
with tab1:
    st.title("🎰福彩快乐8专业数据分析系统")
    st.subheader(f"当前收录：{total}期 | 仅保留2026001-20260075期数据")
    st.error("开奖完全随机，仅历史统计娱乐，不构成购彩建议！")
    if total > 0:
        l = df.iloc[0]
        st.subheader(f"最新{l['period']}期开奖：{' '.join([f'{x:02d}' for x in l.iloc[1:21]])}")

# ========== Tab2 号码库（新增修改功能） ==========
with tab2:
    st.header("📋开奖号码库管理")
    st.info("仅支持2026001-20260075期数据的新增、修改、删除")
    
    # 新增
    with st.form("add"):
        c1, c2 = st.columns(2)
        with c1:
            p1 = st.text_input("期号（2026001-20260075）")
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
                    ok, msg = save_new_data(p1, m2)
                    if ok:
                        st.success(msg)
                        load_data_cached.clear()
                        get_full_analysis_cached.clear()
                        st.rerun()
                    else:
                        st.error(msg)
    
    st.divider()
    
    # 修改
    with st.form("update"):
        c1, c2 = st.columns(2)
        with c1:
            dp = st.selectbox("选择修改期号", df['period'].tolist())
        with c2:
            np = st.text_input("新20个号码空格分隔")
        if st.form_submit_button("确认修改", type="secondary", use_container_width=True):
            if np.strip():
                v2, m2 = validate_numbers(np.split())
                if not v2:
                    st.error(m2)
                else:
                    ok, msg = update_period_data(dp, m2, df)
                    if ok:
                        st.success(msg)
                        load_data_cached.clear()
                        get_full_analysis_cached.clear()
                        st.rerun()
                    else:
                        st.error(msg)
            else:
                st.warning("未输入新号码，未修改")
    
    st.divider()
    
    # 删除
    with st.form("del"):
        dp = st.selectbox("选择删除期号", df['period'].tolist(), key="del_period")
        if st.form_submit_button("确认删除", type="secondary", use_container_width=True):
            df = delete_period_data(dp, df)
            st.success("删除成功")
            load_data_cached.clear()
            get_full_analysis_cached.clear()
            st.rerun()
    
    st.divider()
    st.dataframe(df, use_container_width=True, height=400)

# ========== Tab3 多周期数据（按要求加入相随号、跟随号、冷热温、遗漏） ==========
with tab3:
    st.header("📊多周期数据分析")
    window_options = {"近10期": 10, "近20期": 20, "近50期": 50, "近100期": 100, "150期以上全量": None}
    sel = st.selectbox("选择分析周期", list(window_options.keys()))
    w = window_options[sel]
    if w and total < w:
        st.warning(f"当前仅{total}期，未达到{w}期门槛，请补充数据！")
    else:
        fd = get_full_analysis_cached(df, w)
        st.info(f"分析维度：{sel}，共{fd['total']}期")
        st.divider()
        
        # 1. 冷热温号数据
        st.subheader("🔥 冷热温号数据")
        num_status = get_num_status(fd)
        hot_nums = [n for n in range(1, 81) if num_status[n]["st"] == "hot"]
        cold_nums = [n for n in range(1, 81) if num_status[n]["st"] == "cold"]
        warm_nums = [n for n in range(1, 81) if num_status[n]["st"] == "warm"]
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**热号（{len(hot_nums)}个）**：{' '.join([f'{x:02d}' for x in hot_nums])}")
        with c2:
            st.markdown(f"**温号（{len(warm_nums)}个）**：{' '.join([f'{x:02d}' for x in warm_nums])}")
        with c3:
            st.markdown(f"**冷号（{len(cold_nums)}个）**：{' '.join([f'{x:02d}' for x in cold_nums])}")
        
        st.divider()
        
        # 2. 遗漏数据
        st.subheader("📉 遗漏值全表")
        st.dataframe(fd['miss_analysis']['miss_df'], use_container_width=True, height=300)
        
        st.divider()
        
        # 3. 相随号数据（N期A→N+1期B、C）
        st.subheader("🔗 相随号数据（N期A→N+1期B、C）")
        follow_df = pd.DataFrame([{"上期": f"{k[0]:02d}", "下期": f"{k[1]:02d}", "次数": v} for k, v in fd["follow_matrix"]["top10"]])
        st.dataframe(follow_df, use_container_width=True)
        
        st.divider()
        
        # 4. 跟随号数据（同期A开出同时开出D、F）
        st.subheader("📌 跟随号数据（同期A开出同时开出D、F）")
        co_occur_df = pd.DataFrame([{"A": f"{k[0]:02d}", "B": f"{k[1]:02d}", "次数": v} for k, v in fd["co_occur_matrix"]["top10"]])
        st.dataframe(co_occur_df, use_container_width=True)

# ========== Tab6 选号池生成（仅保留预测池生成，基于近50期） ==========
with tab6:
    st.header("🔮 选号池生成")
    st.info("基于N期正式开奖号码，调用近50期数据，生成N+1期预测号（相随号+跟随号，去重）")
    period_list = df["period"].tolist() if len(df) > 0 else []
    if not period_list:
        st.error("暂无开奖数据，请先初始化号码库！")
    else:
        target_period = st.selectbox("选择【N期】期号", period_list)
        if st.button("🚀 生成N+1期预测号", use_container_width=True, type="primary"):
            # 计算N+1期期号
            next_period_num = int(target_period) + 1
            next_period = str(next_period_num).zfill(7)
            
            # 获取N期开奖号码
            current_idx = df[df["period"] == target_period].index[0]
            current_nums = [int(x) for x in df.iloc[current_idx].iloc[1:21].tolist()]
            
            # 获取近50期数据
            full_analysis_50 = get_full_analysis_cached(df, 50)
            num_status = get_num_status(full_analysis_50)
            
            # 生成相随号（N期A→N+1期B）
            follow_dict = full_analysis_50["follow_matrix"]["dict"]
            l2_result = set()
            for n in current_nums:
                valid_list = []
                for k, c in follow_dict.items():
                    if isinstance(k, tuple) and len(k) == 2:
                        a, b = k
                        if (a == n and b not in current_nums) or (b == n and a not in current_nums):
                            match_num = b if a == n else a
                            valid_list.append((c, match_num))
                valid_list_sorted = sorted(valid_list, key=lambda x: x[0], reverse=True)[:5]
                for c, match_num in valid_list_sorted:
                    l2_result.add(match_num)
            
            # 生成跟随号（同期A开出同时开出D、F）
            co_dict = full_analysis_50["co_occur_matrix"]["dict"]
            l3_result = set()
            for n in current_nums:
                valid_list = []
                for k, c in co_dict.items():
                    if isinstance(k, tuple) and len(k) == 2:
                        a, b = k
                        if (a == n and b not in current_nums) or (b == n and a not in current_nums):
                            match_num = b if a == n else a
                            valid_list.append((c, match_num))
                valid_list_sorted = sorted(valid_list, key=lambda x: x[0], reverse=True)[:5]
                for c, match_num in valid_list_sorted:
                    l3_result.add(match_num)
            
            # 去重并排序
            l2_sorted = sorted(list(l2_result))
            l3_sorted = sorted(list(l3_result - l2_result))
            
            # 存档
            save_path = save_predict_num(next_period, l2_sorted, l3_sorted)
            st.success(f"✅ 【{next_period}期】预测号已自动存档完成！保存路径：{save_path}")
            
            # 展示
            st.divider()
            st.subheader(f"🎯 {next_period}期预测号")
            st.markdown("#### 🔴 相随号")
            st.markdown(" ".join([fmt_num(n, num_status) for n in l2_sorted]) or "无", unsafe_allow_html=True)
            st.markdown("#### 🟡 跟随号（去重后）")
            st.markdown(" ".join([fmt_num(n, num_status) for n in l3_sorted]) or "无", unsafe_allow_html=True)

# ========== Tab4 选号组合（调用预测号，生成三大流派，新增开奖对比） ==========
with tab4:
    st.header("🎯 选号组合")
    st.info("调用N+1期预测号，生成热派、冷派、混合派组合，并进行开奖号码对比")
    period_list = df["period"].tolist() if len(df) > 0 else []
    if not period_list:
        st.error("暂无开奖数据，请先初始化号码库！")
    else:
        target_period = st.selectbox("选择【N期】期号", period_list, key="tab4_period")
        next_period_num = int(target_period) + 1
        next_period = str(next_period_num).zfill(7)
        
        # 加载预测号
        pred_df = load_predict_num(next_period)
        if pred_df is None or pred_df.empty or "号码" not in pred_df.columns:
            st.warning(f"⚠️ 未找到{next_period}期预测号，请先在【选号池生成】模块生成！")
        else:
            predict_nums = pred_df["号码"].tolist()
            
            # 获取近50期和近10期数据
            full_analysis_50 = get_full_analysis_cached(df, 50)
            full_analysis_10 = get_full_analysis_cached(df, 10)
            num_status = get_num_status(full_analysis_50)
            
            # 获取N期开奖号码
            current_idx = df[df["period"] == target_period].index[0]
            current_nums = [int(x) for x in df.iloc[current_idx].iloc[1:21].tolist()]
            
            # 获取前三期连出号
            two_continuous = []
            three_continuous = []
            if current_idx + 2 < len(df):
                n1 = set(df.iloc[current_idx+1].iloc[1:21].tolist())
                n2 = set(df.iloc[current_idx+2].iloc[1:21].tolist())
                two_continuous = list(n1 & n2)
                if current_idx + 3 < len(df):
                    n3 = set(df.iloc[current_idx+3].iloc[1:21].tolist())
                    three_continuous = list(n1 & n2 & n3)
            
            # 生成三大流派组合
            @st.cache_data(ttl=0)
            def generate_genres(predict_nums, full_50, full_10, current_nums, two_con, three_con):
                # 基础候选池
                base_candidate = list(set(predict_nums))
                base_candidate = [n for n in base_candidate if n not in three_con]
                
                # 热号惯性流派
                hot_candidate = [n for n in base_candidate if get_num_status(full_10)[n]["st"] == "hot" and n not in two_con]
                hot_combinations = []
                if len(hot_candidate) >= 10:
                    for i in range(3):
                        comb = sorted(hot_candidate, key=lambda x: (-get_num_status(full_50)[x]["cnt"], x))[:10]
                        hot_combinations.append(comb)
                
                # 冷号回补流派
                cold_candidate = [n for n in base_candidate if get_num_status(full_50)[n]["st"] == "cold" and n not in two_con]
                miss_df = full_50["miss_analysis"]["miss_df"]
                high_miss = miss_df[miss_df["当前遗漏"] >= 5]["号码"].tolist()
                cold_candidate = list(set(cold_candidate + high_miss))
                cold_candidate = [n for n in cold_candidate if n not in three_con]
                cold_combinations = []
                if len(cold_candidate) >= 10:
                    for i in range(3):
                        comb = sorted(cold_candidate, key=lambda x: (-int(miss_df[miss_df["号码"] == x]["当前遗漏"].values[0]), x))[:10]
                        cold_combinations.append(comb)
                
                # 热冷混合流派
                mixed_hot = [n for n in base_candidate if get_num_status(full_10)[n]["st"] == "hot"][:6]
                mixed_cold = [n for n in base_candidate if get_num_status(full_50)[n]["st"] == "cold"][:4]
                mixed_candidate = list(set(mixed_hot + mixed_cold))
                mixed_combinations = []
                if len(mixed_candidate) >= 10:
                    for i in range(3):
                        comb = sorted(mixed_candidate, key=lambda x: (-get_num_status(full_50)[x]["cnt"], x))[:10]
                        mixed_combinations.append(comb)
                
                return {
                    "hot": hot_combinations,
                    "cold": cold_combinations,
                    "mixed": mixed_combinations
                }
            
            genres = generate_genres(predict_nums, full_analysis_50, full_analysis_10, current_nums, two_continuous, three_continuous)
            
            # 存档三大流派
            if genres["hot"]:
                save_genre_comb(next_period, "热派", genres["hot"])
            if genres["cold"]:
                save_genre_comb(next_period, "冷派", genres["cold"])
            if genres["mixed"]:
                save_genre_comb(next_period, "混合派", genres["mixed"])
            
            # 展示三大流派
            st.divider()
            st.subheader(f"🎯 {next_period}期三大流派组合")
            
            st.markdown("#### 🔥 热号惯性流派")
            if genres["hot"]:
                for idx, comb in enumerate(genres["hot"], 1):
                    st.markdown(f"**方案{idx}**：{' '.join([fmt_num(n, num_status) for n in comb])}", unsafe_allow_html=True)
            else:
                st.warning("候选池号码不足，无法生成热派组合")
            
            st.markdown("#### ❄️ 冷号回补流派")
            if genres["cold"]:
                for idx, comb in enumerate(genres["cold"], 1):
                    st.markdown(f"**方案{idx}**：{' '.join([fmt_num(n, num_status) for n in comb])}", unsafe_allow_html=True)
            else:
                st.warning("候选池号码不足，无法生成冷派组合")
            
            st.markdown("#### ⚖️ 热冷混合流派")
            if genres["mixed"]:
                for idx, comb in enumerate(genres["mixed"], 1):
                    st.markdown(f"**方案{idx}**：{' '.join([fmt_num(n, num_status) for n in comb])}", unsafe_allow_html=True)
            else:
                st.warning("候选池号码不足，无法生成混合派组合")
            
            # 开奖号码对比板块
            st.divider()
            st.subheader("📊 开奖号码对比")
            
            # N期开奖号
            st.markdown(f"#### 🔴 {target_period}期开奖号")
            st.markdown(" ".join([fmt_num(n, num_status) for n in sorted(current_nums)]), unsafe_allow_html=True)
            
            # N+1期预测号
            st.markdown(f"#### 🔵 {next_period}期预测号")
            st.markdown(" ".join([fmt_num(n, num_status) for n in sorted(predict_nums)]), unsafe_allow_html=True)
            
            # N+1期热派组合
            st.markdown(f"#### 🔥 {next_period}期热派组合")
            hot_df = load_genre_comb(next_period, "热派")
            if hot_df is not None and not hot_df.empty:
                for idx, row in hot_df.iterrows():
                    comb = [int(x) for x in row["选号号码"].split()]
                    st.markdown(f"**{row['方案编号']}**：{' '.join([fmt_num(n, num_status) for n in comb])}", unsafe_allow_html=True)
            else:
                st.warning("暂无热派组合")
            
            # N+1期冷派组合
            st.markdown(f"#### ❄️ {next_period}期冷派组合")
            cold_df = load_genre_comb(next_period, "冷派")
            if cold_df is not None and not cold_df.empty:
                for idx, row in cold_df.iterrows():
                    comb = [int(x) for x in row["选号号码"].split()]
                    st.markdown(f"**{row['方案编号']}**：{' '.join([fmt_num(n, num_status) for n in comb])}", unsafe_allow_html=True)
            else:
                st.warning("暂无冷派组合")
            
            # N+1期混合派组合
            st.markdown(f"#### ⚖️ {next_period}期混合派组合")
            mixed_df = load_genre_comb(next_period, "混合派")
            if mixed_df is not None and not mixed_df.empty:
                for idx, row in mixed_df.iterrows():
                    comb = [int(x) for x in row["选号号码"].split()]
                    st.markdown(f"**{row['方案编号']}**：{' '.join([fmt_num(n, num_status) for n in comb])}", unsafe_allow_html=True)
            else:
                st.warning("暂无混合派组合")

# ========== Tab5 多期对比（支持多期深度复盘） ==========
with tab5:
    st.header("📈 多期对比")
    st.info("支持同时选择多期数据进行深度复盘，并得出结论")
    period_list = df["period"].tolist() if len(df) > 0 else []
    if not period_list:
        st.error("暂无开奖数据，请先初始化号码库！")
    else:
        selected_periods = st.multiselect("选择要复盘的期号", period_list, default=period_list[:3])
        if st.button("生成多期深度复盘报告", use_container_width=True, type="primary"):
            if not selected_periods:
                st.warning("请至少选择一期！")
            else:
                full_analysis = get_full_analysis_cached(df)
                num_status = get_num_status(full_analysis)
                
                # 存储各期复盘结果
                review_results = []
                for period in selected_periods:
                    row = df[df["period"] == period].iloc[0]
                    nums = [int(x) for x in row.iloc[1:21].tolist()]
                    idx = df[df["period"] == period].index[0]
                    
                    prev_nums = None
                    if idx < len(df) - 1:
                        prev_nums = [int(x) for x in df.iloc[idx + 1].iloc[1:21].tolist()]
                    
                    # 计算结构
                    numbers = sorted(nums)
                    odd = sum(n % 2 for n in numbers)
                    even = 20 - odd
                    small = sum(1 for n in numbers if n <= 40)
                    large = 20 - small
                    r0 = sum(1 for n in numbers if n % 3 == 0)
                    r1 = sum(1 for n in numbers if n % 3 == 1)
                    r2 = sum(1 for n in numbers if n % 3 == 2)
                    sumv = sum(numbers)
                    span = numbers[-1] - numbers[0]
                    
                    review_results.append({
                        "期号": period,
                        "开奖号码": " ".join([f"{x:02d}" for x in numbers]),
                        "奇偶比例": f"{odd}:{even}",
                        "大小比例": f"{small}:{large}",
                        "012路比例": f"{r0}:{r1}:{r2}",
                        "和值": sumv,
                        "跨度": span
                    })
                
                # 展示多期对比表
                st.divider()
                st.subheader("📊 多期核心指标对比表")
                compare_df = pd.DataFrame(review_results)
                st.dataframe(compare_df, hide_index=True, use_container_width=True)
                
                # 总结结论
                st.divider()
                st.subheader("📝 多期对比结论")
                # 统计奇偶
                odd_list = [int(x.split(":")[0]) for x in compare_df["奇偶比例"]]
                avg_odd = round(np.mean(odd_list), 1)
                # 统计大小
                small_list = [int(x.split(":")[0]) for x in compare_df["大小比例"]]
                avg_small = round(np.mean(small_list), 1)
                # 统计和值
                avg_sum = round(np.mean(compare_df["和值"]), 1)
                
                st.info(f"① 所选期数平均奇偶比例为 {avg_odd}:{20-avg_odd}，{'奇数偏多' if avg_odd > 10 else '偶数偏多' if avg_odd < 10 else '奇偶均衡'}；")
                st.info(f"② 所选期数平均大小比例为 {avg_small}:{20-avg_small}，{'小号偏多' if avg_small > 10 else '大号偏多' if avg_small < 10 else '大小均衡'}；")
                st.info(f"③ 所选期数平均和值为 {avg_sum}，可作为下期和值参考；")
                st.info("④ 建议结合近10期冷热数据，优先选择热号惯性或冷号回补方向。")

# ====================== 全局尾部合规声明 ======================
st.divider()
st.markdown("""
<div style="text-align:center;color:#666;font-size:14px">
⚠️ 本系统仅历史数据统计娱乐，彩票开奖完全随机，不构成任何购彩建议，理性购彩遵守法规
</div>
""", unsafe_allow_html=True)