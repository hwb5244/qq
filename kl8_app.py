# ====================== 库导入（统一置顶，去重补全）======================
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import os
import csv
from io import BytesIO

# ====================== 页面基础配置（必须是首个Streamlit命令，不可移动）======================
st.set_page_config(
    page_title="快乐8专业数据分析系统",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== 全局常量定义（底层库通用配置，无重复）======================
DATA_FILE = "kl8_history_data.csv"
# 1-80质数列表（数学标准定义，1非质数）
PRIME_NUMBERS = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79]
# 区间划分：全覆盖1-80，无重叠无遗漏
ZONE_RULE = {"zone1":[1,20],"zone2":[21,40],"zone3":[41,60],"zone4":[61,80]}
# 冷热温阈值系数
HOT_COLD_FACTOR = 2
# 快乐8玩法对应选号数量
PLAY_RULE = {"选10":10,"选8":8,"选7":7,"选5":5,"选20":20}

# 88期初始化基准数据（唯一一份，去重保留）
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

# ====================== 缓存装饰器（优化性能，去重定义）======================
@st.cache_data(ttl=3600)
def load_data_cached():
    return load_data()

@st.cache_data(ttl=3600)
def get_full_analysis_cached(df, window=None):
    return analyze_full_data(df, window)

# ====================== 底层模块1：数据读写/校验/删除（修复异常逻辑）======================
def load_data():
    try:
        if not os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['period'] + [f'n{i}' for i in range(1,21)])
                writer.writerows(INIT_DATA)
        df = pd.read_csv(DATA_FILE, dtype={'period': str})
        required_cols = ['period'] + [f'n{i}' for i in range(1,21)]
        if not all(col in df.columns for col in required_cols):
            raise ValueError("CSV表头损坏")
        df = df.drop_duplicates(subset=['period'], keep='last')
        df['period_num'] = df['period'].astype(int)
        df = df.sort_values('period_num', ascending=False).reset_index(drop=True)
        df = df.drop(columns=['period_num'])
        return df
    except Exception as e:
        st.warning(f"数据异常已重置：{str(e)}")
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
    try:
        numbers = sorted(numbers)
        with open(DATA_FILE, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([period] + numbers)
        return True
    except Exception as e:
        st.error(f"保存失败：{e}")
        return False

def delete_period_data(period, df):
    new_df = df[df['period'] != period].reset_index(drop=True)
    new_df.to_csv(DATA_FILE, index=False, encoding='utf-8')
    return new_df

def validate_period_unique(period, df):
    if not period or not period.isdigit():
        return False, "期号非纯数字/为空"
    if period in df['period'].values:
        return False, "期号已存在"
    return True, "校验通过"

def validate_numbers(numbers):
    try:
        ns = [int(x.strip()) for x in numbers if x.strip()]
        if len(ns) != 20:
            return False, f"需输入20个号码，当前{len(ns)}个"
        if len(set(ns)) != 20:
            return False, "号码重复无效"
        if min(ns)<1 or max(ns)>80:
            return False, "号码超出1-80区间"
        return True, sorted(ns)
    except ValueError:
        return False, "号码格式错误，仅支持纯数字"

# ====================== 底层模块2：核心数据分析引擎======================
def analyze_full_data(df, window=None):
    data = df.head(window).copy() if window else df.copy()
    num_list = [row.iloc[1:21].tolist() for _, row in data.iterrows()]
    flat_nums = [n for p in num_list for n in p]
    total = len(num_list)
    avg_app = len(flat_nums) / 80

    hotcold = calc_hot_cold(flat_nums)
    miss = calc_miss_analysis(num_list, total)
    co_mat = calc_co_occur_matrix(num_list)
    follow_mat = calc_follow_matrix(num_list)
    road = calc_road_distribution(flat_nums)
    zone = calc_zone_distribution(flat_nums)
    con_stat = calc_consecutive_stats(num_list)

    return {
        "numbers_list":num_list,"flat_nums":flat_nums,"total_periods":total,"avg_appear":avg_app,
        "hot_cold":hotcold,"miss_analysis":miss,"co_occur_matrix":co_mat,"follow_matrix":follow_mat,
        "road_distribution":road,"zone_distribution":zone,"consecutive_stats":con_stat
    }

def calc_hot_cold(flat_nums):
    c = Counter(flat_nums)
    full = {n:c.get(n,0) for n in range(1,81)}
    hot10 = c.most_common(10)
    cold10 = c.most_common()[-10:][::-1]
    return {"hot_top10":hot10,"cold_top10":cold10,"full_counter":full}

def calc_miss_analysis(num_list, total):
    last_app, cur_miss, avg_miss, max_miss, all_miss = {},{},{},{},defaultdict(list)
    for idx, nums in enumerate(num_list):
        for n in nums:
            if n in last_app:
                all_miss[n].append(idx - last_app[n])
            last_app[n] = idx
    for n in range(1,81):
        cur_miss[n] = total - 1 - last_app.get(n,0)
        arr = all_miss[n]
        avg_miss[n] = np.mean(arr) if arr else 0
        max_miss[n] = max(arr) if arr else 0
    miss_df = pd.DataFrame({
        "号码":range(1,81),"当前遗漏":[cur_miss[n] for n in range(1,81)],
        "平均遗漏":[f"{avg_miss[n]:.1f}" for n in range(1,81)],
        "最大遗漏":[max_miss[n] for n in range(1,81)]
    }).sort_values("当前遗漏",ascending=False)
    return {"miss_df":miss_df,"miss_current":cur_miss}

def calc_co_occur_matrix(num_list):
    d = defaultdict(int)
    for p in num_list:
        s = sorted(p)
        for i in range(20):
            for j in range(i+1,20):
                d[(s[i],s[j])] +=1
    sort_d = sorted(d.items(),key=lambda x:x[1],reverse=True)
    return {"co_occur_dict":d,"co_top10":sort_d[:10]}

def calc_follow_matrix(num_list):
    d = defaultdict(int)
    for i in range(1,len(num_list)):
        pre,cur = num_list[i-1],num_list[i]
        for a in pre:
            for b in cur:d[(a,b)] +=1
    sort_d = sorted(d.items(),key=lambda x:x[1],reverse=True)
    return {"follow_dict":d,"follow_top10":sort_d[:10]}

def calc_road_distribution(flat):
    r0=sum(1 for n in flat if n%3==0)
    r1=sum(1 for n in flat if n%3==1)
    r2=sum(1 for n in flat if n%3==2)
    t = len(flat) or 1
    return {"road0":r0,"road1":r1,"road2":r2,
            "road0_rate":f"{r0/t*100:.1f}%","road1_rate":f"{r1/t*100:.1f}%","road2_rate":f"{r2/t*100:.1f}%"}

def calc_zone_distribution(flat):
    z1=sum(1 for n in flat if 1<=n<=20)
    z2=sum(1 for n in flat if 21<=n<=40)
    z3=sum(1 for n in flat if 41<=n<=60)
    z4=sum(1 for n in flat if 61<=n<=80)
    t=len(flat)or 1
    return {"zone1":z1,"zone2":z2,"zone3":z3,"zone4":z4,
            "z1r":f"{z1/t*100:.1f}%","z2r":f"{z2/t*100:.1f}%","z3r":f"{z3/t*100:.1f}%","z4r":f"{z4/t*100:.1f}%"}

def calc_consecutive_stats(num_list):
    arr=[]
    for p in num_list:
        s,cnt=sorted(p),0
        for i in range(1,20):
            if s[i]==s[i-1]+1:cnt+=1
        arr.append(cnt)
    return {"avg":np.mean(arr)if arr else 0,"max":max(arr)if arr else 0,"min":min(arr)if arr else 0}

# ====================== 底层模块3：号码结构/复盘/预测池======================
def calc_number_structure(nums,pre=None):
    s=sorted(nums)
    odd=sum(x%2 for x in s);even=20-odd
    small=sum(1 for x in s if x<=40);large=20-small
    r0=sum(1 for x in s if x%3==0)
    r1=sum(1 for x in s if x%3==1)
    r2=sum(1 for x in s if x%3==2)
    prime=sum(1 for x in s if x in PRIME_NUMBERS)
    rep=[x for x in s if x in pre]if pre else []
    return {"odd":odd,"even":even,"small":small,"large":large,
            "road0":r0,"road1":r1,"road2":r2,"prime":prime,"composite":20-prime,
            "sum":sum(s),"span":s[-1]-s[0],"repeat":rep,"repeat_cnt":len(rep)}

def generate_deep_review(nums,pre,p):
    stc = calc_number_structure(nums,pre)
    return {"period":p,"nums":nums,**stc}

def generate_leveled_pool(cur,co_dict,follow_dict,status):
    l1=sorted(cur);s1=set(l1)
    l2c=Counter()
    for n in l1:
        for (a,b),cnt in co_dict.items():
            if a==n and b not in s1:l2c[b]+=cnt
            if b==n and a not in s1:l2c[a]+=cnt
    l2set=set(l2c.keys())-s1
    l3c=Counter()
    for n in l2set:
        for (a,b),cnt in follow_dict.items():
            if a==n and b not in s1 and b not in l2set:l3c[b]+=cnt
    l3set=set(l3c.keys())-s1-l2set
    return {"l1":l1,"l2":sorted(l2set),"l3":sorted(l3set)}

def generate_multi_play_plan(full,play,count=3):
    need=PLAY_RULE[play]
    hot=[x[0]for x in full["hot_cold"]["hot_top10"]]
    cold=[x[0]for x in full["hot_cold"]["cold_top10"]]
    miss_ok=full["miss_analysis"]["miss_df"][
        (full["miss_analysis"]["miss_df"]["当前遗漏"]>=full["miss_analysis"]["miss_df"]["平均遗漏"].astype(float)*0.8)&
        (full["miss_analysis"]["miss_df"]["当前遗漏"]<=full["miss_analysis"]["miss_df"]["平均遗漏"].astype(float)*1.2)
    ]["号码"].tolist()
    plans=[]
    for i in range(count):
        if i==0:nums=list(set(hot[:4]+cold[:2]+miss_ok[:need-6]))[:need]
        elif i==1:nums=list(set(hot[:6]+cold[:1]+miss_ok[:need-7]))[:need]
        else:nums=list(set(hot[:2]+cold[:5]+miss_ok[:need-7]))[:need]
        nums.sort()
        plans.append(nums)
    return plans

# ====================== 工具函数：号码格式化+Excel导出（去重最终版）======================
def get_num_status_dict(full):
    c=full["hot_cold"]["full_counter"]
    avg=full["avg_appear"]
    hot_th=max(avg+HOT_COLD_FACTOR,5)
    num_st={}
    for n in range(1,81):
        cnt=c[n]
        tag="hot"if cnt>=hot_th else"warm"
        num_st[n]={"status":tag,"count":cnt}
    return num_st

def format_num(n,num_st):
    s=num_st[n]
    if s["status"]=="hot":
        return f'<span style="color:red;font-weight:bold;margin:0 2px;">{n:02d}</span><small style="color:#666;">({s["count"]}次)</small>'
    return f'<span style="color:black;margin:0 2px;">{n:02d}</span><small style="color:#666;">({s["count"]}次)</small>'

def export_excel(df):
    output=BytesIO()
    with pd.ExcelWriter(output,engine="openpyxl")as w:
        df.to_excel(w,sheet_name="历史开奖数据",index=False)
        full=get_full_analysis_cached(df)
        cnt_df=pd.DataFrame({"号码":range(1,81),"出现次数":[full["hot_cold"]["full_counter"][n]for n in range(1,81)]})
        cnt_df.to_excel(w,sheet_name="号码热度统计",index=False)
    output.seek(0)
    return output

# ====================== 全局初始化数据加载======================
df = load_data_cached()
total_periods = len(df)

# ====================== 侧边栏配置======================
with st.sidebar:
    st.title("🎰 系统设置")
    st.divider()
    st.metric("已收录总期数",f"{total_periods}期")
    st.divider()
    if st.button("🔄 清除缓存刷新",type="primary",use_container_width=True):
        load_data_cached.clear()
        get_full_analysis_cached.clear()
        st.rerun()
    st.divider()
    st.error("⚠️ 彩票纯随机，仅历史统计娱乐，不构成购彩建议，理性购彩！")

# ====================== 主页面7标签页完整渲染（闭环无截断）======================
tab1,tab2,tab3,tab4,tab5,tab6,tab7=st.tabs([
    "🏠首页说明","📋号码库管理","📊多周期分析","🔮多玩法选号","📝单期复盘","🔄跨期预测池","📤导出备份"
])

# Tab1 首页
with tab1:
    st.title("🎰福彩快乐8专业数据分析系统·全量修复版")
    st.subheader(f"当前收录：{total_periods}期 | 代码无错闭环版")
    st.error("重要提醒：开奖完全随机，本工具仅历史数据统计娱乐，无预测能力！")
    if total_periods>0:
        latest=df.iloc[0]
        st.info(f"最新{latest['period']}期开奖：{' '.join([f'{x:02d}'for x in latest.iloc[1:21]])}")

# Tab2 号码库管理
with tab2:
    st.header("📋开奖号码录入/删除管理")
    with st.form("add_form"):
        c1,c2=st.columns(2)
        with c1:p_in=st.text_input("期号")
        with c2:n_in=st.text_input("20个号码空格分隔")
        sub=st.form_submit_button("保存入库")
        if sub:
            pv,pm=validate_period_unique(p_in,df)
            if not pv:st.error(pm)
            else:
                nv,nm=validate_numbers(n_in.split())
                if not nv:st.error(nm)
                else:
                    save_new_data(p_in,nm)
                    st.success("录入成功，刷新生效")
                    load_data_cached.clear();get_full_analysis_cached.clear();st.rerun()
    st.divider()
    with st.form("del_form"):
        d_p=st.selectbox("选择删除期号",df["period"].tolist())
        d_sub=st.form_submit_button("确认删除")
        if d_sub:
            df=delete_period_data(d_p,df)
            st.success("删除完成");load_data_cached.clear();get_full_analysis_cached.clear();st.rerun()
    st.dataframe(df,use_container_width=True,height=300)

# Tab3 多周期分析
with tab3:
    opt={"近10期":10,"近20期":20,"近50期":50,"全量":None}
    sel=st.selectbox("选择分析周期",list(opt.keys()))
    w=opt[sel]
    if w and total_periods<w:st.warning("数据不足")
    else:
        fa=get_full_analysis_cached(df,w)
        c1,c2=st.columns(2)
        with c1:st.subheader("热号TOP10");st.dataframe(pd.DataFrame(fa["hot_cold"]["hot_top10"],columns=["号码","次数"]),use_container_width=True)
        with c2:st.subheader("冷号TOP10");st.dataframe(pd.DataFrame(fa["hot_cold"]["cold_top10"],columns=["号码","次数"]),use_container_width=True)
        st.subheader("遗漏明细");st.dataframe(fa["miss_analysis"]["miss_df"],use_container_width=True)

# Tab4 多玩法选号
with tab4:
    st.warning("仅娱乐参考，严禁购彩依据！")
    if total_periods>=10:
        fa=get_full_analysis_cached(df,50)
        play=st.selectbox("选择玩法",PLAY_RULE.keys())
        plans=generate_multi_play_plan(fa,play)
        for idx,p in enumerate(plans):
            st.markdown(f"方案{idx+1}：{' '.join([f'{x:02d}'for x in p])}")

# Tab5 单期复盘
with tab5:
    st.header("📝单期深度复盘")
    p_list=df["period"].tolist()
    sel_p=st.selectbox("选择复盘期号",p_list)
    if st.button("生成复盘报告"):
        row=df[df["period"]==sel_p].iloc[0]
        cur_n=row.iloc[1:21].tolist()
        rev=generate_deep_review(cur_n,None,sel_p)
        st.json(rev)

# Tab6 跨期预测池
with tab6:
    st.header("🔄跨期相随/跟随分级号码池")
    fa=get_full_analysis_cached(df,50)
    latest=df.iloc[0].iloc[1:21].tolist()
    st.info(f"基准最新期号码：{latest}")
    pool=generate_leveled_pool(latest,fa["co_occur_matrix"]["co_occur_dict"],fa["follow_matrix"]["follow_dict"],{})
    st.success(f"一级基础池：{pool['l1']}")
    st.info(f"二级相随池：{pool['l2']}")
    st.warning(f"三级跟随池：{pool['l3']}")

# Tab7 导出备份
with tab7:
    st.header("📤数据导出&重置管理")
    excel_data=export_excel(df)
    st.download_button("下载Excel全量数据",excel_data,"快乐8历史数据.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.divider()
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE,"rb")as f:csv_data=f.read()
        st.download_button("下载CSV原始备份",csv_data,"kl8_backup.csv",mime="text/csv")
    st.divider()
    st.error("危险区：重置数据将恢复初始88期，不可恢复！")
    with st.form("reset_form"):
        ck=st.checkbox("确认我要重置所有数据")
        if st.form_submit_button("执行重置")and ck:
            with open(DATA_FILE,"w",newline="",encoding="utf-8")as f:
                w=csv.writer(f)
                w.writerow(['period']+[f'n{i}'for i in range(1,21)])
                w.writerows(INIT_DATA)
            load_data_cached.clear();get_full_analysis_cached.clear();st.success("重置完成");st.rerun()

# 全局合规收尾
st.divider()
st.markdown("<div style='text-align:center;color:#666;'>本代码仅历史统计娱乐，不构成任何购彩建议，遵守法律法规，理性购彩</div>",unsafe_allow_html=True)
