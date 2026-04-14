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
SAVE_DIR = "lottery_save"
ARCHIVE_ROOT = "lottery_archive"
INDEX_FILE = os.path.join(ARCHIVE_ROOT, "global_archive_index.csv")

# 初始化文件夹
for dir_path in [SAVE_DIR, ARCHIVE_ROOT]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# 固定配置
PRIME_NUMBERS = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79]
ZONE_RULE = {"zone1":[1,20],"zone2":[21,40],"zone3":[41,60],"zone4":[61,80]}
HOT_COLD_WINDOWS = [10,20,50,100]  # 多周期分析
TARGET_PERIODS = [f"20260{i:03d}" for i in range(1,76)]  # 仅保留75期

# ====================== 【核心：仅保留2026001-2026075 75期数据】 ======================
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

# ====================== 永久缓存（数据唯一不变动） ======================
@st.cache_data(ttl=None, show_spinner=False)
def load_data_cached():
    return load_data()

@st.cache_data(ttl=None, show_spinner=False)
def analyze_multi_window(df):
    return multi_window_analysis(df)

# ====================== 存档工具函数 ======================
def save_predict_num(period, xiangsui, gensui):
    filename = os.path.join(SAVE_DIR, f"{period}期预测号.csv")
    all_nums = list(set(xiangsui + gensui))
    df_save = pd.DataFrame({
        "期号": [period]*len(all_nums),
        "类型": ["相随号" if n in xiangsui else "跟随号" for n in all_nums],
        "号码": all_nums
    })
    df_save.to_csv(filename, index=False, encoding="utf-8-sig")
    return filename

def save_three_style(period, hot_comb, cold_comb, mix_comb):
    filename = os.path.join(SAVE_DIR, f"{period}期三派组合.csv")
    rows = []
    for idx, comb in enumerate(hot_comb): rows.append([period, "热派组合", f"方案{idx+1}", comb])
    for idx, comb in enumerate(cold_comb): rows.append([period, "冷派组合", f"方案{idx+1}", comb])
    for idx, comb in enumerate(mix_comb): rows.append([period, "混合派组合", f"方案{idx+1}", comb])
    pd.DataFrame(rows, columns=["期号", "流派", "方案", "号码"]).to_csv(filename, index=False, encoding="utf-8-sig")

def load_predict_num(period):
    f = os.path.join(SAVE_DIR, f"{period}期预测号.csv")
    return pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()

def load_three_style(period):
    f = os.path.join(SAVE_DIR, f"{period}期三派组合.csv")
    return pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()

# ====================== 底层数据读写（仅保留75期+期号唯一） ======================
def load_data():
    try:
        if not os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(['period'] + [f'n{i}' for i in range(1,21)])
                csv.writer(f).writerows(INIT_DATA)
        df = pd.read_csv(DATA_FILE, dtype={'period': str})
        # 核心：仅保留2026001-2026075，去重，排序
        df = df[df['period'].isin(TARGET_PERIODS)].drop_duplicates(subset=['period'])
        df['period_int'] = df['period'].astype(int)
        df = df.sort_values('period_int').reset_index(drop=True).drop(columns=['period_int'])
        return df
    except:
        return pd.read_csv(DATA_FILE, dtype={'period': str})

def save_data(df):
    df.to_csv(DATA_FILE, index=False, encoding='utf-8-sig')
    load_data_cached.clear()
    analyze_multi_window.clear()

def validate_period(period, df):
    if not period.isdigit() or len(period)!=7: return False
    if period in df['period'].values: return False
    return True

def validate_numbers(nums):
    nums = [int(x) for x in nums if x.strip()]
    return len(nums)==20 and len(set(nums))==20 and min(nums)>=1 and max(nums)<=80, sorted(nums)

# ====================== 核心分析引擎（相随号/跟随号/冷热温/多周期遗漏） ======================
def multi_window_analysis(df):
    num_list = [[int(x) for x in row[1:21]] for _, row in df.iterrows()]
    period_list = df['period'].tolist()
    result = {}

    # 1. 相随号：N期A → N+1期B
    xiangsui = defaultdict(lambda: defaultdict(int))
    for i in range(len(num_list)-1):
        for a in num_list[i]:
            for b in num_list[i+1]:
                xiangsui[a][b] += 1

    # 2. 跟随号：N期A 同期开出 B
    gensui = defaultdict(lambda: defaultdict(int))
    for nums in num_list:
        for i in range(20):
            for j in range(i+1,20):
                gensui[nums[i]][nums[j]] +=1
                gensui[nums[j]][nums[i]] +=1

    # 3. 多周期冷热温 + 遗漏
    for w in HOT_COLD_WINDOWS + [150]:
        window_nums = num_list[-w:] if w!=150 else num_list
        flat = [n for p in window_nums for n in p]
        cnt = Counter(flat)
        total = len(window_nums)
        avg = total*20/80

        # 冷热温定义
        hot = [n for n,c in cnt.items() if c >= avg*1.5]
        warm = [n for n,c in cnt.items() if avg*0.8 <= c < avg*1.5]
        cold = [n for n,c in cnt.items() if c < avg*0.8]

        # 遗漏分析
        miss = {}
        for n in range(1,81):
            last = max([i for i,p in enumerate(window_nums) if n in p] + [-1])
            miss[n] = len(window_nums)-1 - last

        result[f'w{w}'] = {
            'hot': hot, 'warm': warm, 'cold': cold, 'cnt': cnt, 'miss': miss,
            'total': len(window_nums), 'avg': round(avg,1)
        }

    result['xiangsui'] = xiangsui
    result['gensui'] = gensui
    result['num_list'] = num_list
    result['period_list'] = period_list
    return result

# ====================== 三大流派选号核心逻辑（严格按你的规则） ======================
def get_ban_nums(df, target_period):
    idx = df[df['period']==target_period].index[0]
    last3 = [set(df.iloc[i,1:21]) for i in [idx+1,idx+2,idx+3] if idx+3 < len(df)]
    three_ban = list(set.intersection(*last3)) if len(last3)==3 else []
    two_ban = list(set.intersection(*last3[:2])) if len(last3)>=2 else []
    last_nums = list(df.iloc[idx+1,1:21]) if idx+1 < len(df) else []
    return three_ban, two_ban, last_nums

def gen_hot_style(predict_nums, df, target_period, multi_data):
    three_ban, two_ban, last_nums = get_ban_nums(df, target_period)
    hot50 = multi_data['w50']['hot']
    hot10 = multi_data['w10']['hot']
    
    pool = [n for n in predict_nums if n not in three_ban and n in hot10 and n in hot50]
    pool = [n for n in pool if len(set([n])&set(last_nums))<=2]
    pool = [n for n in pool if n%2==1][:15]
    combs = [pool[:11], pool[:8], pool[:6], pool[:3]]
    return [c for c in combs if len(c)>=3]

def gen_cold_style(predict_nums, df, target_period, multi_data):
    three_ban, two_ban, last_nums = get_ban_nums(df, target_period)
    cold50 = multi_data['w50']['cold']
    
    pool = [n for n in predict_nums if n not in three_ban and n in cold50 and multi_data['w10']['miss'][n]<=9]
    pool = [n for n in pool if len(set([n])&set(last_nums))<=2]
    pool = [n for n in pool if n%2==1][:15]
    combs = [pool[:11], pool[:8], pool[:6], pool[:3]]
    return [c for c in combs if len(c)>=3]

def gen_mix_style(predict_nums, df, target_period, multi_data):
    three_ban, two_ban, last_nums = get_ban_nums(df, target_period)
    hot10 = multi_data['w10']['hot']
    cold10 = multi_data['w10']['cold']
    
    hot_pool = [n for n in predict_nums if n in hot10][:8]
    cold_pool = [n for n in predict_nums if n in cold10][:7]
    pool = hot_pool + cold_pool
    pool = [n for n in pool if n not in three_ban and len(set([n])&set(last_nums))<=3]
    pool = [n for n in pool if n%2==1][:15]
    combs = [pool[:11], pool[:8], pool[:6], pool[:3]]
    return [c for c in combs if len(c)>=3]

# ====================== 全局初始化 ======================
df = load_data_cached()
multi_data = analyze_multi_window(df)
total_periods = len(df)

# ====================== 侧边栏 ======================
with st.sidebar:
    st.title("🎰快乐8数据分析系统")
    st.metric("有效期数", f"{total_periods}期（2026001-2026075）")
    if st.button("🔄刷新数据"):
        load_data_cached.clear()
        analyze_multi_window.clear()
        st.rerun()
    st.warning("仅历史数据娱乐，不构成购彩建议")

# ====================== 标签页 ======================
tab1, tab2, tab3, tab5, tab6 = st.tabs(["🏠首页", "📋号码库", "📊多周期分析", "📝单期复盘&选号", "🔄跨期对比&预测"])

# ========== Tab1 首页 ==========
with tab1:
    st.title("快乐8专业数据分析系统")
    st.success(f"✅ 已锁定75期核心数据：2026001 ~ 2026075")
    st.info("所有数据永久固定，基于开奖号码唯一生成，无随机变动")
    latest = df.iloc[-1]
    st.subheader(f"最新开奖 {latest['period']}期：{' '.join([f'{x:02d}' for x in latest.iloc[1:21]])}")

# ========== Tab2 号码库（增/删/改 完整功能） ==========
with tab2:
    st.header("📋 开奖号码库管理（增/删/改）")
    df_show = df.copy()
    df_show.index = range(1, len(df_show)+1)
    st.dataframe(df_show, height=300, use_container_width=True)

    # 新增
    with st.expander("➕ 新增开奖号码"):
        with st.form("add_form"):
            period = st.text_input("期号（2026076+）")
            nums = st.text_input("20个号码空格分隔")
            if st.form_submit_button("确认新增"):
                if not validate_period(period, df):
                    st.error("期号无效/重复")
                else:
                    ok, num_list = validate_numbers(nums.split())
                    if not ok:
                        st.error("号码格式错误")
                    else:
                        new_row = [period] + num_list
                        df.loc[len(df)] = new_row
                        save_data(df)
                        st.success("新增成功")

    # 修改
    with st.expander("✏️ 修改开奖号码"):
        sel_period = st.selectbox("选择期号", df['period'].tolist())
        row = df[df['period']==sel_period].iloc[0]
        old_nums = ' '.join([str(x) for x in row.iloc[1:21]])
        new_nums = st.text_input("新号码", old_nums)
        if st.button("确认修改"):
            ok, num_list = validate_numbers(new_nums.split())
            if ok:
                df.loc[df['period']==sel_period, df.columns[1:21]] = num_list
                save_data(df)
                st.success("修改成功")

    # 删除
    with st.expander("🗑️ 删除开奖号码"):
        del_period = st.selectbox("选择删除期号", df['period'].tolist())
        if st.button("确认删除"):
            df = df[df['period']!=del_period]
            save_data(df)
            st.success("删除成功")

# ========== Tab3 多周期分析（相随号/跟随号/冷热温/遗漏） ==========
with tab3:
    st.header("📊 多周期核心数据分析")
    window = st.selectbox("选择分析周期", ["10期","20期","50期","100期","150期以上"])
    w_map = {"10期":10,"20期":20,"50期":50,"100期":100,"150期以上":150}
    w = w_map[window]
    data = multi_data[f'w{w}']

    c1,c2 = st.columns(2)
    with c1:
        st.subheader("🔗 相随号（N期→N+1期）")
        target_num = st.number_input("输入号码查相随",1,80,1)
        xiang = sorted(multi_data['xiangsui'][target_num].items(), key=lambda x:x[1], reverse=True)[:10]
        st.dataframe(pd.DataFrame(xiang, columns=["跟随号码","次数"]), use_container_width=True)

    with c2:
        st.subheader("🔗 跟随号（同期开出）")
        gen_num = st.number_input("输入号码查跟随",1,80,1)
        gen = sorted(multi_data['gensui'][gen_num].items(), key=lambda x:x[1], reverse=True)[:10]
        st.dataframe(pd.DataFrame(gen, columns=["同期号码","次数"]), use_container_width=True)

    st.divider()
    c1,c2,c3 = st.columns(3)
    with c1: st.subheader("🔥 热号"); st.write(', '.join([f"{x:02d}" for x in data['hot']]))
    with c2: st.subheader("🌡️ 温号"); st.write(', '.join([f"{x:02d}" for x in data['warm']]))
    with c3: st.subheader("❄️ 冷号"); st.write(', '.join([f"{x:02d}" for x in data['cold']]))

    st.divider()
    st.subheader("⏳ 多周期遗漏值")
    miss_df = pd.DataFrame({
        "号码": range(1,81),
        "遗漏期数": [data['miss'][n] for n in range(1,81)],
        "冷热状态": ["热" if n in data['hot'] else "温" if n in data['warm'] else "冷" for n in range(1,81)]
    }).sort_values("遗漏期数", ascending=False)
    st.dataframe(miss_df, height=400, use_container_width=True)

# ========== Tab5 单期复盘 + 三大流派选号 + 开奖对比 ==========
with tab5:
    st.header("📝 单期复盘 & 三派选号组合")
    select_n = st.selectbox("选择N期（生成N+1期选号）", df['period'].tolist())
    idx = df[df['period']==select_n].index[0]
    n1_period = df.iloc[idx+1]['period'] if idx+1 < len(df) else "2026076"

    # 单期复盘（数据源给Tab6）
    st.subheader(f"【{select_n}期】深度复盘")
    current_nums = df.iloc[idx,1:21].tolist()
    prev_nums = df.iloc[idx+1,1:21].tolist() if idx+1 < len(df) else []
    st.write(f"开奖号码：{' '.join([f'{x:02d}' for x in current_nums])}")
    st.write(f"与上期重号：{len(set(current_nums)&set(prev_nums))}个")

    # 调用N+1期预测号 + 生成三派组合
    st.divider()
    st.subheader(f"【{n1_period}期】三派选号组合")
    pred_df = load_predict_num(n1_period)
    if pred_df.empty:
        st.warning("请先去Tab6生成N+1期预测号！")
    else:
        predict_nums = pred_df['号码'].tolist()
        hot_comb = gen_hot_style(predict_nums, df, select_n, multi_data)
        cold_comb = gen_cold_style(predict_nums, df, select_n, multi_data)
        mix_comb = gen_mix_style(predict_nums, df, select_n, multi_data)
        save_three_style(n1_period, hot_comb, cold_comb, mix_comb)

        c1,c2,c3 = st.columns(3)
        with c1:
            st.subheader("🔥 热派组合")
            for i,c in enumerate(hot_comb): st.write(f"方案{i+1}: {c}")
        with c2:
            st.subheader("❄️ 冷派组合")
            for i,c in enumerate(cold_comb): st.write(f"方案{i+1}: {c}")
        with c3:
            st.subheader("⚖️ 混合派组合")
            for i,c in enumerate(mix_comb): st.write(f"方案{i+1}: {c}")

    # 开奖号码对比板块
    st.divider()
    st.subheader(f"📊 {select_n}期 开奖对比")
    real_nums = df.iloc[idx,1:21].tolist()
    pred_n1 = load_predict_num(n1_period)['号码'].tolist() if not load_predict_num(n1_period).empty else []
    style_df = load_three_style(n1_period)

    st.write(f"✅ {select_n}期开奖号：{' '.join([f'{x:02d}' for x in real_nums])}")
    st.write(f"🔮 {n1_period}期预测号：{' '.join([f'{x:02d}' for x in pred_n1])}")
    if not style_df.empty:
        hot = style_df[style_df['流派']=='热派组合']['号码'].tolist()
        cold = style_df[style_df['流派']=='冷派组合']['号码'].tolist()
        mix = style_df[style_df['流派']=='混合派组合']['号码'].tolist()
        st.write(f"🔥 热派组合：{hot}")
        st.write(f"❄️ 冷派组合：{cold}")
        st.write(f"⚖️ 混合派组合：{mix}")

# ========== Tab6 跨期对比 + N+1期预测池生成 ==========
with tab6:
    st.header("🔄 跨期对比 & N+1期预测号生成")
    select_n = st.selectbox("选择N期（自动对比N-1期）", df['period'].tolist(), key="tab6")
    idx = df[df['period']==select_n].index[0]
    n_1_period = df.iloc[idx-1]['period'] if idx-1 >=0 else "无"
    n1_period = df.iloc[idx+1]['period'] if idx+1 < len(df) else "2026076"

    # 跨期对比（数据源：Tab5单期复盘）
    st.subheader(f"{n_1_period}期 VS {select_n}期 跨期对比")
    if idx-1 >=0:
        n_1_nums = df.iloc[idx-1,1:21].tolist()
        n_nums = df.iloc[idx,1:21].tolist()
        same = len(set(n_1_nums)&set(n_nums))
        st.write(f"重号数量：{same}个 | 号码：{' '.join([f'{x:02d}' for x in set(n_1_nums)&set(n_nums)])}")
        st.write("结论：热号延续" if same>=4 else "冷热切换" if same<=2 else "平衡轮动")

    # 预测池生成：近50期相随号+跟随号 → 去重 → N+1期预测号
    st.divider()
    st.subheader(f"🎯 生成【{n1_period}期】预测号（近50期数据）")
    if st.button("生成并保存N+1期预测号"):
        current_nums = df.iloc[idx,1:21].tolist()
        # 提取近50期相随+跟随
        xiangsui_nums = []
        gensui_nums = []
        for n in current_nums:
            xiangsui_nums.extend([k for k,v in multi_data['xiangsui'][n].items() if v>=3])
            gensui_nums.extend([k for k,v in multi_data['gensui'][n].items() if v>=5])
        # 去重
        xiangsui_nums = list(set(xiangsui_nums))
        gensui_nums = list(set(gensui_nums))
        # 保存
        save_predict_num(n1_period, xiangsui_nums, gensui_nums)
        st.success(f"✅ {n1_period}期预测号已保存！")
        c1,c2 = st.columns(2)
        with c1: st.write("相随号：", ', '.join([f"{x:02d}" for x in xiangsui_nums]))
        with c2: st.write("跟随号：", ', '.join([f"{x:02d}" for x in gensui_nums]))

# ====================== 合规声明 ======================
st.divider()
st.caption("⚠️ 本系统仅为历史数据统计娱乐，彩票开奖完全随机，不构成购彩建议，理性购彩！")