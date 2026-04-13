import streamlit as st
import pandas as pd
import numpy as np
import os
import collections
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# -------------------------- 页面配置 --------------------------
st.set_page_config(
    page_title="快乐8智能分析系统",
    layout="wide",
    initial_sidebar_state="expanded"
)
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# -------------------------- 会话状态初始化 --------------------------
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "history_df" not in st.session_state:
    st.session_state.history_df = None
if "predict_nums" not in st.session_state:
    st.session_state.predict_nums = []
if "hit_result" not in st.session_state:
    st.session_state.hit_result = None

# -------------------------- 路径配置 --------------------------
BASE_DIR = "happy8_data"
os.makedirs(BASE_DIR, exist_ok=True)
DATA_FILE = os.path.join(BASE_DIR, "lottery_history.csv")
SAVE_DIR = os.path.join(BASE_DIR, "save_records")
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------- 初始化默认数据 --------------------------
def init_default_data():
    if os.path.exists(DATA_FILE):
        return
    periods = []
    for i in range(1, 89):
        period = f"2025{i:06d}"
        nums = list(np.random.choice(range(1, 81), 20, replace=False))
        nums.sort()
        row = {"期号": period}
        for idx, n in enumerate(nums, 1):
            row[f"号码{idx}"] = n
        periods.append(row)
    df = pd.DataFrame(periods)
    df.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")

# -------------------------- 数据加载 --------------------------
def load_data():
    try:
        if not os.path.exists(DATA_FILE):
            init_default_data()
        df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
        df = df.sort_values("期号", ascending=True).reset_index(drop=True)
        num_cols = [c for c in df.columns if "号码" in c]
        df["所有号码"] = df[num_cols].apply(lambda x: sorted(x.tolist()), axis=1)
        st.session_state.history_df = df
        st.session_state.data_loaded = True
        return df
    except Exception as e:
        st.error(f"加载失败：{str(e)}")
        init_default_data()
        return load_data()

# -------------------------- 冷热号分析 --------------------------
def get_hot_cold(df, recent=20):
    if len(df) < 5:
        return {}, {}, {}
    df_use = df.tail(recent)
    all_nums = []
    for nlist in df_use["所有号码"].tolist():
        all_nums.extend(nlist)
    cnt = collections.Counter(all_nums)
    sorted_list = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
    hot = dict(sorted_list[:16])
    warm = dict(sorted_list[16:32])
    cold = dict(sorted_list[32:])
    return hot, warm, cold

# -------------------------- 遗漏号统计 --------------------------
def get_missing_stats(df):
    miss = {n:0 for n in range(1,81)}
    history = df["所有号码"].tolist()[::-1]
    
    for num in range(1,81):
        for idx, nums in enumerate(history):
            if num in nums:
                miss[num] = idx
                break
    sorted_miss = sorted(miss.items(), key=lambda x:x[1], reverse=True)
   =lambda x:x[1], reverse=True)
    return miss, sorted_miss

# -------------------------- 三连号检查 --------------------------
def has_triple(nums):
    s = sorted(nums)
    for i in range(len(s)-2):
        if s[i+2] - s[i] == 2:
            return True
    return False

# -------------------------- 自定义规则生成号码 --------------------------
def generate_custom_nums(hot, cold, total=8, hot_pct=0.5, allow_triple=False):
    hot_list = list(hot.keys())[:15]
    cold_list = list(cold.keys())[:15]
    if len(hot_list) < 5 or len(cold_list) < 5:
        pool = list(range(1, 81))
        np.random.shuffle(pool)
        return sorted(pool[:total])
    
    hot_need = int(total * hot_pct)
    cold_need = total - hot_need
    selected = []
    
    while len(selected) < hot_need and hot_list:
        num = np.random.choice(hot_list, 1)[0]
        if num not in selected:
            selected.append(num)
    
    while len(selected) < total and cold_list:
        num = np.random.choice(cold_list, 1)[0]
        if num not in selected:
            selected.append(num)
    
    selected = selected[:total]
    if not allow_triple and has_triple(selected):
        return generate_custom_nums(hot, cold, total, hot_pct, allow_triple)
    return sorted(selected)

# -------------------------- 命中率核对 --------------------------
def check_hit_rate(pred_nums, real_nums):
    hit = set(pred_nums) & set(real_nums)
    hit_cnt = len(hit)
    rate = round(hit_cnt / len(pred_nums) * 100, 2) if pred_nums else 0
    return hit_cnt, rate, sorted(list(hit))

# -------------------------- 多期批量复盘 --------------------------
def batch_backtest(df, test_count=10, total_num=8, hot_pct=0.5):
    if len(df) < test_count + 5:
        return []
    df_sub = df.tail(test_count + 5)
    records = []
    
    for i in range(test_count):
        target_idx = -(i+1)
        train_df = df_sub.iloc[:target_idx]
        real_nums = df_sub["所有号码"].iloc[target_idx]
        period = df_sub["期号"].iloc[target_idx]
        
        h,_,c = get_hot_cold(train_df)
        pred = generate_custom_nums(h,c,total_num, hot_pct, False)
        cnt, rate, _ = check_hit_rate(pred, real_nums)
        
        records.append({
            "期号": period,
            "预测": pred,
            "命中数": cnt,
            "命中率%": rate
        })
    return records

# -------------------------- 相随/跟随号 --------------------------
def get_sui_data():
    df = st.session_state.history_df
    if len(df) < 10: return {}
    res = collections.defaultdict(int)
    for nums in df["所有号码"]:
        s = sorted(nums)
        for i in range(len(s)-1):
            if s[i+1]-s[i]==1:
                res[(s[i],s[i+1])] +=1
    return dict(sorted(res.items(), key=lambda x:x[1], reverse=True)[:20])

def get_follow_data():
    df = st.session_state.history_df
    if len(df) <10: return {}
    follow = collections.defaultdict(list)
    data = df["所有号码"].tolist()
    for i in range(len(data)-1):
        cur,nxt = set(data[i]), data[i+1]
        for num in cur:
            follow[num].extend([x for x in nxt if x!=num])
    out={}
    for k,v in follow.items():
        out[k] = collections.Counter(v).most_common(5)
    return out

# -------------------------- 保存记录 --------------------------
def save_record(nums, hit_info):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    df = st.session_state.history_df
    period = df["期号"].iloc[-1] if len(df)>0 else "未知"
    hit_cnt, rate = hit_info if hit_info else (0,0)
    item = {
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "期号": period,
        "预测号码": ",".join(map(str,nums)),
        "命中数": hit_cnt,
        "命中率(%)": rate
    }
    path = os.path.join(SAVE_DIR, f"record_{ts}.csv")
    pd.DataFrame([item]).to_csv(path, index=False, encoding="utf-8-sig")

# -------------------------- 主界面 --------------------------
def main():
    st.title("🎯 快乐8智能分析系统（终极完整版）")
    st.markdown("✅ 自动命中率 | ✅ 自定义规则 | ✅ 批量复盘 | ✅ 遗漏统计 | ✅ 趋势图表")
    st.markdown("---")

    with st.spinner("加载数据中..."):
        df = load_data()
    hot, warm, cold = get_hot_cold(df)
    miss_dict, miss_sorted = get_missing_stats(df)

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("总期数", len(df))
    with col2: st.metric("最新期号", df["期号"].iloc[-1] if len(df) else "无")
    with col3:
        if st.session_state.hit_result:
            hc, hr = st.session_state.hit_result
            st.metric("本次命中率", f"{hr}%")

    st.markdown("---")

    tabs = st.tabs([
        "📊 数据总览",
        "🔥 冷热分析",
        "🔍 遗漏统计",
        "📈 趋势图表",
        "🎯 自定义预测",
        "🧪 多期复盘",
        "💾 存档记录"
    ])
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = tabs

    # Tab1 数据总览
    with tab1:
        st.subheader("最近15期开奖")
        show_cols = ["期号"] + [f"号码{i}" for i in range(1,21)]
        st.dataframe(df[show_cols].tail(15), use_container_width=True)

    # Tab2 冷热号
    with tab2:
        st.subheader("冷热温号统计")
        c1,c2,c3 = st.columns(3)
        with c1: st.success("🔥 热号"); st.write(hot)
        with c2: st.info("🌡 温号"); st.write(warm)
        with c3: st.error("❄️ 冷号"); st.write(cold)

    # Tab3 遗漏号统计（新增）
    with tab3:
        st.subheader("🔍 号码遗漏统计（越久没出越靠前）")
        miss_df = pd.DataFrame(miss_sorted, columns=["号码", "遗漏期数"])
        top_miss = miss_df.head(30)
        st.dataframe(top_miss, use_container_width=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.success("🔥 最热号码（遗漏0期）")
            hottest = miss_df[miss_df["遗漏期数"]==0]["号码"].tolist()
            st.write(hottest)
        with col_b:
            st.error("❄️ 最冷号码（遗漏最久）")
            st.write(miss_df.head(10)["号码"].tolist())

    # Tab4 趋势图 / 统计图（新增）
    with tab4:
        st.subheader("📊 出号频率 & 趋势图表")
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### 热号出号频率柱状图")
            if hot:
                fig, ax = plt.subplots(figsize=(5,3))
                keys = list(hot.keys())
                vals = list(hot.values())
                ax.bar(keys, vals, color="#ff4b4b")
                ax.set_title("热号出现次数")
                st.pyplot(fig)
        
        with c2:
            st.markdown("#### 号码遗漏分布")
            fig, ax = plt.subplots(figsize=(5,3))
            miss_vals = [x[1] for x in miss_sorted[:30]]
            ax.hist(miss_vals, bins=10, color="#3d85c6")
            ax.set_title("前30名遗漏分布")
            st.pyplot(fig)
        
        st.markdown("---")
        st.markdown("#### 最近10期命中趋势预览")
        test_rec = batch_backtest(df, 10, 8, 0.5)
        if test_rec:
            rates = [x["命中率%"] for x in test_rec]
            periods = [x["期号"][-4:] for x in test_rec]
            fig, ax = plt.subplots(figsize=(8,3))
            ax.plot(periods, rates, marker="o", color="green")
            ax.set_title("近10期预测命中率趋势")
            st.pyplot(fig)

    # Tab5 自定义预测
    with tab5:
        st.subheader("🎛 自定义规则预测")
        col_a, col_b = st.columns(2)
        with col_a:
            total_count = st.slider("生成号码数量",6,12,8)
            hot_ratio = st.slider("热号占比",0.2,0.8,0.5)
        with col_b:
            allow_3 = st.checkbox("允许三连号",False)
            auto_check = st.checkbox("自动核对命中率",True)

        if st.button("🚀 生成预测号码",type="primary"):
            with st.spinner("计算中..."):
                nums = generate_custom_nums(hot,cold,total_count,hot_ratio,allow_3)
                st.session_state.predict_nums = nums
                hit_period, real_nums, hit_info = None, None, None
                
                if auto_check:
                    real = df["所有号码"].iloc[-1]
                    h_cnt, h_rate, h_list = check_hit_rate(nums, real)
                    st.session_state.hit_result = (h_cnt, h_rate)
                    st.success(f"✅ 预测：{nums}")
                    st.info(f"📊 开奖：{real}")
                    st.success(f"🎯 命中 {h_cnt} 个 | 命中率 {h_rate}% | {h_list}")
                save_record(nums, st.session_state.hit_result)
                st.balloons()
        
        if st.session_state.predict_nums:
            st.markdown("### 当前预测")
            st.success(st.session_state.predict_nums)

    # Tab6 多期批量复盘（新增）
    with tab6:
        st.subheader("🧪 历史多期批量回测复盘")
        col1, col2 = st.columns(2)
        with col1:
            test_count = st.slider("回测期数",5,30,10)
        with col2:
            test_num = st.slider("每组号码数",6,12,8)
            test_hot = st.slider("回测热号占比",0.2,0.8,0.5)
        
        if st.button("📊 开始批量复盘"):
            with st.spinner("回测计算中..."):
                res = batch_backtest(df, test_count, test_num, test_hot)
                if not res:
                    st.warning("数据不足，无法回测")
                else:
                    result_df = pd.DataFrame(res)
                    avg_rate = round(result_df["命中率%"].mean(),2)
                    max_hit = result_df["命中数"].max()
                    
                    st.success(f"📈 回测完成 | 平均命中率：{avg_rate}% | 最高命中：{max_hit}")
                    st.dataframe(result_df, use_container_width=True)
                    
                    # 复盘趋势图
                    fig, ax = plt.subplots(figsize=(8,3))
                    ax.plot(range(len(res)), [x["命中率%"] for x in res], marker="o", color="blue")
                    ax.set_title(f"近{test_count}期复盘命中率趋势")
                    st.pyplot(fig)

    # Tab7 存档记录
    with tab7:
        st.subheader("💾 历史预测记录")
        records = []
        for f in os.listdir(SAVE_DIR):
            if f.endswith(".csv"):
                temp_df = pd.read_csv(os.path.join(SAVE_DIR,f), encoding="utf-8-sig")
                records.append(temp_df)
        if not records:
            st.info("暂无记录")
        else:
            all_rec = pd.concat(records, ignore_index=True)
            all_rec = all_rec.sort_values("时间", ascending=False)
            st.dataframe(all_rec, use_container_width=True)

if __name__ == "__main__":
    main()
