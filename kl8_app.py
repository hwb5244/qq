import streamlit as st
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from collections import defaultdict

# ========== 全局Session初始化 ==========
init_keys = [
    "raw_original_db", "26_year_db", "tongqi_db", "re_leng_data_db",
    "feature_columns", "feature_engineered_db", "model_dict", "predict_result",
    "hit_summary", "accuracy_detail", "multi_model_dict", "model_metrics",
    "final_predict_result", "multi_model_hit_summary"
]
for key in init_keys:
    if key not in st.session_state:
        st.session_state[key] = pd.DataFrame()

# ========== 全局配置 ==========
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(
    page_title="快乐8数据分析预测系统V2.0",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== 全局常量 ==========
LOTTERY_RULE = {
    "total_numbers": 80,
    "draw_per_period": 20,
    "number_range": range(1, 81),
    "interval_count": 8,
    "tail_number_count": 10
}
CUSTOM_RULE = {
    "follow_number": "同期一起出现的号码（跟随号）",
    "accompany_number": "本期开出N，下期开出M（相随号）",
    "repeat_number": "本期开出A，下期继续开出A（重号）"
}

# ========== 数据加载函数 ==========
@st.cache_data(ttl=3600)
def load_standard_data(uploaded_file=None):
    if uploaded_file is None:
        st.warning("使用测试数据，请上传真实开奖文件")
        mock_periods = 200
        mock_data = []
        for period in range(1, mock_periods+1):
            draw_numbers = sorted(np.random.choice(LOTTERY_RULE["number_range"], size=20, replace=False))
            mock_data.append([f"2026{str(period).zfill(3)}"] + draw_numbers)
        columns = ["期号"] + [f"开奖号码{i}" for i in range(1, 21)]
        raw_df = pd.DataFrame(mock_data, columns=columns)
    else:
        if uploaded_file.name.endswith('.csv'):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file)
    
    st.session_state["raw_original_db"] = raw_df.copy(deep=True)
    df = raw_df.copy(deep=True)
    df = df.sort_values("期号", ascending=True).reset_index(drop=True)
    df["开奖号码集合"] = df[[f"开奖号码{i}" for i in range(1, 21)]].values.tolist()
    df["开奖号码集合"] = df["开奖号码集合"].apply(lambda x: set(int(num) for num in x if pd.notna(num)))
    return df

# ========== 特征工程函数 ==========
def build_feature_engineer(df, rolling_window_list=[5, 10, 30]):
    feature_df = df.copy(deep=True)
    total_periods = len(feature_df)
    
    # 1. 热度、遗漏值
    for window in rolling_window_list:
        hot_count_matrix = np.zeros((total_periods, 80))
        miss_count_matrix = np.zeros((total_periods, 80))
        for period_idx in range(total_periods):
            start_idx = max(0, period_idx - window)
            history_numbers = feature_df.loc[start_idx:period_idx-1, "开奖号码集合"].tolist()
            if len(history_numbers) == 0:
                continue
            for num in LOTTERY_RULE["number_range"]:
                appear_count = sum(1 for draw in history_numbers if num in draw)
                hot_count_matrix[period_idx, num-1] = appear_count
                last_appear = 0
                for i, draw in enumerate(reversed(history_numbers)):
                    if num in draw:
                        last_appear = i+1
                        break
                miss_count_matrix[period_idx, num-1] = last_appear
        for num in LOTTERY_RULE["number_range"]:
            feature_df[f"近{window}期_热度_{num}"] = hot_count_matrix[:, num-1]
            feature_df[f"近{window}期_遗漏值_{num}"] = miss_count_matrix[:, num-1]
    
    # 2. 重号、相随号、跟随号
    repeat_prob_matrix = np.zeros((total_periods, 80))
    accompany_prob_matrix = np.zeros((total_periods, 80))
    follow_cooccur_matrix = np.zeros((total_periods, 80))

    for period_idx in range(1, total_periods):
        history_df = feature_df.loc[0:period_idx-1, :]
        if len(history_df) < 2:
            continue
        
        repeat_count = defaultdict(int)
        total_repeat_samples = len(history_df) - 1
        for i in range(len(history_df)-1):
            last_draw = history_df.iloc[i]["开奖号码集合"]
            current_draw = history_df.iloc[i+1]["开奖号码集合"]
            for num in last_draw:
                if num in current_draw:
                    repeat_count[num] += 1
        for num in LOTTERY_RULE["number_range"]:
            repeat_prob_matrix[period_idx, num-1] = repeat_count.get(num, 0) / total_repeat_samples if total_repeat_samples > 0 else 0

        accompany_count = defaultdict(lambda: defaultdict(int))
        total_accompany_samples = len(history_df) - 1
        for i in range(len(history_df)-1):
            last_draw = history_df.iloc[i]["开奖号码集合"]
            current_draw = history_df.iloc[i+1]["开奖号码集合"]
            for n in last_draw:
                for m in current_draw:
                    accompany_count[n][m] += 1
        last_period_draw = feature_df.iloc[period_idx-1]["开奖号码集合"]
        for m in LOTTERY_RULE["number_range"]:
            total_accompany = sum(accompany_count[n].get(m, 0) for n in last_period_draw)
            accompany_prob_matrix[period_idx, m-1] = total_accompany / (len(last_period_draw) * total_accompany_samples) if total_accompany_samples > 0 and len(last_period_draw) > 0 else 0

        follow_count = defaultdict(lambda: defaultdict(int))
        total_follow_samples = len(history_df)
        for i in range(len(history_df)):
            current_draw = history_df.iloc[i]["开奖号码集合"]
            draw_list = list(current_draw)
            for i in range(len(draw_list)):
                for j in range(i+1, len(draw_list)):
                    n1, n2 = draw_list[i], draw_list[j]
                    follow_count[n1][n2] += 1
                    follow_count[n2][n1] += 1
        for num in LOTTERY_RULE["number_range"]:
            total_follow = sum(follow_count[num].values())
            follow_cooccur_matrix[period_idx, num-1] = total_follow / (20 * total_follow_samples) if total_follow_samples > 0 else 0

    for num in LOTTERY_RULE["number_range"]:
        feature_df[f"重号概率_{num}"] = repeat_prob_matrix[:, num-1]
        feature_df[f"相随号概率_{num}"] = accompany_prob_matrix[:, num-1]
        feature_df[f"跟随号共现度_{num}"] = follow_cooccur_matrix[:, num-1]

    # 3. 区间特征
    def get_interval(num):
        return (num - 1) // 10 + 1
    for window in rolling_window_list:
        interval_hot_matrix = np.zeros((total_periods, 8))
        for period_idx in range(total_periods):
            start_idx = max(0, period_idx - window)
            history_numbers = feature_df.loc[start_idx:period_idx-1, "开奖号码集合"].tolist()
            if len(history_numbers) == 0:
                continue
            for interval in range(1, 9):
                interval_count = sum(1 for draw in history_numbers for num in draw if get_interval(num) == interval)
                interval_hot_matrix[period_idx, interval-1] = interval_count
        for interval in range(1, 9):
            feature_df[f"近{window}期_区间{interval}_热度"] = interval_hot_matrix[:, interval-1]
    for period_idx in range(total_periods):
        current_draw = feature_df.iloc[period_idx]["开奖号码集合"]
        for interval in range(1, 9):
            feature_df.loc[period_idx, f"本期_区间{interval}_出号个数"] = sum(1 for num in current_draw if get_interval(num) == interval)

    # 4. 奇偶特征
    def is_odd(num):
        return num % 2 == 1
    for window in rolling_window_list:
        odd_hot_matrix = np.zeros((total_periods, 2))
        for period_idx in range(total_periods):
            start_idx = max(0, period_idx - window)
            history_numbers = feature_df.loc[start_idx:period_idx-1, "开奖号码集合"].tolist()
            if len(history_numbers) == 0:
                continue
            odd_count = sum(sum(1 for num in draw if is_odd(num)) for draw in history_numbers)
            even_count = sum(sum(1 for num in draw if not is_odd(num)) for draw in history_numbers)
            total_count = odd_count + even_count
            if total_count > 0:
                odd_hot_matrix[period_idx, 1] = odd_count / total_count
                odd_hot_matrix[period_idx, 0] = even_count / total_count
        feature_df[f"近{window}期_奇数出号概率"] = odd_hot_matrix[:, 1]
        feature_df[f"近{window}期_偶数出号概率"] = odd_hot_matrix[:, 0]
    feature_df["本期_奇数个数"] = feature_df["开奖号码集合"].apply(lambda x: sum(1 for num in x if is_odd(num)))
    feature_df["本期_偶数个数"] = feature_df["开奖号码集合"].apply(lambda x: sum(1 for num in x if not is_odd(num)))

    # 5. 尾号特征
    def get_tail(num):
        return num % 10
    for window in rolling_window_list:
        tail_hot_matrix = np.zeros((total_periods, 10))
        for period_idx in range(total_periods):
            start_idx = max(0, period_idx - window)
            history_numbers = feature_df.loc[start_idx:period_idx-1, "开奖号码集合"].tolist()
            if len(history_numbers) == 0:
                continue
            for tail in range(0, 10):
                tail_count = sum(sum(1 for num in draw if get_tail(num) == tail) for draw in history_numbers)
                tail_hot_matrix[period_idx, tail] = tail_count
        for tail in range(0, 10):
            feature_df[f"近{window}期_尾号{tail}_热度"] = tail_hot_matrix[:, tail]

    # 6. 连号、斜连号
    def get_consecutive_group_count(draw_set):
        sorted_draw = sorted(draw_set)
        group_count = 0
        current_group = 1
        for i in range(1, len(sorted_draw)):
            if sorted_draw[i] == sorted_draw[i-1] + 1:
                current_group += 1
            else:
                if current_group >= 2:
                    group_count += 1
                current_group = 1
        if current_group >= 2:
            group_count += 1
        return group_count
    feature_df["本期_连号组数"] = feature_df["开奖号码集合"].apply(get_consecutive_group_count)
    for window in rolling_window_list:
        oblique_prob_matrix = np.zeros((total_periods, 80))
        for period_idx in range(1, total_periods):
            start_idx = max(0, period_idx - window)
            history_df = feature_df.loc[start_idx:period_idx-1, :]
            if len(history_df) < 2:
                continue
            oblique_count = defaultdict(int)
            total_samples = len(history_df) - 1
            for i in range(len(history_df)-1):
                last_draw = history_df.iloc[i]["开奖号码集合"]
                current_draw = history_df.iloc[i+1]["开奖号码集合"]
                for num in last_draw:
                    if (num + 1) in current_draw or (num - 1) in current_draw:
                        oblique_count[num] += 1
            for num in LOTTERY_RULE["number_range"]:
                oblique_prob_matrix[period_idx, num-1] = oblique_count.get(num, 0) / total_samples if total_samples > 0 else 0
        for num in LOTTERY_RULE["number_range"]:
            feature_df[f"近{window}期_斜连号概率_{num}"] = oblique_prob_matrix[:, num-1]

    # 7. 标签构建
    label_matrix = np.zeros((total_periods, 80))
    for period_idx in range(total_periods-1):
        next_draw = feature_df.iloc[period_idx+1]["开奖号码集合"]
        for num in next_draw:
            label_matrix[period_idx, num-1] = 1
    for num in LOTTERY_RULE["number_range"]:
        feature_df[f"下期是否开出_{num}"] = label_matrix[:, num-1]

    feature_df = feature_df.iloc[1:-1, :].reset_index(drop=True)
    feature_columns = [col for col in feature_df.columns if any(k in col for k in ["热度", "遗漏值", "概率", "共现度", "区间", "奇偶", "尾号", "斜连"])]
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(feature_df[feature_columns])
    valid_feature_columns = feature_df[feature_columns].columns[selector.get_support()].tolist()

    st.session_state["feature_columns"] = valid_feature_columns
    st.session_state["feature_engineered_db"] = feature_df.copy(deep=True)
    return feature_df, valid_feature_columns

# ========== 特征分析 ==========
def feature_analysis(feature_df, feature_columns):
    corr_df = feature_df[feature_columns].corr()
    feature_var = feature_df[feature_columns].var().sort_values(ascending=False)
    return corr_df, feature_var

# ========== 多模型训练 ==========
def train_multi_model(feature_df, feature_columns, model_weight_config=None):
    if model_weight_config is None:
        model_weight_config = {
            "LogisticRegression": 0.15, "RandomForest": 0.2, "XGBoost": 0.35, "LightGBM": 0.3
        }
    total_samples = len(feature_df)
    train_size = int(total_samples * 0.7)
    val_size = int(total_samples * 0.15)
    train_df = feature_df.iloc[0:train_size, :]
    val_df = feature_df.iloc[train_size:train_size+val_size, :]
    test_df = feature_df.iloc[train_size+val_size:, :]

    model_dict = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42, class_weight="balanced"),
        "XGBoost": XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, reg_alpha=1, reg_lambda=1, random_state=42, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, reg_alpha=1, reg_lambda=1, random_state=42, verbose=-1)
    }

    number_model_results = defaultdict(dict)
    model_metrics = defaultdict(list)
    test_predict_results = defaultdict(dict)
    latest_predict_results = defaultdict(dict)

    st.info("多模型训练中...")
    progress_bar = st.progress(0)
    for idx, num in enumerate(LOTTERY_RULE["number_range"]):
        progress_bar.progress((idx+1)/80)
        label_col = f"下期是否开出_{num}"
        X_train, y_train = train_df[feature_columns], train_df[label_col]
        X_val, y_val = val_df[feature_columns], val_df[label_col]
        X_test, y_test = test_df[feature_columns], test_df[label_col]
        X_latest = feature_df.iloc[-1:][feature_columns]

        for model_name, model in model_dict.items():
            if model_name in ["XGBoost", "LightGBM"]:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            else:
                model.fit(X_train, y_train)
            test_pred_prob = model.predict_proba(X_test)[:, 1]
            latest_pred_prob = model.predict_proba(X_latest)[:, 1][0]
            number_model_results[num][model_name] = model
            test_predict_results[num][model_name] = test_pred_prob
            latest_predict_results[num][model_name] = latest_pred_prob
            try:
                model_metrics[model_name].append(roc_auc_score(y_test, test_pred_prob))
            except:
                model_metrics[model_name].append(0.5)
    progress_bar.empty()

    model_metrics_summary = [{"模型名称": k, "平均AUC值": round(np.mean(v), 4), "配置权重": model_weight_config[k]} for k, v in model_metrics.items()]
    model_metrics_df = pd.DataFrame(model_metrics_summary)

    final_predict_list = []
    for num in LOTTERY_RULE["number_range"]:
        weighted_prob = sum(latest_predict_results[num][k] * model_weight_config[k] for k in model_weight_config.keys())
        latest_data = feature_df.iloc[-1]
        rule_weight = 0.3*latest_data[f"重号概率_{num}"] + 0.4*latest_data[f"相随号概率_{num}"] + 0.3*latest_data[f"跟随号共现度_{num}"]
        final_prob = 0.6*weighted_prob + 0.4*rule_weight
        final_predict_list.append({"号码": num, "最终融合概率": final_prob, "多模型加权概率": weighted_prob, "规则加权概率": rule_weight})
    final_predict_df = pd.DataFrame(final_predict_list).sort_values("最终融合概率", ascending=False).reset_index(drop=True)

    hit_summary_list = []
    for period in test_df["期号"].unique():
        period_data = test_df[test_df["期号"] == period]
        period_prob = []
        for num in LOTTERY_RULE["number_range"]:
            real = period_data[f"下期是否开出_{num}"].values[0]
            wp = sum(test_predict_results[num][k][0] * model_weight_config[k] for k in model_weight_config.keys())
            period_prob.append({"号码": num, "真实是否开出": real, "融合预测概率": wp})
        top20_hit = pd.DataFrame(period_prob).sort_values("融合预测概率", ascending=False).head(20)["真实是否开出"].sum()
        hit_summary_list.append({"期号": period, "Top20预测命中个数": top20_hit, "命中率": top20_hit/20})
    hit_summary_df = pd.DataFrame(hit_summary_list)

    st.session_state["multi_model_dict"] = number_model_results
    st.session_state["model_metrics"] = model_metrics_df
    st.session_state["final_predict_result"] = final_predict_df
    st.session_state["multi_model_hit_summary"] = hit_summary_df
    return final_predict_df, model_metrics_df, hit_summary_df

# ========== 单模型训练 ==========
def train_predict_model(feature_df, feature_columns):
    total_samples = len(feature_df)
    train_size = int(total_samples * 0.7)
    val_size = int(total_samples * 0.15)
    train_df = feature_df.iloc[0:train_size, :]
    val_df = feature_df.iloc[train_size:train_size+val_size, :]
    test_df = feature_df.iloc[train_size+val_size:, :]

    model_dict = {}
    predict_result = []
    test_accuracy_detail = []
    st.info("单模型训练中...")
    progress_bar = st.progress(0)
    for idx, num in enumerate(LOTTERY_RULE["number_range"]):
        progress_bar.progress((idx+1)/80)
        label_col = f"下期是否开出_{num}"
        X_train, y_train = train_df[feature_columns], train_df[label_col]
        X_val, y_val = val_df[feature_columns], val_df[label_col]
        X_test, y_test = test_df[feature_columns], test_df[label_col]
        model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, reg_alpha=1, reg_lambda=1, random_state=42, eval_metric="logloss")
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        model_dict[num] = model
        test_pred_prob = model.predict_proba(X_test)[:, 1]
        for i in range(len(test_df)):
            test_accuracy_detail.append({"期号": test_df.iloc[i]["期号"], "号码": num, "真实是否开出": test_df.iloc[i][label_col], "模型预测概率": test_pred_prob[i]})
        latest_prob = model.predict_proba(feature_df.iloc[-1:][feature_columns])[:, 1][0]
        predict_result.append({"号码": num, "模型预测概率": latest_prob, "规则加权概率": 0})
    progress_bar.empty()

    predict_df = pd.DataFrame(predict_result)
    latest_data = feature_df.iloc[-1]
    for num in LOTTERY_RULE["number_range"]:
        rw = 0.3*latest_data[f"重号概率_{num}"] + 0.4*latest_data[f"相随号概率_{num}"] + 0.3*latest_data[f"跟随号共现度_{num}"]
        mp = predict_df.loc[predict_df["号码"]==num, "模型预测概率"].values[0]
        predict_df.loc[predict_df["号码"]==num, ["规则加权概率", "最终融合概率"]] = [rw, 0.6*mp+0.4*rw]
    predict_df = predict_df.sort_values("最终融合概率", ascending=False).reset_index(drop=True)

    hit_summary = []
    for p in test_df["期号"].unique():
        sub = pd.DataFrame([d for d in test_accuracy_detail if d["期号"]==p]).sort_values("模型预测概率", ascending=False).head(20)
        hit_summary.append({"期号": p, "Top20预测命中个数": sub["真实是否开出"].sum(), "命中率": sub["真实是否开出"].sum()/20})
    hit_summary_df = pd.DataFrame(hit_summary)

    st.session_state["model_dict"] = model_dict
    st.session_state["predict_result"] = predict_df
    st.session_state["hit_summary"] = hit_summary_df
    st.session_state["accuracy_detail"] = pd.DataFrame(test_accuracy_detail)
    return predict_df, hit_summary_df, pd.DataFrame(test_accuracy_detail)

# ========== 组合生成 ==========
def generate_low_repeat_combinations(predict_df, select_count=8, group_count=5, max_repeat_rate=0.3):
    top_pool = predict_df.head(select_count*3)["号码"].tolist()
    combos = []
    for _ in range(group_count):
        # 修复语法错误的核心代码
        if not combos:
            cand = top_pool[:select_count]
        else:
            cand = [n for n in top_pool if sum(n in c for c in combos)/len(combos) <= max_repeat_rate][:select_count]
        
        combo = sorted(cand if cand else top_pool[:select_count])
        if combo not in combos:
            combos.append(combo)
    while len(combos) < group_count:
        combos.append(sorted(np.random.choice(top_pool, select_count, replace=False)))
    return pd.DataFrame(combos, columns=[f"号码{i+1}" for i in range(select_count)], index=[f"第{i+1}组" for i in range(len(combos))]) 

def generate_dantuo_combinations(predict_df, dan_count=5, tuo_count=8, group_count=5, max_dan_repeat_rate=0.2, max_tuo_repeat_rate=0.4):
    dan_pool = predict_df.head(dan_count*3)["号码"].tolist()
    tuo_pool = predict_df[~predict_df["号码"].isin(dan_pool)].head(tuo_count*3)["号码"].tolist()
    combos = []
    for _ in range(group_count):
        d = sorted(np.random.choice(dan_pool, dan_count, replace=False))
        t = sorted(np.random.choice([x for x in tuo_pool if x not in d], tuo_count, replace=False))
        combos.append((d,t))
    dt = [{"组别":f"第{i+1}组","胆码":"、".join(map(str,d)),"拖码":"、".join(map(str,t))} for i,(d,t) in enumerate(combos)]
    return pd.DataFrame(dt).set_index("组别"), dan_pool, tuo_pool

# ========== 页面UI ==========
st.title("快乐8数据分析&预测系统V2.0")
st.markdown("---")
df = load_standard_data()

# 侧边栏
with st.sidebar:
    st.header("参数配置")
    uploaded_file = st.file_uploader("上传开奖数据CSV/Excel", type=["csv","xlsx"])
    rolling_windows = st.multiselect("滚动周期", [1,5,10,20,30,50,100], default=[5,10,30])
    lr_weight = st.slider("LR权重",0.0,1.0,0.15,0.05)
    rf_weight = st.slider("RF权重",0.0,1.0,0.2,0.05)
    xgb_weight = st.slider("XGB权重",0.0,1.0,0.35,0.05)
    lgb_weight = st.slider("LGB权重",0.0,1.0,0.3,0.05)
    total_w = lr_weight+rf_weight+xgb_weight+lgb_weight
    model_weight_config = {
        "LogisticRegression":lr_weight/total_w,
        "RandomForest":rf_weight/total_w,
        "XGBoost":xgb_weight/total_w,
        "LightGBM":lgb_weight/total_w
    }
    select_cnt = st.slider("单组号码数",5,20,8)
    group_cnt = st.slider("组合组数",1,20,5)
    max_rep = st.slider("最大重复率",0.1,0.5,0.3,0.05)
    dan_cnt = st.slider("胆码数",1,10,5)
    tuo_cnt = st.slider("拖码数",5,20,8)
    dt_group = st.slider("胆拖组数",1,20,5)
    st.caption("仅供数据分析，不构成购彩建议")

# Tab页
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8 = st.tabs([
    "📊数据底层库","🔍基础特征工程","✨高级特征分析","🤖单模型预测",
    "🚀多模型融合","📈命中率复盘","🎫普通组合","💎胆拖组合"
])

# Tab1
with tab1:
    st.header("底层数据库")
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("原始库")
        st.dataframe(st.session_state.raw_original_db, use_container_width=True)
        st.download_button("下载原始库", st.session_state.raw_original_db.to_csv(index=False,encoding="utf-8-sig"), "原始库.csv", "text/csv")
    with col2:
        st.subheader("标准库")
        st.dataframe(df[["期号","开奖号码集合"]], use_container_width=True)
        st.metric("总期数", len(df))

# Tab2
with tab2:
    st.header("特征工程")
    if st.button("一键生成特征", type="primary"):
        with st.spinner("计算中..."):
            build_feature_engineer(df, rolling_windows)
        st.success("特征生成完成")

# Tab3
with tab3:
    st.header("高级特征分析")
    if st.session_state.get("feature_engineered_db", pd.DataFrame()).empty:
        st.warning("先生成特征")
    else:
        fdf = st.session_state.feature_engineered_db
        cols = st.session_state.feature_columns
        corr, fvar = feature_analysis(fdf, cols)
        st.dataframe(fvar, use_container_width=True)

# Tab4
with tab4:
    st.header("单模型预测")
    if st.button("训练单模型", type="primary") and not st.session_state.feature_engineered_db.empty:
        train_predict_model(st.session_state.feature_engineered_db, st.session_state.feature_columns)
        st.success("训练完成")
        st.dataframe(st.session_state.predict_result, use_container_width=True)

# Tab5
with tab5:
    st.header("多模型融合")
    if st.button("训练多模型", type="primary") and not st.session_state.feature_engineered_db.empty:
        train_multi_model(st.session_state.feature_engineered_db, st.session_state.feature_columns, model_weight_config)
        st.success("融合完成")
        st.dataframe(st.session_state.final_predict_result, use_container_width=True)

# Tab6
with tab6:
    st.header("命中率复盘")
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("单模型")
        if not st.session_state.get("hit_summary", pd.DataFrame()).empty:
            st.line_chart(st.session_state.hit_summary, x="期号", y="Top20预测命中个数")
    with col2:
        st.subheader("多模型")
        if not st.session_state.get("multi_model_hit_summary", pd.DataFrame()).empty:
            st.line_chart(st.session_state.multi_model_hit_summary, x="期号", y="Top20预测命中个数")

# Tab7
with tab7:
    st.header("普通组合")
    src = st.radio("数据源", ["单模型","多模型"], horizontal=True)
    pred = st.session_state.predict_result if src=="单模型" else st.session_state.final_predict_result
    if not pred.empty and st.button("生成组合", type="primary"):
        cdf = generate_low_repeat_combinations(pred, select_cnt, group_cnt, max_rep)
        st.dataframe(cdf, use_container_width=True)

# Tab8
with tab8:
    st.header("胆拖组合")
    src = st.radio("胆拖数据源", ["多模型","单模型"], horizontal=True)
    pred = st.session_state.final_predict_result if src=="多模型" else st.session_state.predict_result
    if not pred.empty and st.button("生成胆拖", type="primary"):
        dtdf,_,_ = generate_dantuo_combinations(pred, dan_cnt, tuo_cnt, dt_group)
        st.dataframe(dtdf, use_container_width=True) 
# ===================== 顶部统一导入（无重复、无遗漏、全兼容） =====================
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from collections import defaultdict

# ===================== 全局配置&异常兜底 =====================
warnings.filterwarnings('ignore')
# 全环境中文显示兼容
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Streamlit页面配置（必须放在所有页面元素之前，仅执行一次）
st.set_page_config(
    page_title="快乐8数据分析预测系统V2.0终版",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== 全局常量定义（严格匹配快乐8官方规则） =====================
LOTTERY_RULE = {
    "total_numbers": 80,
    "draw_per_period": 20,
    "number_range": range(1, 81),
    "interval_count": 8,
    "tail_number_count": 10
}
CUSTOM_RULE = {
    "follow_number": "同期一起出现的号码（跟随号）",
    "accompany_number": "本期开出N，下期开出M（相随号）",
    "repeat_number": "本期开出A，下期继续开出A（重号）"
}

# ===================== 全局Session全量初始化（彻底解决KeyError） =====================
def init_session_state():
    init_keys = [
        "raw_original_db", "26_year_db", "tongqi_db", "re_leng_data_db",
        "feature_columns", "feature_engineered_db", "model_dict", "predict_result",
        "hit_summary", "accuracy_detail", "multi_model_dict", "model_metrics",
        "final_predict_result", "multi_model_hit_summary"
    ]
    for key in init_keys:
        if key not in st.session_state:
            st.session_state[key] = pd.DataFrame()
# 页面加载立即执行初始化
init_session_state()

# ===================== 1. 核心数据加载函数（带格式校验+异常处理） =====================
@st.cache_data(ttl=3600)
def load_standard_data(uploaded_file=None):
    """
    数据加载&标准化：带格式校验，兼容官方开奖数据，无文件时自动生成测试数据
    """
    # 无上传文件：生成标准测试数据
    if uploaded_file is None:
        mock_periods = 200
        mock_data = []
        for period in range(1, mock_periods+1):
            draw_numbers = sorted(np.random.choice(LOTTERY_RULE["number_range"], size=LOTTERY_RULE["draw_per_period"], replace=False))
            mock_data.append([f"2026{str(period).zfill(3)}"] + draw_numbers)
        columns = ["期号"] + [f"开奖号码{i}" for i in range(1, 21)]
        raw_df = pd.DataFrame(mock_data, columns=columns)
        st.info("当前使用测试数据，上传官方CSV/Excel文件即可切换真实数据")
    else:
        # 读取上传文件+异常捕获
        try:
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"文件读取失败：{str(e)}，请检查文件格式")
            return pd.DataFrame()
        
        # 数据格式强制校验
        required_cols = ["期号"] + [f"开奖号码{i}" for i in range(1, 21)]
        missing_cols = [col for col in required_cols if col not in raw_df.columns]
        if missing_cols:
            st.error(f"文件格式错误，缺少必填列：{','.join(missing_cols)}")
            st.caption("必填格式：第一列为【期号】，后续20列为【开奖号码1】到【开奖号码20】")
            return pd.DataFrame()

    # 只读原始库永久存档（全程不修改，保证溯源）
    st.session_state["raw_original_db"] = raw_df.copy(deep=True)
    
    # 衍生分析库标准化处理
    df = raw_df.copy(deep=True)
    df = df.sort_values("期号", ascending=True).reset_index(drop=True)
    # 提取开奖号码集合（统一int类型，去空值）
    df["开奖号码集合"] = df[[f"开奖号码{i}" for i in range(1, 21)]].values.tolist()
    df["开奖号码集合"] = df["开奖号码集合"].apply(lambda x: set([int(num) for num in x if pd.notna(num)]))
    
    return df

# ===================== 2. 全量特征工程函数（补全闭环+防数据泄露+边界处理） =====================
def build_feature_engineer(df, rolling_window_list=[5, 10, 30]):
    """
    全量特征工程：基础特征+高级特征，严格滚动窗口计算，杜绝未来数据泄露
    """
    if df.empty:
        st.warning("无有效数据，无法生成特征")
        return pd.DataFrame(), []
    
    feature_df = df.copy(deep=True)
    total_periods = len(feature_df)
    
    # -------------------------- 1. 冷热号、遗漏值核心特征 --------------------------
    for window in rolling_window_list:
        hot_count_matrix = np.zeros((total_periods, LOTTERY_RULE["total_numbers"]))
        miss_count_matrix = np.zeros((total_periods, LOTTERY_RULE["total_numbers"]))
        
        for period_idx in range(total_periods):
            # 核心防泄露：仅用当期之前的历史数据，绝对不用未来数据
            start_idx = max(0, period_idx - window)
            history_numbers = feature_df.loc[start_idx:period_idx-1, "开奖号码集合"].tolist()
            if len(history_numbers) == 0:
                continue
            
            for num in LOTTERY_RULE["number_range"]:
                # 热度计算
                appear_count = sum([1 for draw in history_numbers if num in draw])
                hot_count_matrix[period_idx, num-1] = appear_count
                # 遗漏值计算
                last_appear = 0
                for i, draw in enumerate(reversed(history_numbers)):
                    if num in draw:
                        last_appear = i+1
                        break
                miss_count_matrix[period_idx, num-1] = last_appear
        
        # 特征存入DataFrame
        for num in LOTTERY_RULE["number_range"]:
            feature_df[f"近{window}期_热度_{num}"] = hot_count_matrix[:, num-1]
            feature_df[f"近{window}期_遗漏值_{num}"] = miss_count_matrix[:, num-1]
    
    # -------------------------- 2. 自定义规则特征（重号/相随号/跟随号） --------------------------
    repeat_prob_matrix = np.zeros((total_periods, LOTTERY_RULE["total_numbers"]))
    accompany_prob_matrix = np.zeros((total_periods, LOTTERY_RULE["total_numbers"]))
    follow_cooccur_matrix = np.zeros((total_periods, LOTTERY_RULE["total_numbers"]))

    for period_idx in range(1, total_periods):
        history_df = feature_df.loc[0:period_idx-1, :]
        if len(history_df) < 2:
            continue
        
        # 2.1 重号概率统计
        repeat_count = defaultdict(int)
        total_repeat_samples = len(history_df) - 1
        for i in range(len(history_df)-1):
            last_draw = history_df.iloc[i]["开奖号码集合"]
            current_draw = history_df.iloc[i+1]["开奖号码集合"]
            for num in last_draw:
                if num in current_draw:
                    repeat_count[num] += 1
        for num in LOTTERY_RULE["number_range"]:
            repeat_prob_matrix[period_idx, num-1] = repeat_count.get(num, 0) / total_repeat_samples if total_repeat_samples > 0 else 0

        # 2.2 相随号概率统计
        accompany_count = defaultdict(lambda: defaultdict(int))
        total_accompany_samples = len(history_df) - 1
        for i in range(len(history_df)-1):
            last_draw = history_df.iloc[i]["开奖号码集合"]
            current_draw = history_df.iloc[i+1]["开奖号码集合"]
            for n in last_draw:
                for m in current_draw:
                    accompany_count[n][m] += 1
        last_period_draw = feature_df.iloc[period_idx-1]["开奖号码集合"]
        for m in LOTTERY_RULE["number_range"]:
            total_accompany = sum([accompany_count[n].get(m, 0) for n in last_period_draw])
            accompany_prob_matrix[period_idx, m-1] = total_accompany / (len(last_period_draw) * total_accompany_samples) if total_accompany_samples > 0 and len(last_period_draw) > 0 else 0

        # 2.3 跟随号共现度统计（补全之前断掉的代码）
        follow_count = defaultdict(lambda: defaultdict(int))
        total_follow_samples = len(history_df)
        for i in range(len(history_df)):
            current_draw = history_df.iloc[i]["开奖号码集合"]
            draw_list = list(current_draw)
            for x in range(len(draw_list)):
                for y in range(x+1, len(draw_list)):
                    n1, n2 = draw_list[x], draw_list[y]
                    follow_count[n1][n2] += 1
                    follow_count[n2][n1] += 1
        for num in LOTTERY_RULE["number_range"]:
            total_follow = sum(follow_count[num].values())
            follow_cooccur_matrix[period_idx, num-1] = total_follow / (LOTTERY_RULE["draw_per_period"] * total_follow_samples) if total_follow_samples > 0 else 0

    # 自定义规则特征存入DataFrame
    for num in LOTTERY_RULE["number_range"]:
        feature_df[f"重号概率_{num}"] = repeat_prob_matrix[:, num-1]
        feature_df[f"相随号概率_{num}"] = accompany_prob_matrix[:, num-1]
        feature_df[f"跟随号共现度_{num}"] = follow_cooccur_matrix[:, num-1]

    # -------------------------- 3. 高级特征1：区间分布特征 --------------------------
    def get_interval(num):
        return (num - 1) // 10 + 1
    for window in rolling_window_list:
        interval_hot_matrix = np.zeros((total_periods, LOTTERY_RULE["interval_count"]))
        for period_idx in range(total_periods):
            start_idx = max(0, period_idx - window)
            history_numbers = feature_df.loc[start_idx:period_idx-1, "开奖号码集合"].tolist()
            if len(history_numbers) == 0:
                continue
            for interval in range(1, LOTTERY_RULE["interval_count"]+1):
                interval_count = sum([1 for draw in history_numbers for num in draw if get_interval(num) == interval])
                interval_hot_matrix[period_idx, interval-1] = interval_count
        for interval in range(1, LOTTERY_RULE["interval_count"]+1):
            feature_df[f"近{window}期_区间{interval}_热度"] = interval_hot_matrix[:, interval-1]
    # 每期区间出号个数
    for period_idx in range(total_periods):
        current_draw = feature_df.iloc[period_idx]["开奖号码集合"]
        for interval in range(1, LOTTERY_RULE["interval_count"]+1):
            feature_df.loc[period_idx, f"本期_区间{interval}_出号个数"] = sum([1 for num in current_draw if get_interval(num) == interval])

    # -------------------------- 4. 高级特征2：奇偶比特征 --------------------------
    def is_odd(num):
        return num % 2 == 1
    for window in rolling_window_list:
        odd_hot_matrix = np.zeros((total_periods, 2))
        for period_idx in range(total_periods):
            start_idx = max(0, period_idx - window)
            history_numbers = feature_df.loc[start_idx:period_idx-1, "开奖号码集合"].tolist()
            if len(history_numbers) == 0:
                continue
            odd_count = sum([sum([1 for num in draw if is_odd(num)]) for draw in history_numbers])
            even_count = sum([sum([1 for num in draw if not is_odd(num)]) for draw in history_numbers])
            total_count = odd_count + even_count
            if total_count > 0:
                odd_hot_matrix[period_idx, 1] = odd_count / total_count
                odd_hot_matrix[period_idx, 0] = even_count / total_count
        feature_df[f"近{window}期_奇数出号概率"] = odd_hot_matrix[:, 1]
        feature_df[f"近{window}期_偶数出号概率"] = odd_hot_matrix[:, 0]
    # 每期奇偶出号个数
    feature_df["本期_奇数个数"] = feature_df["开奖号码集合"].apply(lambda x: sum([1 for num in x if is_odd(num)]))
    feature_df["本期_偶数个数"] = feature_df["开奖号码集合"].apply(lambda x: sum([1 for num in x if not is_odd(num)]))

    # -------------------------- 5. 高级特征3：同尾号特征 --------------------------
    def get_tail(num):
        return num % 10
    for window in rolling_window_list:
        tail_hot_matrix = np.zeros((total_periods, LOTTERY_RULE["tail_number_count"]))
        for period_idx in range(total_periods):
            start_idx = max(0, period_idx - window)
            history_numbers = feature_df.loc[start_idx:period_idx-1, "开奖号码集合"].tolist()
            if len(history_numbers) == 0:
                continue
            for tail in range(0, LOTTERY_RULE["tail_number_count"]):
                tail_count = sum([sum([1 for num in draw if get_tail(num) == tail]) for draw in history_numbers])
                tail_hot_matrix[period_idx, tail] = tail_count
        for tail in range(0, LOTTERY_RULE["tail_number_count"]):
            feature_df[f"近{window}期_尾号{tail}_热度"] = tail_hot_matrix[:, tail]

    # -------------------------- 6. 高级特征4：连号&斜连号特征 --------------------------
    def get_consecutive_group_count(draw_set):
        sorted_draw = sorted(draw_set)
        group_count = 0
        current_group = 1
        for i in range(1, len(sorted_draw)):
            if sorted_draw[i] == sorted_draw[i-1] + 1:
                current_group += 1
            else:
                if current_group >= 2:
                    group_count += 1
                current_group = 1
        if current_group >= 2:
            group_count += 1
        return group_count
    feature_df["本期_连号组数"] = feature_df["开奖号码集合"].apply(get_consecutive_group_count)
    # 斜连号特征
    for window in rolling_window_list:
        oblique_prob_matrix = np.zeros((total_periods, LOTTERY_RULE["total_numbers"]))
        for period_idx in range(1, total_periods):
            start_idx = max(0, period_idx - window)
            history_df = feature_df.loc[start_idx:period_idx-1, :]
            if len(history_df) < 2:
                continue
            oblique_count = defaultdict(int)
            total_samples = len(history_df) - 1
            for i in range(len(history_df)-1):
                last_draw = history_df.iloc[i]["开奖号码集合"]
                current_draw = history_df.iloc[i+1]["开奖号码集合"]
                for num in last_draw:
                    if (num + 1) in current_draw or (num - 1) in current_draw:
                        oblique_count[num] += 1
            for num in LOTTERY_RULE["number_range"]:
                oblique_prob_matrix[period_idx, num-1] = oblique_count.get(num, 0) / total_samples if total_samples > 0 else 0
        for num in LOTTERY_RULE["number_range"]:
            feature_df[f"近{window}期_斜连号概率_{num}"] = oblique_prob_matrix[:, num-1]

    # -------------------------- 7. 标签构建（用于模型训练） --------------------------
    label_matrix = np.zeros((total_periods, LOTTERY_RULE["total_numbers"]))
    for period_idx in range(total_periods-1):
        next_draw = feature_df.iloc[period_idx+1]["开奖号码集合"]
        for num in next_draw:
            label_matrix[period_idx, num-1] = 1
    for num in LOTTERY_RULE["number_range"]:
        feature_df[f"下期是否开出_{num}"] = label_matrix[:, num-1]

    # -------------------------- 8. 无效数据剔除&特征筛选 --------------------------
    # 剔除无历史数据的首行、无标签的末行
    feature_df = feature_df.iloc[1:-1, :].reset_index(drop=True)
    # 筛选有效特征（剔除无方差的无效特征）
    feature_columns = [col for col in feature_df.columns if any(keyword in col for keyword in ["热度", "遗漏值", "概率", "共现度", "区间", "奇偶", "尾号", "斜连"])]
    if len(feature_columns) > 0:
        selector = VarianceThreshold(threshold=0.01)
        selector.fit(feature_df[feature_columns])
        valid_feature_columns = feature_df[feature_columns].columns[selector.get_support()].tolist()
    else:
        valid_feature_columns = []
        st.warning("未生成有效特征，请检查数据")

    # 存入全局会话，全页面可调用
    st.session_state["feature_columns"] = valid_feature_columns
    st.session_state["feature_engineered_db"] = feature_df.copy(deep=True)
    
    return feature_df, valid_feature_columns

# ===================== 3. 特征分析配套函数 =====================
def feature_analysis(feature_df, feature_columns):
    """特征贡献度、相关性分析"""
    if feature_df.empty or len(feature_columns) == 0:
        return pd.DataFrame(), pd.Series()
    corr_df = feature_df[feature_columns].corr()
    feature_var = feature_df[feature_columns].var().sort_values(ascending=False)
    return corr_df, feature_var

# ===================== 4. 单模型训练&预测函数 =====================
def train_predict_model(feature_df, feature_columns):
    """单模型训练（XGBoost基准模型），带进度条+异常处理"""
    if feature_df.empty or len(feature_columns) == 0:
        st.warning("无有效特征数据，无法训练模型")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    total_samples = len(feature_df)
    train_size = int(total_samples * 0.7)
    val_size = int(total_samples * 0.15)
    
    # 时序拆分（杜绝随机拆分过拟合）
    train_df = feature_df.iloc[0:train_size, :]
    val_df = feature_df.iloc[train_size:train_size+val_size, :]
    test_df = feature_df.iloc[train_size+val_size:, :]
    
    model_dict = {}
    predict_result = []
    test_accuracy_detail = []
    
    st.info("单模型训练中，预计10-30秒完成...")
    progress_bar = st.progress(0)
    
    for idx, num in enumerate(LOTTERY_RULE["number_range"]):
        progress_bar.progress((idx+1)/LOTTERY_RULE["total_numbers"])
        label_col = f"下期是否开出_{num}"
        
        # 数据集拆分
        X_train = train_df[feature_columns]
        y_train = train_df[label_col]
        X_val = val_df[feature_columns]
        y_val = val_df[label_col]
        X_test = test_df[feature_columns]
        y_test = test_df[label_col]
        
        # 模型初始化&训练
        model = XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            reg_alpha=1, reg_lambda=1, random_state=42, eval_metric="logloss"
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        model_dict[num] = model
        
        # 测试集预测
        test_pred_prob = model.predict_proba(X_test)[:, 1]
        for i in range(len(test_df)):
            test_accuracy_detail.append({
                "期号": test_df.iloc[i]["期号"],
                "号码": num,
                "真实是否开出": test_df.iloc[i][label_col],
                "模型预测概率": test_pred_prob[i]
            })
        
        # 最新一期预测
        latest_feature = feature_df.iloc[-1:][feature_columns]
        latest_prob = model.predict_proba(latest_feature)[:, 1][0]
        predict_result.append({
            "号码": num,
            "模型预测概率": latest_prob,
            "规则加权概率": 0
        })
    
    progress_bar.empty()
    
    # 规则加权融合
    predict_df = pd.DataFrame(predict_result)
    latest_period_data = feature_df.iloc[-1]
    for num in LOTTERY_RULE["number_range"]:
        rule_weight = (
            0.3 * latest_period_data[f"重号概率_{num}"] +
            0.4 * latest_period_data[f"相随号概率_{num}"] +
            0.3 * latest_period_data[f"跟随号共现度_{num}"]
        )
        model_prob = predict_df.loc[predict_df["号码"] == num, "模型预测概率"].values[0]
        final_prob = 0.6 * model_prob + 0.4 * rule_weight
        predict_df.loc[predict_df["号码"] == num, "规则加权概率"] = rule_weight
        predict_df.loc[predict_df["号码"] == num, "最终融合概率"] = final_prob
    
    # 按最终概率排序
    predict_df = predict_df.sort_values("最终融合概率", ascending=False).reset_index(drop=True)
    
    # 命中率复盘统计
    accuracy_df = pd.DataFrame(test_accuracy_detail)
    test_period_list = test_df["期号"].unique()
    hit_count_list = []
    for period in test_period_list:
        period_data = accuracy_df[accuracy_df["期号"] == period]
        top20_pred = period_data.sort_values("模型预测概率", ascending=False).head(20)
        hit_count = top20_pred["真实是否开出"].sum()
        hit_count_list.append({
            "期号": period,
            "Top20预测命中个数": hit_count,
            "命中率": hit_count / LOTTERY_RULE["draw_per_period"]
        })
    hit_summary_df = pd.DataFrame(hit_count_list)
    
    # 存入全局会话
    st.session_state["model_dict"] = model_dict
    st.session_state["predict_result"] = predict_df
    st.session_state["hit_summary"] = hit_summary_df
    st.session_state["accuracy_detail"] = accuracy_df
    
    return predict_df, hit_summary_df, accuracy_df

# ===================== 5. 多模型融合训练&预测函数 =====================
def train_multi_model(feature_df, feature_columns, model_weight_config=None):
    """四模型融合训练（LR+RF+XGBoost+LightGBM），带权重配置+异常处理"""
    if feature_df.empty or len(feature_columns) == 0:
        st.warning("无有效特征数据，无法训练模型")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # 默认权重配置
    if model_weight_config is None:
        model_weight_config = {
            "LogisticRegression": 0.15,
            "RandomForest": 0.2,
            "XGBoost": 0.35,
            "LightGBM": 0.3
        }
    
    # 时序拆分（防过拟合）
    total_samples = len(feature_df)
    train_size = int(total_samples * 0.7)
    val_size = int(total_samples * 0.15)
    train_df = feature_df.iloc[0:train_size, :]
    val_df = feature_df.iloc[train_size:train_size+val_size, :]
    test_df = feature_df.iloc[train_size+val_size:, :]
    
    # 模型字典
    model_dict = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42, class_weight="balanced"),
        "XGBoost": XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, reg_alpha=1, reg_lambda=1, random_state=42, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, reg_alpha=1, reg_lambda=1, random_state=42, verbose=-1)
    }
    
    # 结果存储
    number_model_results = defaultdict(dict)
    model_metrics = defaultdict(list)
    test_predict_results = defaultdict(dict)
    latest_predict_results = defaultdict(dict)
    
    st.info("多模型训练中，预计30-60秒完成...")
    progress_bar = st.progress(0)
    
    # 每个号码单独训练
    for idx, num in enumerate(LOTTERY_RULE["number_range"]):
        progress_bar.progress((idx+1)/LOTTERY_RULE["total_numbers"])
        label_col = f"下期是否开出_{num}"
        
        X_train = train_df[feature_columns]
        y_train = train_df[label_col]
        X_val = val_df[feature_columns]
        y_val = val_df[label_col]
        X_test = test_df[feature_columns]
        y_test = test_df[label_col]
        X_latest = feature_df.iloc[-1:][feature_columns]
        
        # 逐个模型训练
        for model_name, model in model_dict.items():
            if model_name in ["XGBoost", "LightGBM"]:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            else:
                model.fit(X_train, y_train)
            
            test_pred_prob = model.predict_proba(X_test)[:, 1]
            latest_pred_prob = model.predict_proba(X_latest)[:, 1][0]
            
            number_model_results[num][model_name] = model
            test_predict_results[num][model_name] = test_pred_prob
            latest_predict_results[num][model_name] = latest_pred_prob
            
            # 计算AUC指标
            try:
                auc_score = roc_auc_score(y_test, test_pred_prob)
                model_metrics[model_name].append(auc_score)
            except:
                model_metrics[model_name].append(0.5)
    
    progress_bar.empty()
    
    # 模型指标汇总
    model_metrics_summary = []
    for model_name in model_dict.keys():
        avg_auc = np.mean(model_metrics[model_name])
        model_metrics_summary.append({
            "模型名称": model_name,
            "平均AUC值": round(avg_auc, 4),
            "配置权重": model_weight_config[model_name]
        })
    model_metrics_df = pd.DataFrame(model_metrics_summary)
    
    # 多模型加权融合预测
    final_predict_list = []
    for num in LOTTERY_RULE["number_range"]:
        # 单模型概率加权求和
        weighted_prob = 0
        model_prob_detail = {}
        for model_name in model_dict.keys():
            prob = latest_predict_results[num][model_name]
            model_prob_detail[model_name] = prob
            weighted_prob += prob * model_weight_config[model_name]
        
        # 规则加权融合（模型60% + 自定义规则40%）
        latest_period_data = feature_df.iloc[-1]
        rule_weight = (
            0.3 * latest_period_data[f"重号概率_{num}"] +
            0.4 * latest_period_data[f"相随号概率_{num}"] +
            0.3 * latest_period_data[f"跟随号共现度_{num}"]
        )
        final_prob = 0.6 * weighted_prob + 0.4 * rule_weight
        
        final_predict_list.append({
            "号码": num,
            "最终融合概率": final_prob,
            "多模型加权概率": weighted_prob,
            "规则加权概率": rule_weight,
            **model_prob_detail
        })
    
    # 按最终概率降序排序
    final_predict_df = pd.DataFrame(final_predict_list).sort_values("最终融合概率", ascending=False).reset_index(drop=True)
    
    # 测试集命中率复盘
    test_period_list = test_df["期号"].unique()
    hit_summary_list = []
    for period_idx, period in enumerate(test_period_list):
        period_number_prob = []
        for num in LOTTERY_RULE["number_range"]:
            real_label = test_df[test_df["期号"] == period][f"下期是否开出_{num}"].values[0]
            test_weighted_prob = 0
            for model_name in model_dict.keys():
                test_weighted_prob += test_predict_results[num][model_name][period_idx] * model_weight_config[model_name]
            period_number_prob.append({
                "号码": num,
                "真实是否开出": real_label,
                "融合预测概率": test_weighted_prob
            })
        period_number_df = pd.DataFrame(period_number_prob).sort_values("融合预测概率", ascending=False)
        top20_hit = period_number_df.head(20)["真实是否开出"].sum()
        hit_summary_list.append({
            "期号": period,
            "Top20预测命中个数": top20_hit,
            "命中率": top20_hit / LOTTERY_RULE["draw_per_period"]
        })
    hit_summary_df = pd.DataFrame(hit_summary_list)
    
    # 存入全局会话
    st.session_state["multi_model_dict"] = number_model_results
    st.session_state["model_metrics"] = model_metrics_df
    st.session_state["final_predict_result"] = final_predict_df
    st.session_state["multi_model_hit_summary"] = hit_summary_df
    
    return final_predict_df, model_metrics_df, hit_summary_df

# ===================== 6. 低重复率普通组合生成函数 =====================
def generate_low_repeat_combinations(predict_df, select_count=8, group_count=5, max_repeat_rate=0.3):
    """生成低重复率普通打票组合，支持自定义选号个数、组数、重复率"""
    if predict_df.empty:
        st.warning("无预测结果，无法生成组合")
        return pd.DataFrame()
    
    top_pool = predict_df.head(select_count * 3)["号码"].tolist()
    combinations = []
    
    for i in range(group_count):
        if i == 0:
            combo = sorted(top_pool[:select_count])
        else:
            candidate_combo = []
            for num in top_pool:
                repeat_count = 0
                for exist_combo in combinations:
                    if num in exist_combo:
                        repeat_count += 1
                avg_repeat = repeat_count / len(combinations) if len(combinations) > 0 else 0
                if avg_repeat <= max_repeat_rate:
                    candidate_combo.append(num)
                if len(candidate_combo) == select_count:
                    break
            combo = sorted(candidate_combo) if candidate_combo else sorted(top_pool[:select_count])
        
        if combo not in combinations:
            combinations.append(combo)
    
    # 补全不足的组数
    while len(combinations) < group_count:
        backup_combo = sorted(np.random.choice(top_pool, size=select_count, replace=False))
        if backup_combo not in combinations:
            combinations.append(backup_combo)
    
    # 转换为DataFrame
    combo_df = pd.DataFrame(combinations, columns=[f"号码{i+1}" for i in range(select_count)])
    combo_df.index = [f"第{i+1}组" for i in range(len(combo_df))]
    
    return combo_df

# ===================== 7. 胆拖组合生成函数 =====================
def generate_dantuo_combinations(predict_df, dan_count=5, tuo_count=8, group_count=5, max_dan_repeat_rate=0.2, max_tuo_repeat_rate=0.4):
    """生成低重复率胆拖组合，适配5胆N拖、8+8等核心场景"""
    if predict_df.empty:
        st.warning("无预测结果，无法生成胆拖组合")
        return pd.DataFrame(), [], []
    
    # 胆码池：Top高概率号码
    dan_pool = predict_df.head(dan_count * 3)["号码"].tolist()
    # 拖码池：次高概率号码（排除胆码池）
    tuo_pool = predict_df[~predict_df["号码"].isin(dan_pool)].head(tuo_count * 3)["号码"].tolist()
    
    combinations = []
    
    for i in range(group_count):
        # 生成胆码
        if i == 0:
            current_dan = sorted(dan_pool[:dan_count])
        else:
            current_dan = []
            for num in dan_pool:
                repeat_count = 0
                for exist_dan, _ in combinations:
                    if num in exist_dan:
                        repeat_count += 1
                avg_repeat = repeat_count / len(combinations) if len(combinations) > 0 else 0
                if avg_repeat <= max_dan_repeat_rate:
                    current_dan.append(num)
                if len(current_dan) == dan_count:
                    break
            current_dan = sorted(current_dan) if current_dan else sorted(dan_pool[:dan_count])
        
        # 生成拖码
        current_tuo = []
        for num in tuo_pool:
            if num in current_dan:
                continue
            repeat_count = 0
            for _, exist_tuo in combinations:
                if num in exist_tuo:
                    repeat_count += 1
            avg_repeat = repeat_count / len(combinations) if len(combinations) > 0 else 0
            if avg_repeat <= max_tuo_repeat_rate:
                current_tuo.append(num)
            if len(current_tuo) == tuo_count:
                break
        current_tuo = sorted(current_tuo)
        
        # 补全不足的拖码
        while len(current_tuo) < tuo_count:
            backup_num = np.random.choice([num for num in tuo_pool if num not in current_dan and num not in current_tuo])
            current_tuo.append(backup_num)
            current_tuo = sorted(current_tuo)
        
        # 去重
        combo_tuple = (tuple(current_dan), tuple(current_tuo))
        if combo_tuple not in combinations:
            combinations.append(combo_tuple)
    
    # 补全不足的组数
    while len(combinations) < group_count:
        backup_dan = sorted(np.random.choice(dan_pool, size=dan_count, replace=False))
        backup_tuo = sorted(np.random.choice([num for num in tuo_pool if num not in backup_dan], size=tuo_count, replace=False))
        backup_tuple = (tuple(backup_dan), tuple(backup_tuo))
        if backup_tuple not in combinations:
            combinations.append(backup_tuple)
    
    # 转换为DataFrame
    combo_data = []
    for idx, (dan, tuo) in enumerate(combinations):
        combo_data.append({
            "组别": f"第{idx+1}组",
            "胆码": "、".join([str(num) for num in dan]),
            "拖码": "、".join([str(num) for num in tuo]),
            "胆码个数": len(dan),
            "拖码个数": len(tuo)
        })
    dantuo_df = pd.DataFrame(combo_data).set_index("组别")
    
    return dantuo_df, dan_pool, tuo_pool

# ===================== 8. Streamlit完整页面布局（全Tab补全+交互闭环） =====================
 def build_streamlit_page():
     """构建完整Streamlit页面，全功能交互闭环"""
     st.title("快乐8数据分析&预测系统V2.0终版 | 全功能闭环")
     st.markdown("---")
     # 侧边栏参数配置
     with st.sidebar:
         st.header("⚙️ 参数配置中心")
         st.markdown("### 1. 数据上传")
         uploaded_file = st.file_uploader("上传官方开奖数据（CSV/Excel）", type=["csv", "xlsx"])
         st.caption("必填格式：第一列为【期号】，后续20列为【开奖号码1-20】")
         
         st.markdown("---")
         st.markdown("### 2. 特征工程参数")
         rolling_windows = st.multiselect(
             "滚动统计周期（期）",
             options=[1, 5, 10, 20, 30, 50, 100],
             default=[5, 10, 30]
         )
         
         st.markdown("---")
         st.markdown("### 3. 多模型权重配置")
         st.caption("权重总和自动归一化")
         lr_weight = st.slider("逻辑回归权重", 0.0, 1.0, 0.15, 0.05)
         rf_weight = st.slider("随机森林权重", 0.0, 1.0, 0.2, 0.05)
         xgb_weight = st.slider("XGBoost权重", 0.0, 1.0, 0.35, 0.05)
         lgb_weight = st.slider("LightGBM权重", 0.0, 1.0, 0.3, 0.05)
         # 权重自动归一化
         total_weight = lr_weight + rf_weight + xgb_weight + lgb_weight
         model_weight_config = {
             "LogisticRegression": lr_weight / total_weight,
             "RandomForest": rf_weight / total_weight,
             "XGBoost": xgb_weight / total_weight,
             "LightGBM": lgb_weight / total_weight
         }
         
         st.markdown("---")
         st.markdown("### 4. 普通组合参数")
         select_number_count = st.slider("单组选号个数", min_value=5, max_value=20, value=8)
         generate_group_count = st.slider("生成组合组数", min_value=1, max_value=20, value=5)
         max_repeat_rate = st.slider("组间最大重复率", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
         
         st.markdown("---")
         st.markdown("### 5. 胆拖组合参数")
         dan_count = st.slider("胆码个数", min_value=1, max_value=10, value=5)
         tuo_count = st.slider("拖码个数", min_value=5, max_value=20, value=8)
         dantuo_group_count = st.slider("胆拖组合组数", min_value=1, max_value=20, value=5)
         max_dan_repeat = st.slider("胆码最大重复率", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
         max_tuo_repeat = st.slider("拖码最大重复率", min_value=0.2, max_value=0.7, value=0.4, step=0.05)
         
         st.markdown("---")
         st.markdown("### 6. 免责声明")
         st.caption("本系统仅为数据分析工具，彩票开奖为独立随机事件，不构成任何购彩建议，理性购彩，量力而行")
     # 核心数据加载
     df = load_standard_data(uploaded_file)
     # 主页面Tab布局
     tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
         "📊 数据底层库", "🔍 基础特征工程", "✨ 高级特征分析",
         "🤖 单模型预测", "🚀 多模型融合优化", "📈 命中率复盘",
         "🎫 普通打票组合", "💎 胆拖组合生成"
     ])
     # -------------------- Tab1：数据底层库管理 --------------------
     with tab1:
         st.header("📊 底层数据库管理（原始库只读不可修改）")
         if not df.empty:
             col1, col2 = st.columns(2)
             with col1:
                 st.subheader("只读原始底层库")
                 st.dataframe(st.session_state["raw_original_db"], use_container_width=True, height=400)
                 # 原始库下载
                 st.download_button(
                     label="📥 下载原始底层库CSV",
                     data=st.session_state["raw_original_db"].to_csv(index=False, encoding="utf-8-sig"),
                     file_name="快乐8_原始开奖底层库.csv",
                     mime="text/csv",
                     use_container_width=True
                 )
             with col2:
                 st.subheader("标准化分析库")
                 st.dataframe(df[["期号", "开奖号码集合"]], use_container_width=True, height=400)
                 metric_col1, metric_col2 = st.columns(2)
                 with metric_col1:
                     st.metric("有效历史期数", len(df))
                 with metric_col2:
                     st.metric("最新期号", df["期号"].iloc[-1])
         else:
             st.warning("无有效数据，请检查上传文件")
     # -------------------- Tab2：基础特征工程 --------------------
     with tab2:
         st.header("🔍 基础特征工程中心（自定义规则+防数据泄露）")
         if df.empty:
             st.warning("请先在【数据底层库】上传有效数据")
         else:
             if st.button("✅ 一键生成全量特征（含高级特征）", type="primary", use_container_width=True):
                 with st.spinner("特征工程计算中，严格滚动窗口防数据泄露..."):
                     feature_df, valid_feature_columns = build_feature_engineer(df, rolling_window_list=rolling_windows)
                 
                 if not feature_df.empty:
                     st.success(f"全量特征生成完成！有效特征数量：{len(valid_feature_columns)} 个")
                     col1, col2 = st.columns(2)
                     with col1:
                         st.subheader("特征数据集总览")
                         st.dataframe(feature_df, use_container_width=True)
                         metric_col1, metric_col2 = st.columns(2)
                         with metric_col1:
                             st.metric("有效特征数量", len(valid_feature_columns))
                         with metric_col2:
                             st.metric("有效样本量", len(feature_df))
                     with col2:
                         st.subheader("核心有效特征列表")
                         st.write(valid_feature_columns)
                     
                     # 特征库下载
                     st.download_button(
                         label="📥 下载全量特征工程底层库CSV",
                         data=feature_df.to_csv(index=False, encoding="utf-8-sig"),
                         file_name="快乐8_全量特征工程底层库.csv",
                         mime="text/csv",
                         use_container_width=True
                     )
     # -------------------- Tab3：高级特征分析 --------------------
     with tab3:
         st.header("✨ 高级特征分析中心")
         feature_df = st.session_state["feature_engineered_db"]
         feature_columns = st.session_state["feature_columns"]
         if feature_df.empty or len(feature_columns) == 0:
             st.warning("请先在【基础特征工程】完成全量特征生成")
         else:
             # 特征分析
             corr_df, feature_var = feature_analysis(feature_df, feature_columns)
             
             col1, col2 = st.columns(2)
             with col1:
                 st.subheader("特征方差排名（区分度从高到低）")
                 st.dataframe(feature_var, use_container_width=True, column_config={"value": "特征方差"})
             with col2:
                 st.subheader("区间热度统计（最新一期）")
                 interval_cols = [col for col in feature_df.columns if "近30期_区间" in col and "热度" in col]
                 if len(interval_cols) > 0:
                     latest_interval_hot = feature_df[interval_cols].iloc[-1].sort_values(ascending=False)
                     st.bar_chart(latest_interval_hot, use_container_width=True)
             
             st.markdown("---")
             st.subheader("特征相关性热力图（Top20高区分度特征）")
             top20_features = feature_var.head(20).index.tolist()
             top20_corr = feature_df[top20_features].corr()
             # 绘制热力图
             fig, ax = plt.subplots(figsize=(12, 8))
             sns.heatmap(top20_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax, fmt=".2f")
             st.pyplot(fig)
             
             # 高级特征下载
             st.download_button(
                 label="📥 下载高级特征明细CSV",
                 data=feature_df[[col for col in feature_df.columns if any(keyword in col for keyword in ["区间", "奇偶", "尾号", "连号", "斜连"])]].to_csv(index=False, encoding="utf-8-sig"),
                 file_name="快乐8_高级特征明细.csv",
                 mime="text/csv",
                 use_container_width=True
             )
     # -------------------- Tab4：单模型预测 --------------------
     with tab4:
         st.header("🤖 单模型预测中心（XGBoost基准模型）")
         feature_df = st.session_state["feature_engineered_db"]
         feature_columns = st.session_state["feature_columns"]
         if feature_df.empty or len(feature_columns) == 0:
             st.warning("请先在【基础特征工程】完成特征生成")
         else:
             if st.button("✅ 一键训练单模型&生成预测", type="primary", use_container_width=True):
                 with st.spinner("模型训练&预测中，防过拟合优化..."):
                     predict_df, hit_summary_df, accuracy_df = train_predict_model(feature_df, feature_columns)
                 
                 if not predict_df.empty:
                     st.success("单模型预测完成！")
                     st.subheader("下期号码预测排名（按最终融合概率降序）")
                     
                     col1, col2 = st.columns(2)
                     with col1:
                         st.dataframe(predict_df, use_container_width=True)
                     with col2:
                         st.subheader("Top20高概率推荐号码")
                         top20_numbers = predict_df.head(20)["号码"].tolist()
                         st.markdown(f"### {sorted(top20_numbers)}")
                     
                     # 预测结果下载
                     st.download_button(
                         label="📥 下载单模型预测结果CSV",
                         data=predict_df.to_csv(index=False, encoding="utf-8-sig"),
                         file_name=f"快乐8_{df['期号'].iloc[-1]}期_单模型预测号码池.csv",
                         mime="text/csv",
                         use_container_width=True
                     )
     # -------------------- Tab5：多模型融合优化 --------------------
     with tab5:
         st.header("🚀 多模型融合优化中心")
         feature_df = st.session_state["feature_engineered_db"]
         feature_columns = st.session_state["feature_columns"]
         if feature_df.empty or len(feature_columns) == 0:
             st.warning("请先在【基础特征工程】完成全量特征生成")
         else:
             st.subheader("当前模型权重配置（归一化后）")
             weight_col1, weight_col2, weight_col3, weight_col4 = st.columns(4)
             with weight_col1:
                 st.metric("逻辑回归", f"{round(model_weight_config['LogisticRegression']*100, 2)}%")
             with weight_col2:
                 st.metric("随机森林", f"{round(model_weight_config['RandomForest']*100, 2)}%")
             with weight_col3:
                 st.metric("XGBoost", f"{round(model_weight_config['XGBoost']*100, 2)}%")
             with weight_col4:
                 st.metric("LightGBM", f"{round(model_weight_config['LightGBM']*100, 2)}%")
             
             if st.button("✅ 一键训练多模型&融合预测", type="primary", use_container_width=True):
                 with st.spinner("多模型训练&融合预测中，防过拟合优化..."):
                     final_predict_df, model_metrics_df, hit_summary_df = train_multi_model(
                         feature_df, feature_columns, model_weight_config=model_weight_config
                     )
                 
                 if not final_predict_df.empty:
                     st.success("多模型融合预测完成！")
                     
                     st.markdown("---")
                     st.subheader("模型效果对比")
                     st.dataframe(model_metrics_df, use_container_width=True)
                     st.bar_chart(model_metrics_df, x="模型名称", y="平均AUC值", use_container_width=True)
                     
                     st.markdown("---")
                     st.subheader("下期号码融合预测排名（按最终融合概率降序）")
                     col1, col2 = st.columns(2)
                     with col1:
                         st.dataframe(final_predict_df, use_container_width=True)
                     with col2:
                         st.subheader("Top20高概率推荐号码")
                         top20_numbers = final_predict_df.head(20)["号码"].tolist()
                         st.markdown(f"### {sorted(top20_numbers)}")
                         st.subheader("Top5胆码推荐")
                         top5_dan = final_predict_df.head(5)["号码"].tolist()
                         st.markdown(f"### {sorted(top5_dan)}")
                     
                     # 融合预测结果下载
                     st.download_button(
                         label="📥 下载多模型融合预测结果CSV",
                         data=final_predict_df.to_csv(index=False, encoding="utf-8-sig"),
                         file_name=f"快乐8_{df['期号'].iloc[-1]}期_多模型融合预测号码池.csv",
                         mime="text/csv",
                         use_container_width=True
                     )
     # -------------------- Tab6：命中率复盘 --------------------
     with tab6:
         st.header("📈 命中率复盘中心（多维归因，定位准确率短板）")
         col1, col2 = st.columns(2)
         with col1:
             st.subheader("单模型命中率复盘")
             hit_summary_df = st.session_state["hit_summary"]
             if hit_summary_df.empty:
                 st.warning("请先在【单模型预测】完成模型训练")
             else:
                 metric_col1, metric_col2, metric_col3 = st.columns(3)
                 with metric_col1:
                     st.metric("测试集平均命中个数", round(hit_summary_df["Top20预测命中个数"].mean(), 2))
                 with metric_col2:
                     st.metric("测试集平均命中率", f"{round(hit_summary_df['命中率'].mean()*100, 2)}%")
                 with metric_col3:
                     st.metric("最高单期命中个数", hit_summary_df["Top20预测命中个数"].max())
                 
                 st.subheader("每期命中率走势")
                 st.line_chart(hit_summary_df, x="期号", y="Top20预测命中个数", use_container_width=True)
                 
                 # 复盘报告下载
                 st.download_button(
                     label="📥 下载单模型命中率复盘报告CSV",
                     data=hit_summary_df.to_csv(index=False, encoding="utf-8-sig"),
                     file_name="快乐8_单模型命中率复盘报告.csv",
                     mime="text/csv",
                     use_container_width=True
                 )
         
         with col2:
             st.subheader("多模型融合命中率复盘")
             multi_hit_summary_df = st.session_state["multi_model_hit_summary"]
             if multi_hit_summary_df.empty:
                 st.warning("请先在【多模型融合优化】完成模型训练")
             else:
                 metric_col1, metric_col2, metric_col3 = st.columns(3)
                 with metric_col1:
                     st.metric("测试集平均命中个数", round(multi_hit_summary_df["Top20预测命中个数"].mean(), 2))
                 with metric_col2:
                     st.metric("测试集平均命中率", f"{round(multi_hit_summary_df['命中率'].mean()*100, 2)}%")
                 with metric_col3:
                     st.metric("最高单期命中个数", multi_hit_summary_df["Top20预测命中个数"].max())
                 
                 st.subheader("每期命中率走势")
                 st.line_chart(multi_hit_summary_df, x="期号", y="Top20预测命中个数", use_container_width=True)
                 
                 # 复盘报告下载
                 st.download_button(
                     label="📥 下载多模型融合命中率复盘报告CSV",
                     data=multi_hit_summary_df.to_csv(index=False, encoding="utf-8-sig"),
                     file_name="快乐8_多模型融合命中率复盘报告.csv",
                     mime="text/csv",
                     use_container_width=True
                 )
     # -------------------- Tab7：普通打票组合 --------------------
     with tab7:
         st.header("🎫 低重复率普通打票组合生成中心")
         predict_source = st.radio("选择预测数据源", ["单模型预测结果", "多模型融合预测结果"], horizontal=True)
         
         # 数据源选择
         predict_df = None
         if predict_source == "单模型预测结果":
             predict_df = st.session_state["predict_result"]
             if predict_df.empty:
                 st.warning("请先在【单模型预测】完成预测结果生成")
         else:
             predict_df = st.session_state["final_predict_result"]
             if predict_df.empty:
                 st.warning("请先在【多模型融合优化】完成预测结果生成")
         
         if not predict_df.empty:
             if st.button("✅ 生成低重复率普通打票组合", type="primary", use_container_width=True):
                 combo_df = generate_low_repeat_combinations(
                     predict_df=predict_df,
                     select_count=select_number_count,
                     group_count=generate_group_count,
                     max_repeat_rate=max_repeat_rate
                 )
                 
                 if not combo_df.empty:
                     st.success(f"成功生成{generate_group_count}组{select_number_count}码低重复率组合")
                     st.dataframe(combo_df, use_container_width=True)
                     
                     # 组合下载
                     st.download_button(
                         label="📥 下载普通打票组合CSV",
                         data=combo_df.to_csv(index=True, encoding="utf-8-sig"),
                         file_name=f"快乐8_{df['期号'].iloc[-1]}期_普通打票组合.csv",
                         mime="text/csv",
                         use_container_width=True
                     )
     # -------------------- Tab8：胆拖组合生成 --------------------
     with tab8:
         st.header("💎 胆拖组合专属生成中心")
         predict_source = st.radio("选择预测数据源", ["多模型融合预测结果", "单模型预测结果"], horizontal=True)
         
         # 数据源选择
         predict_df = None
         if predict_source == "多模型融合预测结果":
             predict_df = st.session_state["final_predict_result"]
             if predict_df.empty:
                 st.warning("请先在【多模型融合优化】完成预测结果生成")
         else:
             predict_df = st.session_state["predict_result"]
             if predict_df.empty:
                 st.warning("请先在【单模型预测】完成预测结果生成")
         
         if not predict_df.empty:
             # 8+8专属模式一键切换
             if st.checkbox("启用8+8专属模式（8胆8拖）"):
                 dan_count = 8
                 tuo_count = 8
                 st.info("已启用8+8专属模式，胆码个数=8，拖码个数=8")
             
             if st.button("✅ 生成低重复率胆拖组合", type="primary", use_container_width=True):
                 dantuo_df, dan_pool, tuo_pool = generate_dantuo_combinations(
                     predict_df=predict_df,
                     dan_count=dan_count,
                     tuo_count=tuo_count,
                     group_count=dantuo_group_count,
                     max_dan_repeat_rate=max_dan_repeat,
                     max_tuo_repeat_rate=max_tuo_repeat
                 )
                 
                 if not dantuo_df.empty:
                     st.success(f"成功生成{dantuo_group_count}组{dan_count}胆{tuo_count}拖低重复率组合")
                     
                     col1, col2 = st.columns(2)
                     with col1:
                         st.subheader("胆码池（Top高概率）")
                         st.write(sorted(dan_pool))
                     with col2:
                         st.subheader("拖码池（次高概率）")
                         st.write(sorted(tuo_pool))
                     
                     st.markdown("---")
                     st.subheader("胆拖组合明细")
                     st.dataframe(dantuo_df, use_container_width=True)
                     
                     # 胆拖组合下载
                     st.download_button(
                         label="📥 下载胆拖组合CSV",
                         data=dantuo_df.to_csv(index=True, encoding="utf-8-sig"),
                         file_name=f"快乐8_{df['期号'].iloc[-1]}期_{dan_count}胆{tuo_count}拖组合.csv",
                         mime="text/csv",
                         use_container_width=True
                     )
 # ===================== 程序入口（直接运行即可启动） =====================
 if __name__ == "__main__":
     build_streamlit_page()
