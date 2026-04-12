# -------------------------- 1. 基础库导入 & 全局配置 --------------------------
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from collections import defaultdict
import os

# 全局警告过滤
warnings.filterwarnings('ignore')

# matplotlib中文显示全平台兼容修复（核心解决Linux无SimHei字体报错）
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Streamlit页面全局配置（必须放在所有页面元素之前）
st.set_page_config(
    page_title="快乐8数据分析预测系统V2.0",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Streamlit session_state全局初始化（解决key不存在报错）
def init_session_state():
    default_keys = {
        "raw_original_db": None,
        "feature_engineered_db": None,
        "feature_columns": [],
        "model_dict": {},
        "predict_result": None,
        "hit_summary": None,
        "accuracy_detail": None,
        "multi_model_dict": {},
        "model_metrics": None,
        "final_predict_result": None,
        "multi_model_hit_summary": None
    }
    for key, default_value in default_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# 执行初始化
init_session_state()

# 全局常量定义（严格匹配快乐8规则+自定义规则）
LOTTERY_RULE = {
    "total_numbers": 80,  # 快乐8总号码池1-80
    "draw_per_period": 20,  # 每期开奖20个号码
    "number_range": range(1, 81),
    "interval_count": 8,  # 8个区间，每个区间10个号码
    "tail_number_count": 10  # 0-9共10个尾号
}

# 自定义规则定义（完全匹配原有定义）
CUSTOM_RULE = {
    "follow_number": "同期一起出现的号码（跟随号）",
    "accompany_number": "本期开出N，下期开出的M/Q等后续号码（相随号）",
    "repeat_number": "本期开出A，下期继续开出A（重号）"
}

# -------------------------- 2. 工具函数模块（全错误修复+边界兜底） --------------------------
@st.cache_data(ttl=3600)  # 缓存提速，避免重复计算
def load_standard_data(uploaded_file=None):
    """
    数据加载&标准化：严格区分【只读原始底层库】和【衍生分析库】
    原始数据永久不修改，所有计算均在副本上执行
    """
    # 无上传文件时使用模拟测试数据
    if uploaded_file is None:
        st.warning("当前使用示例测试数据，请上传官方开奖CSV/Excel文件获取真实分析结果")
        # 生成模拟历史开奖数据（列：期号, 开奖号码1-20）
        mock_periods = 200
        mock_data = []
        for period in range(1, mock_periods+1):
            draw_numbers = sorted(np.random.choice(LOTTERY_RULE["number_range"], size=LOTTERY_RULE["draw_per_period"], replace=False))
            mock_data.append([f"2026{str(period).zfill(3)}"] + draw_numbers)
        columns = ["期号"] + [f"开奖号码{i}" for i in range(1, 21)]
        raw_df = pd.DataFrame(mock_data, columns=columns)
    else:
        # 读取上传文件，支持csv/excel
        try:
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file, dtype={"期号": str})
            else:
                raw_df = pd.read_excel(uploaded_file, dtype={"期号": str})
        except Exception as e:
            st.error(f"文件读取失败：{str(e)}，已自动切换为测试数据")
            mock_periods = 200
            mock_data = []
            for period in range(1, mock_periods+1):
                draw_numbers = sorted(np.random.choice(LOTTERY_RULE["number_range"], size=LOTTERY_RULE["draw_per_period"], replace=False))
                mock_data.append([f"2026{str(period).zfill(3)}"] + draw_numbers)
            columns = ["期号"] + [f"开奖号码{i}" for i in range(1, 21)]
            raw_df = pd.DataFrame(mock_data, columns=columns)
    
    # 数据列名校验&修复
    required_cols = ["期号"] + [f"开奖号码{i}" for i in range(1, 21)]
    missing_cols = [col for col in required_cols if col not in raw_df.columns]
    if missing_cols:
        st.error(f"数据格式错误，缺失必填列：{missing_cols}，已自动切换为测试数据")
        mock_periods = 200
        mock_data = []
        for period in range(1, mock_periods+1):
            draw_numbers = sorted(np.random.choice(LOTTERY_RULE["number_range"], size=LOTTERY_RULE["draw_per_period"], replace=False))
            mock_data.append([f"2026{str(period).zfill(3)}"] + draw_numbers)
        columns = ["期号"] + [f"开奖号码{i}" for i in range(1, 21)]
        raw_df = pd.DataFrame(mock_data, columns=columns)

    # 【只读原始底层库】永久存档，不做任何修改
    st.session_state["raw_original_db"] = raw_df.copy(deep=True)
    
    # 衍生分析库：数据清洗&标准化
    df = raw_df.copy(deep=True)
    # 期号按升序排列（旧期在前，新期在后，保证时序正确）
    df = df.sort_values("期号", ascending=True).reset_index(drop=True)
    # 提取每期开奖号码集合，方便后续计算
    df["开奖号码集合"] = df[[f"开奖号码{i}" for i in range(1, 21)]].values.tolist()
    df["开奖号码集合"] = df["开奖号码集合"].apply(lambda x: set([int(num) for num in x if pd.notna(num) and str(num).strip() != ""]))
    # 过滤无效行（开奖号码不足20个）
    df = df[df["开奖号码集合"].apply(len) == LOTTERY_RULE["draw_per_period"]].reset_index(drop=True)
    
    return df

# ================================== 特征工程核心函数 ==================================
def build_feature_engineer(df, rolling_window_list=[5, 10, 30]):
    """
    特征工程核心模块：基础特征+高级特征，严格滚动窗口计算（杜绝未来数据泄露）
    新增：区间分布、奇偶比、同尾号、连号、斜连号特征
    输出：带全量特征的数据集，用于模型训练&预测
    """
    feature_df = df.copy(deep=True)
    total_periods = len(feature_df)
    
    # -------------------------- 1. 基础统计特征：冷热号、遗漏值 --------------------------
    for window in rolling_window_list:
        # 滚动窗口内号码出现频次（热度）
        hot_count_matrix = np.zeros((total_periods, LOTTERY_RULE["total_numbers"]))
        # 滚动窗口内号码遗漏值（未出现的期数）
        miss_count_matrix = np.zeros((total_periods, LOTTERY_RULE["total_numbers"]))
        
        for period_idx in range(total_periods):
            # 【核心防泄露】仅用当期之前的历史数据计算特征，绝对不用未来数据
            start_idx = max(0, period_idx - window)
            history_numbers = feature_df.loc[start_idx:period_idx-1, "开奖号码集合"].tolist()
            
            if len(history_numbers) == 0:
                continue
            
            # 计算每个号码的出现频次（热度）
            for num in LOTTERY_RULE["number_range"]:
                appear_count = sum([1 for draw in history_numbers if num in draw])
                hot_count_matrix[period_idx, num-1] = appear_count
                # 计算遗漏值
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
    
    # -------------------------- 2. 自定义规则特征：重号、相随号、跟随号 --------------------------
    # 2.1 重号特征：上期号码在本期出现的概率
    repeat_prob_matrix = np.zeros((total_periods, LOTTERY_RULE["total_numbers"]))
    # 2.2 相随号特征：上期开出N，下期开出M的共现概率
    accompany_prob_matrix = np.zeros((total_periods, LOTTERY_RULE["total_numbers"]))
    # 2.3 跟随号特征：号码同期共现频次
    follow_cooccur_matrix = np.zeros((total_periods, LOTTERY_RULE["total_numbers"]))
    
    for period_idx in range(1, total_periods):
        history_df = feature_df.loc[0:period_idx-1, :]
        if len(history_df) < 2:
            continue
        
        # 重号概率统计
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
        
        # 相随号概率统计（核心匹配定义）
        accompany_count = defaultdict(lambda: defaultdict(int))
        total_accompany_samples = len(history_df) - 1
        for i in range(len(history_df)-1):
            last_draw = history_df.iloc[i]["开奖号码集合"]
            current_draw = history_df.iloc[i+1]["开奖号码集合"]
            for n in last_draw:
                for m in current_draw:
                    accompany_count[n][m] += 1
        # 用上期开奖号码，计算对应相随号概率
        last_period_draw = feature_df.iloc[period_idx-1]["开奖号码集合"]
        for m in LOTTERY_RULE["number_range"]:
            total_accompany = sum([accompany_count[n].get(m, 0) for n in last_period_draw])
            accompany_prob_matrix[period_idx, m-1] = total_accompany / (len(last_period_draw) * total_accompany_samples) if total_accompany_samples > 0 else 0
        
        # 跟随号共现统计（修复循环变量i重复覆盖的致命错误）
        follow_count = defaultdict(lambda: defaultdict(int))
        total_follow_samples = len(history_df)
        for draw_idx in range(len(history_df)):
            current_draw = history_df.iloc[draw_idx]["开奖号码集合"]
            draw_list = list(current_draw)
            # 内层循环变量改为j，避免覆盖外层draw_idx
            for i in range(len(draw_list)):
                for j in range(i+1, len(draw_list)):
                    n1, n2 = draw_list[i], draw_list[j]
                    follow_count[n1][n2] += 1
                    follow_count[n2][n1] += 1
        # 计算每个号码的平均跟随共现度
        for num in LOTTERY_RULE["number_range"]:
            total_follow = sum(follow_count[num].values())
            follow_cooccur_matrix[period_idx, num-1] = total_follow / (LOTTERY_RULE["draw_per_period"] * total_follow_samples) if total_follow_samples > 0 else 0
    
    # 自定义特征存入DataFrame
    for num in LOTTERY_RULE["number_range"]:
        feature_df[f"重号概率_{num}"] = repeat_prob_matrix[:, num-1]
        feature_df[f"相随号概率_{num}"] = accompany_prob_matrix[:, num-1]
        feature_df[f"跟随号共现度_{num}"] = follow_cooccur_matrix[:, num-1]

    # -------------------------- 3. 高级特征1：区间分布特征 --------------------------
    # 区间划分：1-10(区间1), 11-20(区间2)...71-80(区间8)
    def get_interval(num):
        return (num - 1) // 10 + 1
    
    for window in rolling_window_list:
        interval_hot_matrix = np.zeros((total_periods, LOTTERY_RULE["interval_count"]))
        for period_idx in range(total_periods):
            start_idx = max(0, period_idx - window)
            history_numbers = feature_df.loc[start_idx:period_idx-1, "开奖号码集合"].tolist()
            if len(history_numbers) == 0:
                continue
            # 统计每个区间的出号频次
            for interval in range(1, LOTTERY_RULE["interval_count"]+1):
                interval_count = 0
                for draw in history_numbers:
                    interval_count += sum([1 for num in draw if get_interval(num) == interval])
                interval_hot_matrix[period_idx, interval-1] = interval_count
        # 存入特征
        for interval in range(1, LOTTERY_RULE["interval_count"]+1):
            feature_df[f"近{window}期_区间{interval}_热度"] = interval_hot_matrix[:, interval-1]
    
    # 每期区间出号个数特征
    for period_idx in range(total_periods):
        current_draw = feature_df.iloc[period_idx]["开奖号码集合"]
        for interval in range(1, LOTTERY_RULE["interval_count"]+1):
            feature_df.loc[period_idx, f"本期_区间{interval}_出号个数"] = sum([1 for num in current_draw if get_interval(num) == interval])

    # -------------------------- 4. 高级特征2：奇偶比特征 --------------------------
    def is_odd(num):
        return num % 2 == 1
    
    for window in rolling_window_list:
        odd_hot_matrix = np.zeros((total_periods, 2))  # 0:偶数 1:奇数
        for period_idx in range(total_periods):
            start_idx = max(0, period_idx - window)
            history_numbers = feature_df.loc[start_idx:period_idx-1, "开奖号码集合"].tolist()
            if len(history_numbers) == 0:
                continue
            # 统计奇偶出号频次
            odd_count = sum([sum([1 for num in draw if is_odd(num)]) for draw in history_numbers])
            even_count = sum([sum([1 for num in draw if not is_odd(num)]) for draw in history_numbers])
            total_count = odd_count + even_count
            odd_hot_matrix[period_idx, 1] = odd_count / total_count if total_count > 0 else 0
            odd_hot_matrix[period_idx, 0] = even_count / total_count if total_count > 0 else 0
        # 存入特征
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
            # 统计每个尾号的出号频次
            for tail in range(0, LOTTERY_RULE["tail_number_count"]):
                tail_count = sum([sum([1 for num in draw if get_tail(num) == tail]) for draw in history_numbers])
                tail_hot_matrix[period_idx, tail] = tail_count
        # 存入特征
        for tail in range(0, LOTTERY_RULE["tail_number_count"]):
            feature_df[f"近{window}期_尾号{tail}_热度"] = tail_hot_matrix[:, tail]

    # -------------------------- 6. 高级特征4：连号&斜连号特征 --------------------------
    # 连号特征：每期连号组数
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
    
    # 斜连号特征：隔期+1/-1出号概率
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
        # 存入特征
        for num in LOTTERY_RULE["number_range"]:
            feature_df[f"近{window}期_斜连号概率_{num}"] = oblique_prob_matrix[:, num-1]

    # -------------------------- 7. 标签构建：下期号码是否开出（用于模型训练） --------------------------
    label_matrix = np.zeros((total_periods, LOTTERY_RULE["total_numbers"]))
    for period_idx in range(total_periods-1):
        next_draw = feature_df.iloc[period_idx+1]["开奖号码集合"]
        for num in next_draw:
            label_matrix[period_idx, num-1] = 1
    for num in LOTTERY_RULE["number_range"]:
        feature_df[f"下期是否开出_{num}"] = label_matrix[:, num-1]
    
    # 剔除第一期无历史数据的空行，剔除最后一期无标签的行
    feature_df = feature_df.iloc[1:-1, :].reset_index(drop=True)
    
    # 特征筛选：剔除无方差的无效特征，降噪提效
    feature_columns = [col for col in feature_df.columns if any(keyword in col for keyword in ["热度", "遗漏值", "概率", "共现度", "区间", "奇偶", "尾号", "斜连"])]
    # 空特征兜底
    if len(feature_columns) == 0:
        feature_columns = [col for col in feature_df.columns if col not in ["期号", "开奖号码集合"] + [f"开奖号码{i}" for i in range(1,21)] + [f"下期是否开出_{num}" for num in LOTTERY_RULE["number_range"]]]
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(feature_df[feature_columns])
    valid_feature_columns = feature_df[feature_columns].columns[selector.get_support()].tolist()
    # 有效特征兜底
    if len(valid_feature_columns) == 0:
        valid_feature_columns = feature_columns[:20]
    
    # 存入session_state
    st.session_state["feature_columns"] = valid_feature_columns
    st.session_state["feature_engineered_db"] = feature_df.copy(deep=True)
    
    return feature_df, valid_feature_columns

# 特征分析配套函数
def feature_analysis(feature_df, feature_columns):
    """特征贡献度、相关性分析"""
    # 特征相关性计算
    corr_df = feature_df[feature_columns].corr()
    # 特征方差分析
    feature_var = feature_df[feature_columns].var().sort_values(ascending=False)
    return corr_df, feature_var

# ================================== 多模型融合训练核心函数 ==================================
def train_multi_model(feature_df, feature_columns, model_weight_config=None):
    """
    多模型融合训练&预测：LR、RF、XGBoost、LightGBM四大模型
    支持自定义权重融合，严格时序拆分防过拟合
    输出：单模型结果、融合预测结果、模型对比指标
    """
    # 默认权重配置（可自定义）
    if model_weight_config is None:
        model_weight_config = {
            "LogisticRegression": 0.15,
            "RandomForest": 0.2,
            "XGBoost": 0.35,
            "LightGBM": 0.3
        }
    
    # 数据集严格时序拆分（杜绝随机拆分导致的过拟合）
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
    
    # 每个号码的模型训练结果存储
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
        
        # 数据集拆分
        X_train = train_df[feature_columns]
        y_train = train_df[label_col]
        X_val = val_df[feature_columns]
        y_val = val_df[label_col]
        X_test = test_df[feature_columns]
        y_test = test_df[label_col]
        X_latest = feature_df.iloc[-1:][feature_columns]  # 最新一期特征，用于下期预测
        
        # 逐个模型训练
        for model_name, model in model_dict.items():
            # 训练模型
            try:
                if model_name in ["XGBoost", "LightGBM"]:
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    model.fit(X_train, y_train)
                
                # 预测
                test_pred_prob = model.predict_proba(X_test)[:, 1]
                latest_pred_prob = model.predict_proba(X_latest)[:, 1][0]
                
                # 存储结果
                number_model_results[num][model_name] = model
                test_predict_results[num][model_name] = test_pred_prob
                latest_predict_results[num][model_name] = latest_pred_prob
                
                # 计算指标（AUC），修复全0/全1标签报错
                if len(np.unique(y_test)) >= 2:
                    auc_score = roc_auc_score(y_test, test_pred_prob)
                else:
                    auc_score = 0.5  # 极端情况默认0.5
                model_metrics[model_name].append(auc_score)
            except Exception as e:
                # 异常兜底
                number_model_results[num][model_name] = None
                test_predict_results[num][model_name] = np.zeros(len(X_test))
                latest_predict_results[num][model_name] = 0.5
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
        
        # 存入结果
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
        period_data = test_df[test_df["期号"] == period]
        period_number_prob = []
        for num in LOTTERY_RULE["number_range"]:
            real_label = period_data[f"下期是否开出_{num}"].values[0]
            # 测试集多模型融合概率
            test_weighted_prob = 0
            for model_name in model_dict.keys():
                test_weighted_prob += test_predict_results[num][model_name][period_idx] * model_weight_config[model_name]
            period_number_prob.append({
                "号码": num,
                "真实是否开出": real_label,
                "融合预测概率": test_weighted_prob
            })
        # 统计Top20命中个数
        period_number_df = pd.DataFrame(period_number_prob).sort_values("融合预测概率", ascending=False)
        top20_hit = period_number_df.head(20)["真实是否开出"].sum()
        hit_summary_list.append({
            "期号": period,
            "Top20预测命中个数": top20_hit,
            "命中率": top20_hit / LOTTERY_RULE["draw_per_period"]
        })
    hit_summary_df = pd.DataFrame(hit_summary_list)
    
    # 存入session_state
    st.session_state["multi_model_dict"] = number_model_results
    st.session_state["model_metrics"] = model_metrics_df
    st.session_state["final_predict_result"] = final_predict_df
    st.session_state["multi_model_hit_summary"] = hit_summary_df
    
    return final_predict_df, model_metrics_df, hit_summary_df

# 单模型训练函数（保留兼容）
def train_predict_model(feature_df, feature_columns):
    """原有单模型训练函数，保留兼容原有功能"""
    total_samples = len(feature_df)
    train_size = int(total_samples * 0.7)
    val_size = int(total_samples * 0.15)
    
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
        
        X_train = train_df[feature_columns]
        y_train = train_df[label_col]
        X_val = val_df[feature_columns]
        y_val = val_df[label_col]
        X_test = test_df[feature_columns]
        y_test = test_df[label_col]
        
        model = XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            reg_alpha=1,
            reg_lambda=1,
            random_state=42,
            eval_metric="logloss"
        )
        
        try:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            model_dict[num] = model
            
            test_pred_prob = model.predict_proba(X_test)[:, 1]
            test_df[f"预测概率_{num}"] = test_pred_prob
            
            for i in range(len(test_df)):
                period = test_df.iloc[i]["期号"]
                real_label = test_df.iloc[i][label_col]
                pred_prob = test_pred_prob[i]
                test_accuracy_detail.append({
                    "期号": period,
                    "号码": num,
                    "真实是否开出": real_label,
                    "模型预测概率": pred_prob
                })
            
            latest_feature = feature_df.iloc[-1:][feature_columns]
            latest_prob = model.predict_proba(latest_feature)[:, 1][0]
            predict_result.append({
                "号码": num,
                "模型预测概率": latest_prob,
                "规则加权概率": 0
            })
        except Exception as e:
            model_dict[num] = None
            predict_result.append({
                "号码": num,
                "模型预测概率": 0.5,
                "规则加权概率": 0
            })
    
    progress_bar.empty()
    
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
    
    predict_df = predict_df.sort_values("最终融合概率", ascending=False).reset_index(drop=True)
    
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
    
    st.session_state["model_dict"] = model_dict
    st.session_state["predict_result"] = predict_df
    st.session_state["hit_summary"] = hit_summary_df
    st.session_state["accuracy_detail"] = accuracy_df
    
    return predict_df, hit_summary_df, accuracy_df

# 普通组合生成函数（保留兼容）
def generate_low_repeat_combinations(predict_df, select_count=8, group_count=5, max_repeat_rate=0.3):
    """低重复率普通组合生成"""
    # 号码池长度兜底
    top_pool_size = max(select_count * 3, 20)
    top_pool = predict_df.head(top_pool_size)["号码"].tolist()
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
                if avg_repeat <= max_repeat_rate and num not in candidate_combo:
                    candidate_combo.append(num)
                if len(candidate_combo) == select_count:
                    break
            # 长度不足兜底
            while len(candidate_combo) < select_count:
                backup_num = np.random.choice([n for n in top_pool if n not in candidate_combo])
                candidate_combo.append(backup_num)
            combo = sorted(candidate_combo)
        
        if combo not in combinations:
            combinations.append(combo)
    
    # 补全不足的组数
    while len(combinations) < group_count:
        backup_combo = sorted(np.random.choice(top_pool, size=select_count, replace=False))
        if backup_combo not in combinations:
            combinations.append(backup_combo)
    
    combo_df = pd.DataFrame(combinations, columns=[f"号码{i+1}" for i in range(select_count)])
    combo_df.index = [f"第{i+1}组" for i in range(len(combo_df))]
    
    return combo_df

# ================================== 胆拖组合生成核心函数 ==================================
 def generate_dantuo_combinations(predict_df, dan_count=5, tuo_count=8, group_count=5, max_dan_repeat_rate=0.2, max_tuo_repeat_rate=0.4):
     """
     胆拖组合生成函数，适配5胆N拖、8+8等核心场景
     dan_count: 胆码个数
     tuo_count: 拖码个数
     group_count: 生成组数
     max_dan_repeat_rate: 胆码最大重复率
     max_tuo_repeat_rate: 拖码最大重复率
     """
     # 号码池长度兜底
     dan_pool_size = max(dan_count * 3, 15)
     tuo_pool_size = max(tuo_count * 3, 20)
     
     # 胆码池：Top高概率号码
     dan_pool = predict_df.head(dan_pool_size)["号码"].tolist()
     # 拖码池：次高概率号码（排除胆码池）
     tuo_pool = predict_df[~predict_df["号码"].isin(dan_pool)].head(tuo_pool_size)["号码"].tolist()
     
     # 拖码池长度兜底
     if len(tuo_pool) < tuo_count:
         supplement_nums = [n for n in LOTTERY_RULE["number_range"] if n not in dan_pool and n not in tuo_pool]
         tuo_pool += supplement_nums[:tuo_count - len(tuo_pool)]
     
     combinations = []
     
     for i in range(group_count):
         # 生成胆码
         if i == 0:
             # 第一组取概率最高的胆码
             current_dan = sorted(dan_pool[:dan_count])
         else:
             # 后续组控制胆码重复率
             current_dan = []
             for num in dan_pool:
                 repeat_count = 0
                 for exist_dan, _ in combinations:
                     if num in exist_dan:
                         repeat_count += 1
                 avg_repeat = repeat_count / len(combinations) if len(combinations) > 0 else 0
                 if avg_repeat <= max_dan_repeat_rate and num not in current_dan:
                     current_dan.append(num)
                 if len(current_dan) == dan_count:
                     break
             # 长度不足兜底
             while len(current_dan) < dan_count:
                 backup_num = np.random.choice([n for n in dan_pool if n not in current_dan])
                 current_dan.append(backup_num)
             current_dan = sorted(current_dan)
         
         # 生成拖码
         current_tuo = []
         for num in tuo_pool:
             # 拖码不能和胆码重复
             if num in current_dan:
                 continue
             # 控制拖码重复率
             repeat_count = 0
             for _, exist_tuo in combinations:
                 if num in exist_tuo:
                     repeat_count += 1
             avg_repeat = repeat_count / len(combinations) if len(combinations) > 0 else 0
             if avg_repeat <= max_tuo_repeat_rate and num not in current_tuo:
                 current_tuo.append(num)
             if len(current_tuo) == tuo_count:
                 break
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
# -------------------------- 3. Streamlit页面布局（全Tab修复+边界兜底） --------------------------
st.title("快乐8数据分析&预测系统V2.0 | 4大新增核心模块")
st.markdown("---")

# 侧边栏：参数配置
with st.sidebar:
    st.header("参数配置中心")
    st.markdown("### 1. 数据上传")
    uploaded_file = st.file_uploader("上传官方开奖数据（CSV/Excel）", type=["csv", "xlsx"])
    st.caption("*数据格式要求：第一列为【期号】，后续20列为【开奖号码1-20】*")
    
    st.markdown("---")
    st.markdown("### 2. 特征工程参数")
    rolling_windows = st.multiselect(
        "滚动统计周期（期）",
        options=[1, 5, 10, 20, 30, 50, 100],
        default=[5, 10, 30]
    )
    
    st.markdown("---")
    st.markdown("### 3. 多模型权重配置")
    st.caption("权重总和自动归一化为1")
    lr_weight = st.slider("逻辑回归权重", 0.0, 1.0, 0.15, 0.05)
    rf_weight = st.slider("随机森林权重", 0.0, 1.0, 0.2, 0.05)
    xgb_weight = st.slider("XGBoost权重", 0.0, 1.0, 0.35, 0.05)
    lgb_weight = st.slider("LightGBM权重", 0.0, 1.0, 0.3, 0.05)
    # 权重归一化
    total_weight = lr_weight + rf_weight + xgb_weight + lgb_weight
    model_weight_config = {
        "LogisticRegression": lr_weight / total_weight,
        "RandomForest": rf_weight / total_weight,
        "XGBoost": xgb_weight / total_weight,
        "LightGBM": lgb_weight / total_weight
    }
    
    st.markdown("---")
    st.markdown("### 4. 普通组合生成参数")
    select_number_count = st.slider("单组选号个数", min_value=5, max_value=20, value=8)
    generate_group_count = st.slider("生成组合组数", min_value=1, max_value=20, value=5)
    max_repeat_rate = st.slider("组间最大重复率", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
    
    st.markdown("---")
    st.markdown("### 5. 胆拖组合生成参数")
    dan_count = st.slider("胆码个数", min_value=1, max_value=10, value=5)
    tuo_count = st.slider("拖码个数", min_value=5, max_value=20, value=8)
    dantuo_group_count = st.slider("胆拖组合组数", min_value=1, max_value=20, value=5)
    max_dan_repeat = st.slider("胆码最大重复率", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    max_tuo_repeat = st.slider("拖码最大重复率", min_value=0.2, max_value=0.7, value=0.4, step=0.05)
    
    st.markdown("---")
    st.markdown("### 6. 免责声明")
    st.caption("本系统仅为数据分析工具，彩票开奖为独立随机事件，不构成任何购彩建议，理性购彩，量力而行")

# 主页面：Tab布局
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 数据底层库", 
    "🔍 基础特征工程", 
    "✨ 高级特征分析", 
    "🤖 单模型预测", 
    "🚀 多模型融合优化", 
    "📈 命中率复盘", 
    "🎫 普通打票组合", 
    "💎 胆拖组合生成"
])

# Tab1：数据底层库管理
with tab1:
    st.header("底层数据库管理（原始库只读不可修改）")
    # 加载数据
    df = load_standard_data(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("只读原始底层库")
        st.dataframe(st.session_state["raw_original_db"], use_container_width=True)
        # 原始库下载
        st.download_button(
            label="下载原始底层库CSV",
            data=st.session_state["raw_original_db"].to_csv(index=False).encode("utf-8"),
            file_name="快乐8_原始开奖底层库.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        st.subheader("标准化分析库")
        st.dataframe(df[["期号", "开奖号码集合"]], use_container_width=True)
        st.metric("有效历史期数", len(df))

# Tab2：基础特征工程
with tab2:
    st.header("基础特征工程中心（自定义规则+防数据泄露）")
    if "raw_original_db" not in st.session_state or st.session_state["raw_original_db"] is None:
        st.warning("请先在【数据底层库】上传或加载数据")
    else:
        # 滚动窗口非空校验
        if len(rolling_windows) == 0:
            st.error("请至少选择一个滚动统计周期")
        else:
            if st.button("一键生成全量特征（含高级特征）", type="primary", use_container_width=True):
                with st.spinner("特征工程计算中，严格滚动窗口防数据泄露..."):
                    feature_df, valid_feature_columns = build_feature_engineer(df, rolling_window_list=rolling_windows)
                
                st.success("全量特征生成完成！含基础特征+高级特征")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("特征数据集总览")
                    st.dataframe(feature_df, use_container_width=True)
                    st.metric("有效特征数量", len(valid_feature_columns))
                    st.metric("有效样本量", len(feature_df))
                
                with col2:
                    st.subheader("核心有效特征列表")
                    st.write(valid_feature_columns)
                
                # 特征库下载
                st.download_button(
                    label="下载全量特征工程底层库CSV",
                    data=feature_df.to_csv(index=False).encode("utf-8"),
                    file_name="快乐8_全量特征工程底层库.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# Tab3：高级特征分析
with tab3:
    st.header("高级特征分析中心")
    if "feature_engineered_db" not in st.session_state or st.session_state["feature_engineered_db"] is None:
        st.warning("请先在【基础特征工程】完成全量特征生成")
    else:
        feature_df = st.session_state["feature_engineered_db"]
        feature_columns = st.session_state["feature_columns"]
        
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
            else:
                st.info("无区间热度特征，请重新生成特征")
        
        st.markdown("---")
        st.subheader("特征相关性热力图（Top20高区分度特征）")
        top20_features = feature_var.head(20).index.tolist()
        if len(top20_features) >= 2:
            top20_corr = feature_df[top20_features].corr()
            # 绘制热力图
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(top20_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax, fmt=".2f")
            st.pyplot(fig)
        else:
            st.warning("有效特征不足，无法绘制热力图")
        
        # 高级特征下载
        st.download_button(
            label="下载高级特征明细CSV",
            data=feature_df[[col for col in feature_df.columns if any(keyword in col for keyword in ["区间", "奇偶", "尾号", "连号", "斜连"])]].to_csv(index=False).encode("utf-8"),
            file_name="快乐8_高级特征明细.csv",
            mime="text/csv",
            use_container_width=True
        )

# Tab4：单模型预测
with tab4:
    st.header("单模型预测中心（XGBoost基准模型）")
    if "feature_engineered_db" not in st.session_state or st.session_state["feature_engineered_db"] is None:
        st.warning("请先在【基础特征工程】完成特征生成")
    else:
        feature_df = st.session_state["feature_engineered_db"]
        feature_columns = st.session_state["feature_columns"]
        
        if st.button("一键训练单模型&生成预测", type="primary", use_container_width=True):
            with st.spinner("模型训练&预测中，防过拟合优化..."):
                predict_df, hit_summary_df, accuracy_df = train_predict_model(feature_df, feature_columns)
            
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
                label="下载单模型预测结果CSV",
                data=predict_df.to_csv(index=False).encode("utf-8"),
                file_name=f"快乐8_{df.iloc[-1]['期号']}期_单模型预测号码池.csv",
                mime="text/csv",
                use_container_width=True
            )

# Tab5：多模型融合优化
with tab5:
    st.header("多模型融合优化中心")
    if "feature_engineered_db" not in st.session_state or st.session_state["feature_engineered_db"] is None:
        st.warning("请先在【基础特征工程】完成全量特征生成")
    else:
        feature_df = st.session_state["feature_engineered_db"]
        feature_columns = st.session_state["feature_columns"]
        
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
        
        if st.button("一键训练多模型&融合预测", type="primary", use_container_width=True):
            with st.spinner("多模型训练&融合预测中，防过拟合优化..."):
                final_predict_df, model_metrics_df, hit_summary_df = train_multi_model(
                    feature_df, feature_columns, model_weight_config=model_weight_config
                )
            
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
                label="下载多模型融合预测结果CSV",
                data=final_predict_df.to_csv(index=False).encode("utf-8"),
                file_name=f"快乐8_{df.iloc[-1]['期号']}期_多模型融合预测号码池.csv",
                mime="text/csv",
                use_container_width=True
            )

# Tab6：命中率复盘
with tab6:
    st.header("命中率复盘中心（多维归因，定位准确率短板）")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("单模型命中率复盘")
        if "hit_summary" not in st.session_state or st.session_state["hit_summary"] is None:
            st.warning("请先在【单模型预测】完成模型训练")
        else:
            hit_summary_df = st.session_state["hit_summary"]
            accuracy_df = st.session_state["accuracy_detail"]
            
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
                label="下载单模型命中率复盘报告CSV",
                data=hit_summary_df.to_csv(index=False).encode("utf-8"),
                file_name="快乐8_单模型命中率复盘报告.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        st.subheader("多模型融合命中率复盘")
        if "multi_model_hit_summary" not in st.session_state or st.session_state["multi_model_hit_summary"] is None:
            st.warning("请先在【多模型融合优化】完成模型训练")
        else:
            multi_hit_summary_df = st.session_state["multi_model_hit_summary"]
            
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
                label="下载多模型融合命中率复盘报告CSV",
                data=multi_hit_summary_df.to_csv(index=False).encode("utf-8"),
                file_name="快乐8_多模型融合命中率复盘报告.csv",
                mime="text/csv",
                use_container_width=True
            )

# Tab7：普通打票组合
with tab7:
    st.header("低重复率普通打票组合生成中心")
    predict_source = st.radio("选择预测数据源", ["单模型预测结果", "多模型融合预测结果"], horizontal=True)
    
    predict_df = None
    if predict_source == "单模型预测结果":
        if "predict_result" not in st.session_state or st.session_state["predict_result"] is None:
            st.warning("请先在【单模型预测】完成预测结果生成")
        else:
            predict_df = st.session_state["predict_result"]
    else:
        if "final_predict_result" not in st.session_state or st.session_state["final_predict_result"] is None:
            st.warning("请先在【多模型融合优化】完成预测结果生成")
        else:
            predict_df = st.session_state["final_predict_result"]
    
    if predict_df is not None:
        if st.button("生成低重复率普通打票组合", type="primary", use_container_width=True):
            combo_df = generate_low_repeat_combinations(
                predict_df=predict_df,
                select_count=select_number_count,
                group_count=generate_group_count,
                max_repeat_rate=max_repeat_rate
            )
            
            st.success(f"成功生成{generate_group_count}组{select_number_count}码低重复率组合")
            st.dataframe(combo_df, use_container_width=True)
            
            # 组合下载
            st.download_button(
                label="下载普通打票组合CSV",
                data=combo_df.to_csv(index=True).encode("utf-8"),
                file_name=f"快乐8_{df.iloc[-1]['期号']}期_普通打票组合.csv",
                mime="text/csv",
                use_container_width=True
            )

# Tab8：胆拖组合生成
with tab8:
    st.header("胆拖组合专属生成中心")
    predict_source = st.radio("选择预测数据源", ["多模型融合预测结果", "单模型预测结果"], horizontal=True)
    
    predict_df = None
    if predict_source == "多模型融合预测结果":
        if "final_predict_result" not in st.session_state or st.session_state["final_predict_result"] is None:
            st.warning("请先在【多模型融合优化】完成预测结果生成")
        else:
            predict_df = st.session_state["final_predict_result"]
    else:
        if "predict_result" not in st.session_state or st.session_state["predict_result"] is None:
            st.warning("请先在【单模型预测】完成预测结果生成")
        else:
            predict_df = st.session_state["predict_result"]
    
    if predict_df is not None:
        # 8+8专属模式一键切换
        if st.checkbox("启用8+8专属模式（8胆8拖）"):
            dan_count = 8
            tuo_count = 8
            st.info("已启用8+8专属模式，胆码个数=8，拖码个数=8")
        
        if st.button("生成低重复率胆拖组合", type="primary", use_container_width=True):
            dantuo_df, dan_pool, tuo_pool = generate_dantuo_combinations(
                predict_df=predict_df,
                dan_count=dan_count,
                tuo_count=tuo_count,
                group_count=dantuo_group_count,
                max_dan_repeat_rate=max_dan_repeat,
                max_tuo_repeat_rate=max_tuo_repeat
            )
            
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
                label="下载胆拖组合CSV",
                data=dantuo_df.to_csv(index=True).encode("utf-8"),
                file_name=f"快乐8_{df.iloc[-1]['期号']}期_{dan_count}胆{tuo_count}拖组合.csv",
                mime="text/csv",
                use_container_width=True
            )

# 全局底部免责声明
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 14px; line-height: 1.8; padding: 10px 0;">
⚠️ 本系统仅用于福彩快乐8历史开奖数据的统计与娱乐性分析，彩票开奖为完全随机独立事件<br>
所有分析结果、选号参考、预测内容均不构成任何购彩建议，请理性购彩，量力而行，遵守国家相关法律法规
</div>
""", unsafe_allow_html=True) 
