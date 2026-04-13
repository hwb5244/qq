import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import os
import csv
import datetime
import shutil
import zipfile
import warnings

# 屏蔽无关警告，优化运行体验
warnings.filterwarnings("ignore")

# 页面配置：必须是第一个Streamlit命令，绝对禁止任何Streamlit代码放在此之前
st.set_page_config(
    page_title="快乐8专业数据分析系统",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== 全局常量定义（执行顺序100%合规，先路径后业务规则） ======================
# 核心数据路径（先定义基础路径，再定义衍生路径，彻底解决路径未定义错误）
BASE_DIR = os.getcwd()
DATA_FILE = os.path.join(BASE_DIR, "kl8_history_data.csv")
SAVE_DIR = os.path.join(BASE_DIR, "lottery_save")
ARCHIVE_ROOT = os.path.join(BASE_DIR, "KL8_Lottery_Data_Archive")
INDEX_FILE = os.path.join(ARCHIVE_ROOT, "05_存档总索引表", "index.csv")

# 批量复盘全局存档配置（仅定义1次，放在ARCHIVE_ROOT之后，杜绝变量未定义）
BATCH_REVIEW_DIR = os.path.join(ARCHIVE_ROOT, "06_全量批量复盘存档")
BATCH_REVIEW_SUMMARY = os.path.join(BATCH_REVIEW_DIR, "全量期数复盘总表.csv")
BATCH_REVIEW_DETAIL_DIR = os.path.join(BATCH_REVIEW_DIR, "单期复盘明细")

# 业务规则常量（仅定义1次，无重复，可按需修改）
PRIME_NUMBERS = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79]
ZONE_RULE = {"zone1":[1,20],"zone2":[21,40],"zone3":[41,60],"zone4":[61,80]}
HOT_COLD_FACTOR = 2
PLAY_RULE = {"选10":10,"选8":8,"选7":7,"选5":5,"选20":20}
# 分析周期配置，可按需修改
PERIOD_WINDOW_OPTIONS = {
    "近10期": 10,
    "近20期": 20,
    "近50期": 50,
    "近100期": 100,
    "150期全量": None
}
# 固定玩法配置（全系统统一，杜绝重复定义）
FIX_PLAY_CONFIG = [
    {"玩法名称":"11码", "选号个数":11, "固定生成组数":3},
    {"玩法名称":"8码", "选号个数":8,  "固定生成组数":5},
    {"玩法名称":"6码", "选号个数":6,  "固定生成组数":10},
    {"玩法名称":"3码", "选号个数":3,  "固定生成组数":10}
]
MAX_OVERLAP_BETWEEN_TREND = 1  # 双流派核心池最大重叠数

# 统一初始化所有文件夹（所有路径定义完成后再创建，加异常捕获，杜绝路径创建失败崩溃）
REQUIRED_DIRS = [
    SAVE_DIR, 
    ARCHIVE_ROOT, 
    os.path.join(ARCHIVE_ROOT, "01_基准原始库"),
    os.path.join(ARCHIVE_ROOT, "02_增量开奖数据库"),
    os.path.join(ARCHIVE_ROOT, "03_每期预测号存档库"),
    os.path.join(ARCHIVE_ROOT, "04_每期选号组合存档库"),
    os.path.join(ARCHIVE_ROOT, "05_存档总索引表"),
    BATCH_REVIEW_DIR, 
    BATCH_REVIEW_DETAIL_DIR
]

for dir_path in REQUIRED_DIRS:
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
    except Exception as e:
        st.error(f"文件夹初始化失败：{str(e)}，请检查磁盘权限")  
        # ====================== 75期原始开奖基准数据 - 核心禁止删除 ======================
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

# ====================== 核心数据IO工具函数（全量校验+异常捕获，杜绝崩溃） ======================
def load_data():
    """加载全量开奖数据，自动校验完整性，损坏则自动恢复基准数据"""
    try:
        base_period_set = set(INIT_PERIOD_LIST)
        # 校验数据文件是否存在且完整
        if os.path.exists(DATA_FILE):
            temp_df = pd.read_csv(DATA_FILE, dtype={'period': str})
            file_period_set = set(temp_df['period'].tolist())
            # 基准期号缺失则重置
            if not base_period_set.issubset(file_period_set):
                raise ValueError("基准期号缺失，自动重置数据")
        else:
            raise ValueError("数据文件不存在，初始化基准数据")

        # 加载并校验数据合法性
        df = pd.read_csv(DATA_FILE, dtype={'period': str})
        required_cols = ['period'] + [f'n{i}' for i in range(1, 21)]
        if not all(col in df.columns for col in required_cols):
            raise ValueError("表头损坏，自动重置数据")
            
        # 过滤无效行
        valid_rows = []
        for _, row in df.iterrows():
            period = str(row['period']).strip()
            try:
                nums = [int(row[f'n{i}']) for i in range(1,21)]
                # 校验号码合法性：20个不重复、1-80范围
                if len(nums)!=20 or len(set(nums))!=20 or min(nums)<1 or max(nums)>80:
                    continue
                valid_rows.append(row)
            except Exception:
                continue
                
        # 去重+排序
        df = pd.DataFrame(valid_rows).reset_index(drop=True)
        df = df.drop_duplicates(subset=['period'], keep='last')
        df['period_num'] = df['period'].astype(int)
        df = df.sort_values('period_num', ascending=False).reset_index(drop=True)
        df = df.drop(columns=['period_num'])
        return df
    except Exception as e:
        # 异常时自动恢复基准数据
        with open(DATA_FILE, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['period'] + [f'n{i}' for i in range(1, 21)])
            w.writerows(INIT_DATA)
        df = pd.read_csv(DATA_FILE, dtype={'period': str})
        st.warning(f"数据已自动修复为基准原始数据：{str(e)}")
        return df

def save_new_data(period, numbers):
    """新增开奖数据，返回是否成功"""
    try:
        period = str(period).strip().zfill(7)
        with open(DATA_FILE, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([period] + sorted(numbers))
        return True
    except Exception as e:
        st.error(f"数据保存失败：{str(e)}")
        return False

def delete_period_data(period, df):
    """删除期号数据，禁止删除基准期号"""
    period = str(period).strip()
    if period in INIT_PERIOD_LIST:
        return df, False, "禁止删除：该期号为系统基准原始数据，无法删除"
    new_df = df[df['period'] != period].reset_index(drop=True)
    new_df.to_csv(DATA_FILE, index=False, encoding='utf-8')
    return new_df, True, "删除成功"

def validate_period_unique(period, df):
    """校验期号唯一性"""
    period = str(period).strip()
    if not period or not period.isdigit():
        return False, "期号必须为非空纯数字"
    if period in df['period'].values:
        return False, "期号已存在，禁止重复录入"
    return True, "校验通过"

def validate_numbers(nums):
    """校验开奖号码合法性"""
    try:
        ns = [int(x.strip()) for x in nums if x.strip()]
        if len(ns) != 20:
            return False, f"需输入20个号码，当前仅{len(ns)}个"
        if len(set(ns)) != 20:
            return False, "号码存在重复，请检查"
        if min(ns) < 1 or max(ns) > 80:
            return False, "号码必须在1-80范围内"
        return True, sorted(ns)
    except ValueError:
        return False, "号码格式错误，仅支持数字"

# ====================== 存档文件管理工具函数 ======================
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
    # ====================== 缓存装饰器（性能优化，精准控制缓存时效） ======================
@st.cache_data(ttl=3600)  # 静态数据1小时缓存
def load_data_cached():
    return load_data()

@st.cache_data(ttl=3600)  # 全量分析数据1小时缓存
def get_full_analysis_cached(df, window=None):
    return analyze_full_data(df, window)

# ====================== 存档管理核心工具函数 ======================
def save_predict_num(target_period, data_end_period, level2_list, level3_list):
    """保存预测号池，自动层间隔离"""
    # 清洗数据
    level2_clean = sorted(list(set(
        int(n) for n in level2_list
        if str(n).strip().isdigit() and 1 <= int(n) <= 80
    )))
    level3_raw = sorted(list(set(
        int(n) for n in level3_list
        if str(n).strip().isdigit() and 1 <= int(n) <= 80
    )))
    # 层间隔离：三级不能包含二级号码
    level3_clean = [num for num in level3_raw if num not in level2_clean]
    
    # 构造存档数据
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
    """保存选号组合"""
    filename = os.path.join(SAVE_DIR, f"{period}期选号组合.csv")
    rows = []
    for idx, nums in enumerate(comb_list):
        rows.append([period, play_type, f"方案{idx+1}", " ".join([f"{n:02d}" for n in nums])])
    df_save = pd.DataFrame(rows, columns=["期号", "玩法类型", "方案编号", "选号号码"])
    # 增量写入，不覆盖历史
    if os.path.exists(filename):
        df_old = pd.read_csv(filename, encoding="utf-8-sig")
        df_save = pd.concat([df_old, df_save], ignore_index=True).drop_duplicates(subset=["期号", "玩法类型", "方案编号"], keep="last")
    df_save.to_csv(filename, index=False, encoding="utf-8-sig")
    return filename

def load_predict_num(period):
    """加载指定期号的预测池数据"""
    filename = os.path.join(SAVE_DIR, f"{period}期预测号.csv")
    if os.path.exists(filename):
        return pd.read_csv(filename, encoding="utf-8-sig")
    return None

def load_all_select_comb():
    """加载所有选号组合数据"""
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

# ====================== 通用计算工具函数 ======================
def calc_match_rate(predict_nums, real_nums):
    """计算预测号码匹配率"""
    predict_set = set([int(x) for x in predict_nums])
    real_set = set([int(x) for x in real_nums])
    match = predict_set & real_set
    match_cnt = len(match)
    rate = round(match_cnt / len(predict_set) * 100, 2) if predict_set else 0
    return {"匹配号码": sorted(list(match)), "匹配个数": match_cnt, "正确率%": rate}

def fmt_num(n, num_status_dict):
    """号码格式化，带冷热色标"""
    s = num_status_dict.get(n, {"st": "warm", "road": "0路", "cnt": 0})
    if s['st'] == "hot":
        return f'<span style="color:red;font-weight:bold;margin:0 2px">{n:02d}</span><small style="color:#666">({s["road"]},{s["cnt"]}次)</small>'
    elif s['st'] == "cold":
        return f'<span style="color:blue;font-weight:bold;margin:0 2px">{n:02d}</span><small style="color:#666">({s["road"]},{s["cnt"]}次)</small>'
    else:
        return f'<span style="color:black;margin:0 2px">{n:02d}</span><small style="color:#666">({s["road"]},{s["cnt"]}次)</small>'

def get_num_status(full_analysis):
    """获取号码冷热状态"""
    c = full_analysis['hot_cold']['full']
    avg = full_analysis['avg']
    hot = max(avg + HOT_COLD_FACTOR, 5)
    cold = min(avg - HOT_COLD_FACTOR, avg * 0.5)
    status_dict = {}
    for n in range(1, 81):
        cnt = c[n]
        r = n % 3
        st = "hot" if cnt >= hot else "cold" if cnt <= cold else "warm"
        status_dict[n] = {"st": st, "road": f"{r}路" if r != 0 else "0路", "cnt": cnt}
    return status_dict  
    # ====================== 数据分析核心引擎 ======================
def analyze_full_data(df, window=None):
    """全量数据分析，支持指定周期窗口"""
    # 边界处理
    if df.empty:
        raise ValueError("无有效开奖数据")
    data = df.head(window).copy() if window else df.copy()
    # 提取每期号码列表
    num_list = [[int(x) for x in row.iloc[1:21].tolist()] for _, row in data.iterrows()]
    flat_nums = [n for p in num_list for n in p]
    total_period = len(num_list)
    avg_occur = len(flat_nums) / 80  # 号码平均出现次数

    return {
        "hot_cold": calc_hot_cold(flat_nums),
        "miss_analysis": calc_miss_analysis(num_list, total_period),
        "co_occur_matrix": calc_co_occur(num_list),
        "follow_matrix": calc_follow(num_list),
        "road": calc_road(flat_nums),
        "zone": calc_zone(flat_nums),
        "con": calc_con(num_list),
        "nums_list": num_list,
        "flat": flat_nums,
        "total": total_period,
        "avg": avg_occur
    }

def calc_hot_cold(flat_nums):
    """计算号码冷热分布"""
    count = Counter(flat_nums)
    # 补全1-80所有号码，默认0次
    full_count = {n: count.get(n, 0) for n in range(1, 81)}
    return {
        "hot_top10": count.most_common(10),
        "cold_top10": count.most_common()[-10:][::-1],
        "full": full_count
    }

def calc_miss_analysis(num_list, total_period):
    """计算号码遗漏分析"""
    last_appear = {}  # 号码上次出现的索引
    miss_count = {}   # 当前遗漏期数
    avg_miss = {}     # 平均遗漏期数
    max_miss = {}     # 最大遗漏期数
    miss_history = defaultdict(list)  # 历史遗漏记录

    for idx, nums in enumerate(num_list):
        for n in nums:
            if n in last_appear:
                miss_history[n].append(idx - last_appear[n])
            last_appear[n] = idx

    # 补全所有号码的遗漏数据
    for n in range(1, 81):
        # 当前遗漏期数
        miss_count[n] = total_period - 1 - last_appear.get(n, -1)
        # 平均/最大遗漏
        history = miss_history[n]
        avg_miss[n] = round(np.mean(history), 1) if history else 0
        max_miss[n] = max(history) if history else 0

    # 构造遗漏DataFrame
    miss_df = pd.DataFrame({
        "号码": range(1, 81),
        "当前遗漏": [miss_count[n] for n in range(1, 81)],
        "平均遗漏": [f"{avg_miss[n]:.1f}" for n in range(1, 81)],
        "最大遗漏": [max_miss[n] for n in range(1, 81)],
        "出现次数": [len(miss_history[n]) + 1 if n in last_appear else 0 for n in range(1, 81)],
        "回补率%": [f"{min(100, round(miss_count[n]/avg_miss[n]*100, 1)) if avg_miss[n] > 0 else 0.0}" for n in range(1, 81)]
    }).sort_values("当前遗漏", ascending=False).reset_index(drop=True)

    return {
        "miss_df": miss_df,
        "mi": miss_count,
        "mc": avg_miss,
        "ma": max_miss
    }

def calc_co_occur(num_list):
    """计算同期同现矩阵（跟随号）"""
    co_count = defaultdict(int)
    for nums in num_list:
        sorted_nums = sorted(nums)
        for i in range(20):
            for j in range(i + 1, 20):
                co_count[(sorted_nums[i], sorted_nums[j])] += 1
    return {
        "dict": co_count,
        "top10": sorted(co_count.items(), key=lambda x: x[1], reverse=True)[:10]
    }

def calc_follow(num_list):
    """计算跨期相随矩阵（N期开A，N+1期开B）"""
    follow_count = defaultdict(int)
    for i in range(1, len(num_list)):
        pre_nums = num_list[i-1]
        curr_nums = num_list[i]
        for a in pre_nums:
            for b in curr_nums:
                follow_count[(a, b)] += 1
    return {
        "dict": follow_count,
        "top10": sorted(follow_count.items(), key=lambda x: x[1], reverse=True)[:10]
    }

def calc_road(flat_nums):
    """计算012路分布"""
    r0 = sum(1 for n in flat_nums if n % 3 == 0)
    r1 = sum(1 for n in flat_nums if n % 3 == 1)
    r2 = sum(1 for n in flat_nums if n % 3 == 2)
    total = len(flat_nums) if len(flat_nums) > 0 else 1
    return {
        "r0": r0, "r1": r1, "r2": r2,
        "r0r": f"{r0/total*100:.1f}%",
        "r1r": f"{r1/total*100:.1f}%",
        "r2r": f"{r2/total*100:.1f}%"
    }

def calc_zone(flat_nums):
    """计算区间分布"""
    z1 = sum(1 for n in flat_nums if 1 <= n <= 20)
    z2 = sum(1 for n in flat_nums if 21 <= n <= 40)
    z3 = sum(1 for n in flat_nums if 41 <= n <= 60)
    z4 = sum(1 for n in flat_nums if 61 <= n <= 80)
    total = len(flat_nums) if len(flat_nums) > 0 else 1
    return {
        "z1": z1, "z2": z2, "z3": z3, "z4": z4,
        "z1r": f"{z1/total*100:.1f}%",
        "z2r": f"{z2/total*100:.1f}%",
        "z3r": f"{z3/total*100:.1f}%",
        "z4r": f"{z4/total*100:.1f}%"
    }

def calc_con(num_list):
    """计算连号分布"""
    con_count_list = []
    for nums in num_list:
        sorted_nums = sorted(nums)
        con_cnt = 0
        i = 0
        while i < 19:
            if sorted_nums[i+1] == sorted_nums[i] + 1:
                con_cnt += 1
                while i < 19 and sorted_nums[i+1] == sorted_nums[i] + 1:
                    i += 1
            i += 1
        con_count_list.append(con_cnt)
    return {
        "avg": round(np.mean(con_count_list), 2) if con_count_list else 0,
        "max": max(con_count_list) if con_count_list else 0,
        "min": min(con_count_list) if con_count_list else 0
    }  
    # ====================== 号码结构计算核心函数 ======================
def calc_number_structure(numbers, prev_numbers=None):
    """计算单期号码结构"""
    numbers = sorted([int(n) for n in numbers])
    prev_numbers = [int(n) for n in prev_numbers] if prev_numbers is not None else None

    # 基础指标
    odd = sum(n % 2 for n in numbers)
    even = 20 - odd
    small = sum(1 for n in numbers if n <= 40)
    large = 20 - small
    r0 = sum(1 for n in numbers if n % 3 == 0)
    r1 = sum(1 for n in numbers if n % 3 == 1)
    r2 = sum(1 for n in numbers if n % 3 == 2)
    prime = sum(1 for n in numbers if n in PRIME_NUMBERS)
    composite = 20 - prime
    sum_val = sum(numbers)
    span = numbers[-1] - numbers[0]

    # 连号计算
    con_list = []
    i = 0
    while i < 19:
        if numbers[i+1] == numbers[i] + 1:
            start = numbers[i]
            while i < 19 and numbers[i+1] == numbers[i] + 1:
                i += 1
            con_list.append(f"{start}-{numbers[i]}")
        i += 1

    # 跨期指标
    repeat_nums = [n for n in numbers if n in prev_numbers] if prev_numbers else []
    oblique_nums = [n for n in numbers if (n-1 in prev_numbers) or (n+1 in prev_numbers)] if prev_numbers else []

    # 同尾计算
    tail_count = Counter([n % 10 for n in numbers])
    tail_dict = {t: [x for x in numbers if x % 10 == t] for t, c in tail_count.items() if c >= 2}

    # 区间分布
    z1 = sum(1 for n in numbers if 1 <= n <= 20)
    z2 = sum(1 for n in numbers if 21 <= n <= 40)
    z3 = sum(1 for n in numbers if 41 <= n <= 60)
    z4 = sum(1 for n in numbers if 61 <= n <= 80)

    return {
        "nums": numbers, "odd": odd, "even": even, "oe": f"{odd}:{even}",
        "small": small, "large": large, "sl": f"{small}:{large}",
        "r0": r0, "r1": r1, "r2": r2, "road": f"{r0}:{r1}:{r2}",
        "prime": prime, "composite": composite, "pc": f"{prime}:{composite}",
        "sum": sum_val, "span": span, "con": con_list, "con_cnt": len(con_list),
        "repeat": repeat_nums, "repeat_cnt": len(repeat_nums),
        "oblique": oblique_nums, "oblique_cnt": len(oblique_nums),
        "tail": tail_dict, "tail_cnt": len(tail_dict),
        "z1": z1, "z2": z2, "z3": z3, "z4": z4
    }

def generate_deep_review(nums, prev_nums=None, period="未知"):
    """生成单期深度复盘结果"""
    structure = calc_number_structure(nums, prev_nums)
    return {"period": period, **structure}

def generate_leveled_pool(history_nums, co_occur_dict, follow_dict, num_status_dict, max_predict_count=30):
    """生成三级预测号码池"""
    if not history_nums or len(history_nums) < 1:
        return {"l1": [], "l2": [], "l3": [], "co": {}, "follow": {}}
    
    l1_base = set(history_nums[-1])
    l2_result = set()
    l3_result = set()
    co_map = defaultdict(list)
    follow_map = defaultdict(list)

    # 生成二级相随号（基于同期同现）
    try:
        co_dict = co_occur_dict if isinstance(co_occur_dict, dict) else {}
        for n in l1_base:
            valid_list = []
            for k, cnt in co_dict.items():
                if not isinstance(k, tuple) or len(k)!=2: continue
                a,b = k
                if (a == n and b not in l1_base) or (b == n and a not in l1_base):
                    match_num = b if a == n else a
                    valid_list.append((match_num, cnt))
                    co_map[n].append((match_num, cnt))
            # 取Top3
            for match_num, cnt in sorted(valid_list, key=lambda x:x[1], reverse=True)[:3]:
                l2_result.add(match_num)
    except Exception:
        pass

    # 生成三级跟随号（基于跨期相随）
    try:
        follow_dict_valid = follow_dict if isinstance(follow_dict, dict) else {}
        for n in l2_result:
            valid_list = []
            for k, cnt in follow_dict_valid.items():
                if not isinstance(k, tuple) or len(k)!=2: continue
                a,b = k
                if a == n and b not in l1_base and b not in l2_result:
                    valid_list.append((b, cnt))
                    follow_map[n].append((b, cnt))
            # 取Top3
            for match_num, cnt in sorted(valid_list, key=lambda x:x[1], reverse=True)[:3]:
                l3_result.add(match_num)
    except Exception:
        pass

    # 排序+数量控制
    l2_sorted = sorted(l2_result, key=lambda x: (-num_status_dict[x]["cnt"], x))
    l3_sorted = sorted(l3_result, key=lambda x: (-num_status_dict[x]["cnt"], x))
    if len(l2_sorted) + len(l3_sorted) > max_predict_count:
        l2_max = min(len(l2_sorted), int(max_predict_count*0.7))
        l3_max = max_predict_count - l2_max
        l2_sorted, l3_sorted = l2_sorted[:l2_max], l3_sorted[:l3_max]
        
    return {
        "l1": list(l1_base), "l2": l2_sorted, "l3": l3_sorted,
        "co": co_map, "follow": follow_map
    }

# ====================== 4铁律组合生成核心函数（修复所有bug） ======================
@st.cache_data(ttl=0)  # 动态数据不缓存，每次生成都是新的
def build_iron_rule_combination(
    predict_pool, full_candidate_pool, two_con, three_con, last_real_nums,
    hot12_list, hot24_list, df_back, need_cnt, group_cnt, seed_key, max_overlap=2
):
    """
    4铁律合规组合生成
    铁律1：剔除三期连开必杀号，两期连开降权
    铁律2：与上期开奖号重合率≤20%
    铁律3：组间核心号码重叠度≤max_overlap
    铁律4：奇偶/大小均衡，预测池优先
    """
    final_combs = []
    # 固定随机种子，保证同参数生成结果一致
    np.random.seed(hash(seed_key) % 2**32)

    try:
        # 步骤1：合规过滤
        # 剔除三期连开号
        predict_pool_clean = list(set([n for n in predict_pool if n not in three_con]))
        full_candidate_clean = list(set([n for n in full_candidate_pool if n not in three_con]))
        
        # 预测池优先，不足则从候选池补充
        if len(predict_pool_clean) >= need_cnt:
            base_candidate = predict_pool_clean
        else:
            supplement_nums = [n for n in full_candidate_clean if n not in predict_pool_clean]
            base_candidate = predict_pool_clean + supplement_nums
        
        # 候选池不足，直接返回空
        if len(base_candidate) < need_cnt:
            return final_combs

        # 步骤2：号码评分
        # 高回补号码
        high_back_nums = set()
        if not df_back.empty and "回补率%" in df_back.columns:
            df_back["temp_num"] = df_back["回补率%"].astype(str).str.replace("%", "").astype(float)
            high_back_nums = set(df_back[df_back["temp_num"] >= 80]["号码"].tolist())

        # 评分规则
        score_dict = {}
        hot12_set = set(hot12_list)
        hot24_set = set(hot24_list)
        predict_set = set(predict_pool_clean)
        two_con_set = set(two_con)

        for n in base_candidate:
            score = 0
            if n in predict_set: score += 100  # 预测池优先
            if n in hot24_set: score += 50    # 24期热号
            if n in hot12_set: score += 30    # 12期热号
            if n in high_back_nums: score += 20 # 高回补
            if n in two_con_set: score -= 50  # 两期连开降权
            score_dict[n] = score

        # 按评分排序
        sorted_nums = sorted(base_candidate, key=lambda x: (-score_dict.get(x, 0), x))
        idx = 0
        max_try = 200  # 最大尝试次数，避免死循环
        last_num_len = len(last_real_nums) if len(last_real_nums) > 0 else 20

        # 步骤3：生成合规组合
        while len(final_combs) < group_cnt and idx < max_try and idx + need_cnt <= len(sorted_nums):
            temp_comb = sorted_nums[idx:idx+need_cnt]
            # 铁律2：与上期重合率≤20%
            overlap = set(temp_comb) & set(last_real_nums)
            overlap_rate = len(overlap) / last_num_len
            # 铁律3：组间重叠度≤max_overlap
            overlap_with_exist = False
            for exist_comb in final_combs:
                if len(set(temp_comb) & set(exist_comb)) > max_overlap:
                    overlap_with_exist = True
                    break
            # 奇偶/大小均衡校验
            odd_rate = sum(1 for n in temp_comb if n%2==1) / need_cnt
            small_rate = sum(1 for n in temp_comb if n<=40) / need_cnt
            balance_ok = 0.3 <= odd_rate <= 0.7 and 0.3 <= small_rate <= 0.7
            
            # 全部合规则加入结果
            if overlap_rate <= 0.20 and temp_comb not in final_combs and not overlap_with_exist and balance_ok:
                final_combs.append(temp_comb)
            idx += 2

    except Exception as e:
        st.error(f"组合生成失败：{str(e)}")
    
    return final_combs  
    # ====================== 辅助计算工具函数 ======================
def calc_occur_rate(df, window=10):
    """计算指定周期内号码出现次数"""
    try:
        data = df.head(window).copy()
        num_list = [[int(x) for x in row.iloc[1:21].tolist()] for _, row in data.iterrows()]
        flat_nums = [n for p in num_list for n in p]
        occur_count = Counter(flat_nums)
        full_occur = {n: occur_count.get(n, 0) for n in range(1, 81)}
        return full_occur, num_list
    except Exception as e:
        st.error(f"号码出现率计算失败：{str(e)}")
        return {n:0 for n in range(1,81)}, []

def calc_follow_probability(df, target_nums, min_occur=4, min_rate=0.4):
    """计算目标号码的高概率跨期相随号"""
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
    """计算欠开区间及对应号码"""
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

# ====================== 多周期&跨期对比共用工具函数 ======================
def get_period_follow_data(df, period_window, target_num=None):
    """获取指定周期的同期跟随号数据"""
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
    """获取指定周期的跨期相随号数据"""
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
    """两期对比核心函数，彻底解决所有报错"""
    try:
        # 临时生成期号数字列，不修改原df
        df_temp = df.copy()
        df_temp['period_num'] = df_temp['period'].astype(int)
        df_sorted = df_temp.sort_values("period_num", ascending=False).reset_index(drop=True)
        
        # 校验期号是否存在
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

        # 跳过已存在的文件
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
            # 1. 单期复盘
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

            # 2. 跨期预测池生成
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

            # 3. 4铁律选号组合生成
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
    # ====================== 全局数据初始化 ======================
df = load_data_cached()
total_period = len(df)

# 侧边栏配置
with st.sidebar:
    st.title("🎰快乐8数据分析系统")
    st.divider()
    st.metric("总收录期数", f"{total_period}期")
    if total_period > 0:
        st.caption(f"最新期号：{df.iloc[0]['period']}")
        st.caption(f"最早期号：{df.iloc[-1]['period']}")
    st.divider()
    if st.button("🔄 清除缓存刷新数据", use_container_width=True):
        load_data_cached.clear()
        get_full_analysis_cached.clear()
        st.rerun()
    st.divider()
    st.warning("仅历史数据统计娱乐，不构成购彩建议，理性购彩！")

# 标签页定义（变量名和标签一一对应，顺序严格，tab4单独拆分）
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏠 首页", 
    "📋 号码库", 
    "📊 多周期分析", 
    "🔮 双流派选号", 
    "📝 单期复盘", 
    "🔄 跨期对比", 
    "⚙️ 系统设置", 
    "📦 全量批量复盘"
])   
# ========== Tab1 首页 ==========
with tab1:
    st.title("🎰 福彩快乐8专业数据分析系统")
    st.subheader(f"当前收录：{total_period}期 | 双流派升级终版 | 4铁律合规组合生成")
    st.error("⚠️ 开奖完全随机，本系统仅历史数据统计娱乐，不构成任何购彩建议！")
    if total_period > 0:
        latest_row = df.iloc[0]
        st.divider()
        st.subheader(f"📢 最新{latest_row['period']}期开奖号码")
        latest_nums = [int(x) for x in latest_row.iloc[1:21].tolist()]
        # 加载冷热状态
        full_ana = get_full_analysis_cached(df)
        num_status = get_num_status(full_ana)
        nums_html = " ".join([fmt_num(n, num_status) for n in sorted(latest_nums)])
        st.markdown(f"### {nums_html}", unsafe_allow_html=True)

        # 核心指标概览
        st.divider()
        st.subheader("📈 核心数据概览")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("近10期连号平均组数", full_ana["con"]["avg"])
        with c2:
            st.metric("0路号码占比", full_ana["road"]["r0r"])
        with c3:
            st.metric("1路号码占比", full_ana["road"]["r1r"])
        with c4:
            st.metric("2路号码占比", full_ana["road"]["r2r"])
        
        # 区间分布
        st.divider()
        st.subheader("📊 全量区间分布")
        zone_df = pd.DataFrame([{
            "区间": "1-20",
            "累计出号": full_ana["zone"]["z1"],
            "占比": full_ana["zone"]["z1r"]
        },{
            "区间": "21-40",
            "累计出号": full_ana["zone"]["z2"],
            "占比": full_ana["zone"]["z2r"]
        },{
            "区间": "41-60",
            "累计出号": full_ana["zone"]["z3"],
            "占比": full_ana["zone"]["z3r"]
        },{
            "区间": "61-80",
            "累计出号": full_ana["zone"]["z4"],
            "占比": full_ana["zone"]["z4r"]
        }])
        st.dataframe(zone_df, hide_index=True, use_container_width=True)

# ========== Tab2 号码库管理 ==========
with tab2:
    st.header("📋 开奖号码库管理")
    st.info("⚠️ 系统基准75期原始数据禁止删除，仅可删除手动新增的期号")
    # 新增数据表单
    with st.form("add_data_form", border=True):
        c1, c2 = st.columns(2)
        with c1:
            input_period = st.text_input("期号", placeholder="例：2026076", key="add_period")
        with c2:
            input_nums = st.text_input("开奖号码（20个数字，空格分隔）", placeholder="例：01 02 03 ... 80", key="add_nums")
        submit_add = st.form_submit_button("✅ 保存录入", use_container_width=True, type="primary")
        if submit_add:
            v1, m1 = validate_period_unique(input_period, df)
            if not v1:
                st.error(m1)
            else:
                v2, m2 = validate_numbers(input_nums.split())
                if not v2:
                    st.error(m2)
                else:
                    if save_new_data(input_period, m2):
                        st.success("✅ 数据录入成功！")
                        load_data_cached.clear()
                        get_full_analysis_cached.clear()
                        st.rerun()
    st.divider()
    # 删除数据表单
    with st.form("del_data_form", border=True):
        del_period = st.selectbox("选择要删除的期号", df['period'].tolist(), key="del_period")
        submit_del = st.form_submit_button("⚠️ 确认删除", use_container_width=True, type="secondary")
        if submit_del:
            new_df, del_success, del_msg = delete_period_data(del_period, df)
            if del_success:
                st.success(del_msg)
                load_data_cached.clear()
                get_full_analysis_cached.clear()
                st.rerun()
            else:
                st.error(del_msg)
    st.divider()
    # 数据表格展示
    st.subheader("📋 全量开奖数据")
    st.dataframe(df, use_container_width=True, height=400)

# ========== Tab3 多周期相随号&跟随号深度分析（完整补全） ==========
with tab3:
    st.header("📊 多周期相随号&跟随号深度分析")
    st.info("周期统一：20/50/100/150期 | 相随号=跨期N→N+1跟随 | 跟随号=同期同频出现 | 数据全同源，无逻辑冲突")
    st.divider()

    # 全局周期配置（与模块6函数完全匹配）
    PERIOD_LIST = [20, 50, 100, 150]
    period_tab_list = st.tabs([f"近{p}期" for p in PERIOD_LIST])
    period_df = load_data_cached()

    # 按周期生成Tab内容，循环内组件key唯一，杜绝重复ID报错
    for idx, period_window in enumerate(PERIOD_LIST):
        with period_tab_list[idx]:
            st.subheader(f"📈 近{period_window}期 相随号&跟随号全景数据")
            # 子Tab分相随号/跟随号/单号码查询，key唯一
            xiang_sui_tab, gen_sui_tab, num_query_tab = st.tabs([
                f"🔗 跨期相随号(N→N+1)_{period_window}", 
                f"📌 同期跟随号(同频出现)_{period_window}", 
                f"🔍 单号码精准查询_{period_window}"
            ])
            
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
                query_num = st.number_input(
                    "选择要查询的号码", 
                    min_value=1, max_value=80, 
                    value=1, step=1, 
                    key=f"query_num_{period_window}"
                )
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

# ========== Tab4 双流派选号+开奖核对中心（完整核心模块，零报错） ==========
with tab4:
    st.header("🔮 双流派选号+开奖核对中心 | 4铁律合规体系")
    st.info("双流派并行：热号惯性流派+冷号回补流派 | 全流程4铁律校验 | 自动存档+开奖核对全闭环")
    st.divider()

    # 全局基础数据预加载（所有变量均已在前置模块定义，无未定义报错）
    df = load_data_cached()
    total = len(df)
    if total < 3:
        st.error("数据不足3期，无法进行选号分析，请补充开奖数据")
        st.stop()

    # 1. 选择预测目标期号
    st.subheader("📌 选择预测目标期号")
    latest_period = df.iloc[0]['period']
    try:
        target_period = str(int(latest_period) + 1).zfill(7)
    except:
        target_period = "2026076"
    tar_p = st.text_input("预测目标期号", value=target_period, key="target_period_tab4")
    st.divider()

    # 2. 预计算核心基础数据（仅计算1次，提升性能）
    with st.spinner("正在加载核心分析数据..."):
        # 多周期出现率
        occur_10, num_list_10 = calc_occur_rate(df, 10)
        occur_5, num_list_5 = calc_occur_rate(df, 5)
        occur_3, num_list_3 = calc_occur_rate(df, 3)
        # 全量分析
        full_analysis_20 = get_full_analysis_cached(df, 20)
        full_analysis_all = get_full_analysis_cached(df)
        num_status_dict = get_num_status(full_analysis_all)
        # 连号数据
        last_1 = set(df.iloc[0].iloc[1:21].tolist())
        last_2 = set(df.iloc[1].iloc[1:21].tolist())
        last_3 = set(df.iloc[2].iloc[1:21].tolist())
        two_con = list(last_1 & last_2)  # 两期连开号
        three_con = list(last_1 & last_2 & last_3)  # 三期连开必杀号
        last_pre_real = list(last_1)  # 上期开奖号
        # 高概率相随号
        high_prob_follow_nums = calc_follow_probability(df, last_pre_real)
        # 欠开区间
        under_zone_nums, zone_occur_3 = get_under_open_zone(num_list_3)
        # 热号列表
        hot10_plain = [x[0] for x in full_analysis_20["hot_cold"]["hot_top10"]]
        hot20_plain = [x[0] for x in full_analysis_all["hot_cold"]["hot_top10"]]
        df_back_plain = full_analysis_20["miss_analysis"]["miss_df"]
        # 全量候选池
        full_candidate_pool = list(range(1, 81))
        # 预测池加载
        predict_df = load_predict_num(tar_p)
        predict_pool = []
        if predict_df is not None and not predict_df.empty and "号码" in predict_df.columns:
            l2_nums = predict_df[predict_df["候选等级"] == "二级相随号"]["号码"].tolist()
            l3_nums = predict_df[predict_df["候选等级"] == "三级跟随号"]["号码"].tolist()
            predict_pool = l2_nums + l3_nums
            st.success(f"✅ 已加载{tar_p}期预测池，共{len(predict_pool)}个号码")
        else:
            st.warning(f"⚠️ 未找到{tar_p}期预测池，请先在「跨期对比」模块生成")
        # 开奖核对用真实号码
        real_check_nums = []
        if tar_p in df["period"].values:
            real_check_row = df[df["period"] == tar_p].iloc[0]
            real_check_nums = [int(x) for x in real_check_row.iloc[1:21].tolist()]

    # 主标签页拆分
    main_tab1, main_tab2, main_tab3, main_tab4, main_tab5 = st.tabs([
        "📈 行情判断",
        "🔥 热号惯性流派",
        "🧊 冷号回补流派",
        "📊 开奖核对中心",
        "💡 双流派复盘优化"
    ])

    # ========== 子标签1：行情判断 ==========
    with main_tab1:
        st.header("📈 行情趋势判断中心")
        st.info("基于近3期区间出号、冷热分布、连号情况，判断当前行情偏向")
        st.divider()

        # 近3期区间出号统计
        st.subheader("📊 近3期区间出号统计")
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

        # 行情偏向判断
        st.divider()
        st.subheader("📝 行情偏向判断")
        hot_total = sum(1 for n in last_pre_real if num_status_dict[n]["st"] == "hot")
        cold_total = sum(1 for n in last_pre_real if num_status_dict[n]["st"] == "cold")
        if hot_total >= 12:
            st.success("✅ 当前行情偏向【热号惯性】，强者恒强，优先使用热号惯性流派")
        elif cold_total >= 12:
            st.success("✅ 当前行情偏向【冷号回补】，均值回归，优先使用冷号回补流派")
        else:
            st.info("⚖️ 当前行情均衡，建议双流派均衡配置，分散风险")
        
        # 核心指标展示
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("上期热号个数", hot_total)
        with c2:
            st.metric("上期冷号个数", cold_total)
        with c3:
            st.metric("上期连号组数", full_analysis_all["con"]["avg"])

    # ========== 子标签2：热号惯性流派 ==========
    with main_tab2:
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

    # ========== 子标签3：冷号回补流派 ==========
    with main_tab3:
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

    # ========== 子标签4：开奖核对中心 ==========
    with main_tab4:
        st.header("📊 开奖核对中心｜正式号 vs 预测池 vs 双流派号码")
        st.info("功能：展示本期正式开奖号码 → 对比预测池号码、热号流派号码、冷号流派号码 → 自动对比解析")
        
        # 获取期号列表并倒序排列
        all_p_list = sorted(df["period"].astype(str).tolist(), reverse=True)
        check_p = st.selectbox("选择需要核对的期号", all_p_list, key="final_check_tab4")

        # 1. 加载本期正式开奖号码
        real_nums = []
        match_idx = df[df["period"] == check_p].index
        if len(match_idx) > 0:
            row = df.loc[match_idx[0]]
            real_nums = sorted([int(x) for x in row.iloc[1:21].tolist()])

        st.subheader("🔴 本期正式开奖号码")
        if real_nums:
            real_str = "  ".join(f"{n:02d}" for n in real_nums)
            st.markdown(f"### {real_str}", unsafe_allow_html=True)
        else:
            st.error("未找到该期正式开奖号码")
            st.stop()

        # 2. 加载预测池
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

        # 3. 加载热号流派所有筛选号码
        st.divider()
        st.subheader("🔥 热号惯性流派 · 全量筛选号码")
        all_comb_df = load_all_select_comb()
        hot_nums_set = set()
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

        # 4. 加载冷号流派所有筛选号码
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

        # 5. 自动对比命中统计
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

        # 6. 自动智能解析
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

                # 7. 全组合命中明细
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
                        except Exception:
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
    with main_tab5:
        st.header("💡 双流派复盘优化中心")
        st.info("自动统计历史期数命中率，给出流派优化建议")
        st.divider()
        all_comb_df = load_all_select_comb()
        if all_comb_df.empty:
            st.warning("暂无历史组合存档数据，无法复盘")
        else:
            period_list = sorted(all_comb_df["期号"].unique(), reverse=True)
            review_period = st.multiselect("选择复盘期号", period_list, default=period_list[:5] if len(period_list)>=5 else period_list)
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
                        period = r["期号"]
                        if period not in df["period"].values:
                            continue
                        real_row = df[df["period"]==period].iloc[0]
                        real_nums = [int(x) for x in real_row.iloc[1:21].tolist()]
                        comb_nums = [int(x) for x in str(r["选号号码"]).split()]
                        hit_res = calc_match_rate(comb_nums, real_nums)
                        hit_list.append(hit_res["正确率%"])
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
    review_mode = st.radio("选择复盘方式", ["选择历史期号", "手动录入新期号码"], horizontal=True, key="review_mode_tab5")

    if review_mode == "选择历史期号":
        period_list = df["period"].tolist()
        selected_period = st.selectbox("选择要复盘的期号", period_list, key="selected_period_tab5")
        
        if st.button("生成深度复盘报告", use_container_width=True, type="primary", key="gen_review_btn_tab5"):
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
        with st.form("manual_review_form", border=True, key="manual_review_form_tab5"):
            manual_period = st.text_input("期号（如：2026089）", placeholder="例：2026089", key="manual_period_tab5")
            manual_nums = st.text_input("开奖号码（20个数字，空格分隔）", placeholder="例：08 09 13 14 ... 80", key="manual_nums_tab5")
            submit_manual = st.form_submit_button("生成复盘报告", use_container_width=True, type="primary")

            if submit_manual:
                if not manual_period or not manual_period.isdigit():
                    st.error("❌ 期号必须为非空纯数字！")
                else:
                    num_valid, num_msg = validate_numbers(manual_nums.strip().split())
                    if not num_valid:
                        st.error(f"❌ {num_msg}")
                    else:
                        prev_nums = [int(x) for x in df.iloc[0].iloc[1:21].tolist()] if total_period > 0 else None
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
                            if st.button("✅ 一键保存到号码库", type="primary", use_container_width=True, key="save_manual_btn_tab5"):
                                save_success = save_new_data(manual_period, num_msg)
                                if save_success:
                                    st.success(f"✅ 成功入库{manual_period}期数据！")
                                    load_data_cached.clear()
                                    get_full_analysis_cached.clear()
                                    st.rerun()    

# ========== Tab6 跨期对比与下期预测号码池生成 ==========
with tab6:
    st.header("🔄 跨期对比与下期预测号码池生成")
    st.info("✅ 需求全覆盖：N与N-1期对比 | 基底随期号变动 | 二级随基底生成 | 三级随二级生成 | 同源近20期相随/跟随数据")
    st.divider()

    period_list = df["period"].tolist()
    if not period_list:
        st.error("暂无开奖数据，请先初始化号码库")
        st.stop()

    # 1. 选择分析期号N
    st.subheader("📌 选择分析期号N")
    select_period_N = st.selectbox("本期期号N", period_list, index=0, key="cross_period_N_tab6")
    st.divider()

    # 2. 自动生成N与N-1期对比
    with st.spinner("正在生成两期对比数据..."):
        compare_result = get_two_period_compare(df, select_period_N)
        if "error" in compare_result:
            st.error(compare_result["error"])
            st.stop()

        base_nums = compare_result["N_nums"]

        # 两期对比表格
        st.subheader(f"📊 {select_period_N}期 VS {compare_result['N_1_period']}期 核心指标对比")
        compare_table_df = pd.DataFrame(compare_result["compare_table"][1:], columns=compare_result["compare_table"][0])
        st.dataframe(compare_table_df, hide_index=True, use_container_width=True)

        # 文字总结
        st.subheader("📝 两期对比总结")
        for txt in compare_result["summary"]:
            st.info(txt)

        # 基底号码展示
        st.divider()
        st.subheader(f"🔴 本期{select_period_N}期 基底参考号")
        full_ana_20 = get_full_analysis_cached(df, window=20)
        num_status_dict = get_num_status(full_ana_20)
        base_html = " ".join([fmt_num(n, num_status_dict) for n in sorted(base_nums)])
        st.markdown(base_html, unsafe_allow_html=True)

    # 3. 生成二级、三级号码
    st.divider()
    if st.button("🚀 生成二级相随层+三级跟随层", use_container_width=True, type="primary", key="gen_level_btn_tab6"):
        with st.spinner("生成中..."):
            # 二级相随层
            st.divider()
            st.subheader("🟡 二级相随层")
            level2_result = set()
            level2_detail = []

            for base_num in base_nums:
                xs = get_period_xiang_sui_data(df, 20, target_num=base_num)["target_xiang_sui"]
                if xs:
                    top3 = xs[:3]
                    for sui_num, cnt in top3:
                        if sui_num not in base_nums:
                            level2_result.add(sui_num)
                            level2_detail.append({
                                "基底号码": f"{base_num:02d}",
                                "二级相随号": f"{sui_num:02d}",
                                "近20期相随次数": cnt,
                                "概率": f"{round(cnt/20*100,1)}%"
                            })

            level2_sorted = sorted(level2_result)
            if level2_detail:
                st.dataframe(pd.DataFrame(level2_detail), hide_index=True, use_container_width=True)
                st.markdown(f"**最终二级池：** {' '.join(f'{x:02d}' for x in level2_sorted)}")
            else:
                st.warning("无二级相随号")

            # 三级跟随层
            st.divider()
            st.subheader("🟢 三级跟随层")
            level3_result = set()
            level3_detail = []

            for l2_num in level2_sorted:
                xs = get_period_xiang_sui_data(df, 20, target_num=l2_num)["target_xiang_sui"]
                if xs:
                    top3 = xs[:3]
                    for sui_num, cnt in top3:
                        if sui_num not in base_nums and sui_num not in level2_sorted:
                            level3_result.add(sui_num)
                            level3_detail.append({
                                "二级号码": f"{l2_num:02d}",
                                "三级跟随号": f"{sui_num:02d}",
                                "近20期相随次数": cnt,
                                "概率": f"{round(cnt/20*100,1)}%"
                            })

            level3_sorted = sorted(level3_result)
            if level3_detail:
                st.dataframe(pd.DataFrame(level3_detail), hide_index=True, use_container_width=True)
                st.markdown(f"**最终三级池：** {' '.join(f'{x:02d}' for x in level3_sorted)}")
            else:
                st.warning("无三级跟随号")

            # 自动存档
            st.divider()
            try:
                target_period = str(int(select_period_N) + 1).zfill(7)
                save_predict_num(
                    target_period=target_period,
                    data_end_period=select_period_N,
                    level2_list=level2_sorted,
                    level3_list=level3_sorted
                )
                st.success(f"✅ {target_period} 预测池已保存，可在双流派选号模块直接调用")
            except Exception as e:
                st.error(f"存档失败：{e}")

st.caption("🔒 层间完全隔离，同源近20期数据，无逻辑冲突")

# ========== Tab7 系统设置页（数据管理+备份迁移+系统重置） ==========
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
            use_container_width=True,
            key="download_raw_csv_tab7"
        )
    else:
        st.warning("数据文件不存在，请先初始化系统")

    st.divider()
    # 2. 自动生成存档文件管理
    st.subheader("📂 自动生成存档文件管理（支持单删/批量删除）")
    if os.path.exists(SAVE_DIR):
        save_files = os.listdir(SAVE_DIR)
        if save_files:
            st.write(f"当前自动生成存档总数：{len(save_files)}个")
            # 单个文件删除
            del_single_file = st.selectbox("选择单个文件删除", save_files, key="del_single_file_tab7")
            if st.button("删除选中文件", use_container_width=True, type="secondary", key="del_single_btn_tab7"):
                del_success, del_msg = delete_single_archive_file(del_single_file)
                if del_success:
                    st.success(del_msg)
                    st.rerun()
                else:
                    st.error(del_msg)
            st.divider()
            # 批量删除所有存档
            if st.button("清空所有自动生成的预测号/选号组合存档", use_container_width=True, type="secondary", key="del_all_btn_tab7"):
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
    if st.button("清空所有批量复盘生成的存档数据", use_container_width=True, type="secondary", key="del_batch_review_tab7"):
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
        st.metric("总收录期数", f"{total_period}期")
    with col_stat2:
        st.metric("最早期号", df.iloc[-1]["period"] if total_period > 0 else "无")
    with col_stat3:
        st.metric("最新期号", df.iloc[0]["period"] if total_period > 0 else "无")
    with col_stat4:
        st.metric("总号码记录数", f"{total_period * 20}个")

    st.divider()
    # 5. 全库一键打包备份/迁移
    st.subheader("💾 全库一键打包备份 | 跨代码/跨电脑迁移专用")
    try:
        zip_name = f"KL8全量外置存档_一键迁移包_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.zip"
        zip_path = os.path.join(ARCHIVE_ROOT, zip_name)

        if st.button("📦 开始打包全部外置数据（适配代码更替/换服务器/换电脑）", use_container_width=True, type="primary", key="pack_zip_btn_tab7"):
            with st.spinner("正在压缩全库存档，请稍候..."):
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(ARCHIVE_ROOT):
                        for file in files:
                            fp = os.path.join(root, file)
                            arcname = os.path.relpath(fp, ARCHIVE_ROOT)
                            zipf.write(fp, arcname)
            st.success("✅ 打包完成！更替代码只需要复制压缩包，新环境解压配置路径即可秒读所有历史数据")
            with open(zip_path, "rb") as f:
                st.download_button(
                    "⬇️ 下载全库迁移压缩包", 
                    f, 
                    file_name=zip_name, 
                    use_container_width=True,
                    key="download_zip_btn_tab7"
                )

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
    st.error("此操作清空增量数据，仅恢复初始75期基准，更替代码无需点这里！")
    with st.form("reset_data_form", border=True, key="reset_data_form_tab7"):
        reset_confirm = st.checkbox("我已知风险，确认重置回原始75期基准数据", key="reset_confirm_tab7")
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
    st.info("自动完成75期全量数据的「单期深度复盘+跨期对比预测池+4铁律选号组合」生成，结果永久存档，后期随时可调用")
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        overwrite_mode = st.checkbox("覆盖已存在的存档数据（增量模式不勾选，全量重算勾选）", value=False, key="overwrite_mode_tab8")
    with c2:
        st.metric("当前可处理总期数", f"{len(df)}期")

    run_batch = st.button("🚀 开始全量自动复盘", use_container_width=True, type="primary", key="run_batch_btn_tab8")
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
                    use_container_width=True,
                    key="download_batch_summary_tab8"
                )

    st.divider()
    st.subheader("📋 历史批量复盘存档查看")
    if os.path.exists(BATCH_REVIEW_SUMMARY):
        history_df = pd.read_csv(BATCH_REVIEW_SUMMARY, encoding="utf-8-sig")
        st.dataframe(history_df, hide_index=True, use_container_width=True, height=300)
        st.subheader("🔍 单期复盘明细查询")
        sel_period = st.selectbox("选择要查看的期号", df["period"].tolist(), key="sel_period_tab8")
        detail_file = os.path.join(BATCH_REVIEW_DETAIL_DIR, f"{sel_period}期_复盘明细.csv")
        if os.path.exists(detail_file):
            detail_df = pd.read_csv(detail_file, encoding="utf-8-sig")
            st.dataframe(detail_df, hide_index=True, use_container_width=True)
            with open(detail_file, "rb") as f:
                st.download_button(
                    f"下载{sel_period}期复盘明细", 
                    f.read(), 
                    file_name=f"{sel_period}期_复盘明细.csv", 
                    mime="text/csv",
                    use_container_width=True,
                    key=f"download_detail_{sel_period}_tab8"
                )
        else:
            st.warning("该期暂无复盘明细，请先执行批量复盘")
    else:
        st.info("暂无批量复盘存档，请先点击「开始全量自动复盘」生成数据")

# ====================== 全局尾部合规声明（完整闭合） ======================
st.divider()
st.markdown('<div style="text-align:center;color:#666;font-size:14px">温馨提示:本系统仅历史数据统计娱乐,彩票开奖完全随机,不构成购彩建议,理性购彩遵守法规</div>', unsafe_allow_html=True)

        