import streamlit as st
import pandas as pd
import numpy as np
import os
from collections import defaultdict, Counter
import itertools
import re
from datetime import datetime
import plotly.express as px

# ==================== 全局配置 ====================
st.set_page_config(
    page_title="快乐8智能分析系统",
    layout="wide",
    page_icon="🎰",
    initial_sidebar_state="collapsed"
)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
DATA_FILE = os.path.join(DATA_DIR, "快八开奖号.csv")
PREDICT_FILE = os.path.join(DATA_DIR, "预测历史存档.csv")
VERIFY_FILE = os.path.join(DATA_DIR, "验证复盘存档.csv")
PRIME_SET = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79}

# ==================== 持久化存档辅助函数 ====================
def save_predict_record(period, hot_pool, cold_pool, wen_pool, core_25, combo_dict):
    try:
        def list2str(lst):
            return ",".join([str(x) for x in lst])
        def combo2str(combo_list):
            return "|".join([",".join([str(x) for x in combo]) for combo in combo_list])
        
        record = {
            "期号": str(period),
            "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "热号池": list2str(hot_pool),
            "冷号池": list2str(cold_pool),
            "稳胆池": list2str(wen_pool),
            "25码核心池": list2str(core_25),
            "11码组合": combo2str(combo_dict['11码']),
            "8码组合": combo2str(combo_dict['8码']),
            "6码组合": combo2str(combo_dict['6码']),
            "3码组合": combo2str(combo_dict['3码'])
        }
        
        if os.path.exists(PREDICT_FILE):
            df_predict = pd.read_csv(PREDICT_FILE, dtype={'期号': str})
            df_predict = df_predict[df_predict['期号'] != str(period)]
            df_predict = pd.concat([df_predict, pd.DataFrame([record])], ignore_index=True)
        else:
            df_predict = pd.DataFrame([record])
        
        df_predict.to_csv(PREDICT_FILE, index=False, encoding='utf-8-sig')
        return True
    except Exception as e:
        st.error(f"预测存档失败: {str(e)}")
        return False

def load_predict_record(period=None):
    try:
        if not os.path.exists(PREDICT_FILE):
            return pd.DataFrame()
        df = pd.read_csv(PREDICT_FILE, dtype={'期号': str})
        if period:
            df = df[df['期号'] == str(period)]
        return df
    except:
        return pd.DataFrame()

def str2list(s):
    if pd.isna(s) or s == "":
        return []
    return [int(x) for x in s.split(",")]

def str2combo(s):
    if pd.isna(s) or s == "":
        return []
    return [tuple([int(x) for x in combo.split(",")]) for combo in s.split("|")]

def save_verify_record(period, open_nums, core_pool_hit, core_pool_hit_rate, best_hit, best_combo, best_type, combo_detail):
    try:
        def list2str(lst):
            return ",".join([str(x) for x in lst])
        
        record = {
            "期号": str(period),
            "复盘时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "开奖号码": list2str(open_nums),
            "核心池命中个数": core_pool_hit,
            "核心池命中率": f"{core_pool_hit_rate:.2%}",
            "最佳组合命中": best_hit,
            "最佳组合内容": list2str(best_combo),
            "最佳组合类型": best_type
        }
        
        if os.path.exists(VERIFY_FILE):
            df_verify = pd.read_csv(VERIFY_FILE, dtype={'期号': str})
            df_verify = df_verify[df_verify['期号'] != str(period)]
            df_verify = pd.concat([df_verify, pd.DataFrame([record])], ignore_index=True)
        else:
            df_verify = pd.DataFrame([record])
        
        df_verify.to_csv(VERIFY_FILE, index=False, encoding='utf-8-sig')
        return True
    except Exception as e:
        st.error(f"复盘存档失败: {str(e)}")
        return False

def load_verify_record(period=None):
    try:
        if not os.path.exists(VERIFY_FILE):
            return pd.DataFrame()
        df = pd.read_csv(VERIFY_FILE, dtype={'期号': str})
        if period:
            df = df[df['期号'] == str(period)]
        return df
    except:
        return pd.DataFrame()

# ==================== 兼容新/旧版Pandas的样式函数 ====================
def style_dataframe(df, highlight_func, subset_cols):
    try:
        return df.style.map(highlight_func, subset=subset_cols)
    except AttributeError:
        try:
            return df.style.applymap(highlight_func, subset=subset_cols)
        except:
            return df

# ==================== 核心缓存函数 ====================
@st.cache_data(ttl=3600)
def init_base_data():
    INITIAL_DATA = [
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
        ["2026015",2,8,9,11,14,17,18,19,27,29,31,34,36,41,55,60,64,70,72,79]
    ]
    if not os.path.exists(DATA_FILE):
        df_init = pd.DataFrame(INITIAL_DATA, columns=["期号"] + [f"N{i}" for i in range(1,21)])
        df_init['期号'] = df_init['期号'].astype(str)
        df_init.to_csv(DATA_FILE, index=False, encoding='utf-8-sig')
    return pd.read_csv(DATA_FILE, encoding='utf-8-sig', dtype={'期号': str})

@st.cache_data(ttl=60)
def load_lottery_data():
    try:
        df = pd.read_csv(DATA_FILE, encoding='utf-8-sig', dtype={'期号': str})
        return df
    except:
        return init_base_data()

def save_lottery_data(df):
    df['期号'] = df['期号'].astype(str)
    df.to_csv(DATA_FILE, index=False, encoding='utf-8-sig')
    load_lottery_data.clear()
    init_base_data.clear()

@st.cache_data
def parse_numbers(input_str):
    nums = re.split(r'[、,，\s]+', input_str.strip())
    nums = [int(n) for n in nums if n.isdigit()]
    nums = sorted(list(set([n for n in nums if 1<=n<=80])))
    return nums

@st.cache_data
def get_period_numbers(df, period):
    try:
        row = df[df['期号'] == str(period)]
        if len(row) == 0:
            return []
        return [int(row[f'N{i}'].values[0]) for i in range(1,21)]
    except:
        return []

@st.cache_data
def calc_base_metrics(numbers):
    if not numbers or len(numbers)!=20:
        return {'奇偶比':'-', '大小比':'-', '质合比':'-', '012路':'-', '和值':0, '跨度':0}
    odd = len([n for n in numbers if n%2==1])
    small = len([n for n in numbers if n<=40])
    prime = len([n for n in numbers if n in PRIME_SET])
    lu0 = len([n for n in numbers if n%3==0])
    lu1 = len([n for n in numbers if n%3==1])
    return {
        '奇偶比': f"{odd}:{20-odd}", '大小比': f"{small}:{20-small}",
        '质合比': f"{prime}:{20-prime}", '012路': f"{lu0}:{lu1}:{20-lu0-lu1}",
        '和值': sum(numbers), '跨度': max(numbers)-min(numbers)
    }

@st.cache_data
def full_cycle_analysis(df, cycle_len):
    try:
        periods = sorted(df['期号'].tolist())
        if len(periods) < cycle_len:
            return None
        cycle_periods = periods[-cycle_len:]
        
        count = Counter()
        miss = {num: cycle_len for num in range(1,81)}
        xiang_sui = defaultdict(list)
        gen_sui = defaultdict(list)
        period_num_map = {}
        
        for i, p in enumerate(cycle_periods):
            nums = get_period_numbers(df, p)
            period_num_map[p] = nums
            count.update(nums)
            
            for num in nums:
                miss[num] = (len(cycle_periods)-1 - i)
            
            if i > 0:
                prev_p = cycle_periods[i-1]
                prev_nums = period_num_map[prev_p]
                for a in prev_nums:
                    xiang_sui[a].extend(nums)
                for b in nums:
                    gen_sui[b].extend([x for x in nums if x!=b])
        
        status = {num: "冷" if m>=10 else "温" if 5<=m<=9 else "热" for num,m in miss.items()}
        
        xiang_sui_prob = defaultdict(dict)
        for a in xiang_sui.keys():
            a_total = count.get(a, 1)
            b_count = Counter(xiang_sui[a])
            for b, cnt in b_count.items():
                xiang_sui_prob[a][b] = round(cnt / a_total * 100, 2)
        
        return {
            'count': count, 'miss': miss, 'status': status,
            'xiang_sui': xiang_sui, 'gen_sui': gen_sui,
            'xiang_sui_prob': xiang_sui_prob,
            'periods': cycle_periods, 'total_periods': len(cycle_periods),
            'period_num_map': period_num_map
        }
    except Exception as e:
        st.error(f"周期分析出错: {str(e)}")
        return None

@st.cache_data
def calc_single_period_miss(df, target_period):
    periods = sorted(df['期号'].tolist())
    target_idx = periods.index(str(target_period))
    miss_dict = {}
    for num in range(1,81):
        last_seen = -1
        for i in range(target_idx-1, -1, -1):
            nums = get_period_numbers(df, periods[i])
            if num in nums:
                last_seen = target_idx - i - 1
                break
        miss_dict[num] = last_seen if last_seen != -1 else 999
    return miss_dict

# ==================== Tab 1-3 (保持不变，仅展示核心修复的Tab4-5) ====================
def render_tab1():
    st.header("📚 号码库管理")
    try:
        df = load_lottery_data()
        periods = sorted(df['期号'].tolist())

        # 选项卡：新增和修改
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("➕ 新增期号")
            next_period_val = f"2026{str(len(df)+1).zfill(3)}"
            if periods:
                try:
                    last_p = periods[-1]
                    year = str(last_p)[:4]
                    seq = int(str(last_p)[4:])
                    next_period_val = f"{year}{str(seq+1).zfill(3)}"
                except:
                    pass
            
            new_period = st.text_input("期号", value=next_period_val)
            new_num_input = st.text_area("开奖号码(20个，用逗号或空格分隔)", height=100)
            
            if st.button("✅ 保存新期号", type="primary", use_container_width=True):
                with st.spinner("保存中..."):
                    new_period_str = str(new_period)
                    if new_period_str in df['期号'].values:
                        st.error(f"❌ 期号{new_period_str}已存在")
                    else:
                        nums = parse_numbers(new_num_input)
                        if len(nums)!=20:
                            st.error(f"❌ 需20个号码，当前解析出{len(nums)}个")
                        else:
                            df.loc[len(df)] = [new_period_str] + nums
                            save_lottery_data(df)
                            st.success("✅ 保存成功！")
                            st.rerun()
        
        with col2:
            st.subheader("✏️ 修改期号")
            if periods:
                selected_period = st.selectbox("选择要修改的期号", periods)
                if selected_period:
                    # 获取当前期号的数据
                    current_row = df[df['期号'] == selected_period]
                    current_nums = [current_row[f'N{i}'].values[0] for i in range(1, 21)]
                    current_nums_str = ", ".join(map(str, current_nums))
                    
                    # 显示当前数据
                    st.write(f"**当前期号:** {selected_period}")
                    st.write(f"**当前号码:** {current_nums_str}")
                    
                    # 输入新数据
                    new_num_input = st.text_area("新开奖号码", value=current_nums_str, height=100)
                    
                    if st.button("💾 保存修改", type="primary", use_container_width=True):
                        with st.spinner("修改中..."):
                            nums = parse_numbers(new_num_input)
                            if len(nums)!=20:
                                st.error(f"❌ 需20个号码，当前解析出{len(nums)}个")
                            else:
                                # 更新数据
                                df.loc[df['期号'] == selected_period, [f'N{i}' for i in range(1, 21)]] = nums
                                save_lottery_data(df)
                                st.success("✅ 修改成功！")
                                st.rerun()
            else:
                st.info("ℹ️ 暂无数据可修改")

        # 数据展示和下载
        st.markdown("---")
        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            st.subheader("📋 最新10期数据")
        with col_btn2:
            st.download_button("📥 下载完整CSV", df.to_csv(index=False, encoding='utf-8-sig'),
                              "快八开奖号.csv", use_container_width=True)
        
        # 美化数据表格展示
        df_display = df.sort_values('期号', ascending=False).head(10).copy()
        # 将号码列合并显示
        def format_numbers(row):
            nums = [row[f'N{i}'] for i in range(1, 21)]
            return ', '.join([f'{int(n):02d}' for n in nums])
        
        df_display['开奖号码'] = df_display.apply(format_numbers, axis=1)
        df_display = df_display[['期号', '开奖号码']]
        
        # 样式化表格
        def highlight_period(row):
            return ['background-color: #e6f3ff'] * len(row)
        
        st.dataframe(
            df_display.style.set_properties(**{
                'font-size': '14px',
                'text-align': 'left'
            }).set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#4a90d9'), ('color', 'white'), ('font-weight', 'bold')]},
                {'selector': 'td', 'props': [('padding', '8px')]}
            ]),
            use_container_width=True
        )
    except Exception as e:
        st.error(f"页面出错: {str(e)}")

def get_numbers_list(df, period):
    try:
        row = df[df['期号'] == str(period)]
        if len(row) == 0:
            return []
        return [int(row[f'N{i}'].values[0]) for i in range(1,21)]
    except:
        return []

def render_tab2():
    st.header("📊 周期分析")
    df = load_lottery_data()
    if len(df) < 10:
        st.warning("⚠️ 数据量不足，至少需要10期数据")
        return

    periods = sorted(df['期号'].tolist())
    
    # 1. 周期选择器 - 放到主界面
    col_period, col_cycles = st.columns([1, 2])
    with col_period:
        target_period = st.selectbox("📌 分析目标期", periods, index=len(periods)-1)
    with col_cycles:
        cycles = st.multiselect("📊 统计周期(可多选)", [10, 20, 50, 80, 100, 150], default=[10, 20, 50, 100])
    
    # 获取目标期之前的数据索引
    target_idx = periods.index(target_period)
    
    # 2. 核心统计函数 - 不同周期使用不同的冷热温标准
    def analyze_cycle(cycle_len):
        if target_idx < cycle_len:
            return None
        # 截取数据
        cycle_periods = periods[target_idx-cycle_len : target_idx]
        cycle_data = df[df['期号'].isin(cycle_periods)]
        
        # 统计每个号码出现次数
        all_nums = []
        for _, row in cycle_data.iterrows():
            all_nums.extend([row[f'N{i}'] for i in range(1,21)])
        count = Counter(all_nums)
        
        # 计算遗漏 (最后一次出现距离目标期的期数)
        miss = {}
        for num in range(1, 81):
            last_seen = -1
            for i, p in enumerate(reversed(cycle_periods)):
                nums = get_numbers_list(df, p)
                if num in nums:
                    last_seen = i
                    break
            miss[num] = last_seen if last_seen != -1 else cycle_len
        
        # 根据不同周期设置不同的冷热温判断标准
        # 理论基础：每期开20个号码，80个号码中选
        # 100期理论平均 = 25次，80期=20次，50期=12.5次，20期=5次
        # 使用基于理论平均值的动态阈值
        
        if cycle_len == 100:
            # 100期标准：热号≥30次(1.2倍均值)，冷号≤15次(0.6倍均值)
            cold_thresh = 15
            warm_low, warm_high = 16, 29
            hot_thresh = 30
        elif cycle_len == 80:
            # 80期标准：热号≥24次，冷号≤12次
            cold_thresh = 12
            warm_low, warm_high = 13, 23
            hot_thresh = 24
        elif cycle_len == 50:
            # 50期标准：热号≥15次，冷号≤7次
            cold_thresh = 7
            warm_low, warm_high = 8, 14
            hot_thresh = 15
        elif cycle_len == 20:
            # 20期标准：热号≥6次，冷号≤3次
            cold_thresh = 3
            warm_low, warm_high = 4, 5
            hot_thresh = 6
        elif cycle_len >= 150:
            # 150期标准：热号≥45次，冷号≤22次
            cold_thresh = 22
            warm_low, warm_high = 23, 44
            hot_thresh = 45
        else:
            # 默认标准（10期等）：热号≥3次，冷号≤1次
            cold_thresh = 1
            warm_low, warm_high = 2, 2
            hot_thresh = 3
        
        # 冷热温定义
        status = {}
        for num in range(1, 81):
            c = count.get(num, 0)
            m = miss[num]
            # 综合考虑出现次数和遗漏值
            if c <= cold_thresh or m >= cycle_len * 0.8:
                status[num] = "冷"
            elif c >= hot_thresh or (warm_low <= c <= warm_high):
                status[num] = "热"
            else:
                status[num] = "温"
        
        return { 
            'count': count, 'miss': miss, 'status': status, 
            'periods': cycle_periods, 'data': cycle_data,
            'thresholds': {'cold': cold_thresh, 'warm': (warm_low, warm_high), 'hot': hot_thresh}
        }

    # 执行分析
    results = {}
    for c in cycles:
        res = analyze_cycle(c)
        if res:
            results[c] = res

    if not results:
        st.error("❌ 所选周期数据不足")
        return

    # 展示各周期参数标准
    st.markdown("### 📊 各周期冷热温参数标准")
    threshold_data = []
    for c in sorted(cycles):
        if c in results:
            thresh = results[c]['thresholds']
            threshold_data.append({
                '周期': f'{c}期',
                '🔥 热号标准': f'≥{thresh["hot"]}次',
                '🌡️ 温号标准': f'{thresh["warm"][0]}-{thresh["warm"][1]}次',
                '❄️ 冷号标准': f'≤{thresh["cold"]}次'
            })
    df_thresh = pd.DataFrame(threshold_data)
    
    # 美化参数表格
    st.dataframe(
        df_thresh.style.set_properties(**{
            'text-align': 'center'
        }).set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#4a90d9'), ('color', 'white'), ('font-weight', 'bold'), ('text-align', 'center')]},
            {'selector': 'td', 'props': [('padding', '10px')]}
        ]),
        use_container_width=True
    )
    
    # 理论说明
    with st.expander("💡 查看理论基础", expanded=False):
        st.write("快乐8每期从80个号码中开出20个号码。不同周期的理论平均值不同：")
        st.write("- **100期平均**：25次/号码")
        st.write("- **80期平均**：20次/号码")
        st.write("- **50期平均**：12.5次/号码")
        st.write("- **20期平均**：5次/号码")
        st.success("系统根据各周期的理论均值动态调整冷热温阈值")

    # 3. 展示统计总表
    st.markdown("### 📈 号码综合统计表")
    stats_list = []
    for num in range(1, 81):
        row = {'号码': f'{num:02d}'}
        for c in cycles:
            if c in results:
                row[f'{c}期_出现'] = results[c]['count'].get(num, 0)
                row[f'{c}期_遗漏'] = results[c]['miss'][num]
                row[f'{c}期_状态'] = results[c]['status'][num]
        stats_list.append(row)
    df_stats = pd.DataFrame(stats_list)
    
    # 样式高亮：热号红，冷号蓝
    def highlight_status(val):
        if val == '热':
            return 'background-color: #ffcccc; color: red; font-weight: bold'
        elif val == '冷':
            return 'background-color: #cce5ff; color: blue; font-weight: bold'
        elif val == '温':
            return 'background-color: #fff3cd; color: #856404'
        return ''
    
    # 兼容新旧版本pandas的样式函数
    status_cols = [col for col in df_stats.columns if '状态' in col]
    try:
        styled_df = df_stats.style.map(highlight_status, subset=status_cols)
    except AttributeError:
        try:
            styled_df = df_stats.style.applymap(highlight_status, subset=status_cols)
        except:
            styled_df = df_stats
    
    # 美化表格整体样式
    st.dataframe(
        styled_df.set_properties(**{
            'font-size': '12px',
            'text-align': 'center'
        }).set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#4a90d9'), ('color', 'white'), ('font-weight', 'bold'), ('text-align', 'center')]},
            {'selector': 'td', 'props': [('padding', '6px')]}
        ]),
        use_container_width=True,
        height=500
    )

    # 4. 相随号与跟随号分析 - 按周期分别分析
    st.markdown("---")
    st.subheader("🔗 相随号 & 跟随号深度分析")
    
    # 为每个周期计算相随号和跟随号
    xiang_sui_by_cycle = {}
    gen_sui_by_cycle = {}
    
    for c in sorted(cycles):
        if c in results:
            res = results[c]
            period_list = res['periods']
            
            xiang_sui = defaultdict(list)
            gen_sui = defaultdict(list)
            
            for i in range(len(period_list)-1):
                p_n = period_list[i]
                p_n1 = period_list[i+1]
                nums_n = get_numbers_list(df, p_n)
                nums_n1 = get_numbers_list(df, p_n1)
                
                # 相随号
                for a in nums_n:
                    xiang_sui[a].extend(nums_n1)
                
                # 跟随号
                for b in nums_n1:
                    others = [x for x in nums_n1 if x != b]
                    gen_sui[b].extend(others)
            
            xiang_sui_by_cycle[c] = xiang_sui
            gen_sui_by_cycle[c] = gen_sui
    
    # 选择分析周期
    selected_cycle = st.selectbox("选择分析周期", sorted(cycles), index=len(cycles)-1)
    
    if selected_cycle in xiang_sui_by_cycle:
        xiang_sui = xiang_sui_by_cycle[selected_cycle]
        gen_sui = gen_sui_by_cycle[selected_cycle]
        
        # 根据周期设置相随号和跟随号的阈值
        if selected_cycle == 100:
            # 100期：相随号≥8次，跟随号≥15次
            xiang_sui_thresh = 8
            gen_sui_thresh = 15
        elif selected_cycle == 80:
            # 80期：相随号≥6次，跟随号≥12次
            xiang_sui_thresh = 6
            gen_sui_thresh = 12
        elif selected_cycle == 50:
            # 50期：相随号≥4次，跟随号≥8次
            xiang_sui_thresh = 4
            gen_sui_thresh = 8
        elif selected_cycle == 20:
            # 20期：相随号≥2次，跟随号≥4次
            xiang_sui_thresh = 2
            gen_sui_thresh = 4
        else:
            # 默认：相随号≥1次，跟随号≥2次
            xiang_sui_thresh = 1
            gen_sui_thresh = 2
        
        # 展示阈值说明
        st.info(f"💡 {selected_cycle}期相随号/跟随号判定标准：\n- 强相随号：≥{xiang_sui_thresh}次\n- 强跟随号：≥{gen_sui_thresh}次")
        
        # 展示Top相随号
        c1, c2 = st.columns(2)
        with c1:
            sel_num_xs = st.selectbox("选择号码查看其相随号", range(1,81), index=0)
            xs_count = Counter(xiang_sui.get(sel_num_xs, []))
            st.write(f"**{sel_num_xs} 的 Top 10 相随号 (出现次数):**")
            df_xs = pd.DataFrame(xs_count.most_common(10), columns=['号码', '次数'])
            
            # 高亮显示强相随号
            def highlight_strong_xs(val):
                return 'background-color: #ffcccc; color: red' if val >= xiang_sui_thresh else ''
            
            try:
                st.bar_chart(df_xs.set_index('号码'))
                # 展示详细数据
                st.write("**详细数据:**")
                styled_xs = df_xs.style.applymap(highlight_strong_xs, subset=['次数'])
                st.dataframe(styled_xs, use_container_width=True)
            except:
                st.bar_chart(df_xs.set_index('号码'))
                st.dataframe(df_xs, use_container_width=True)

        with c2:
            sel_num_gs = st.selectbox("选择号码查看其跟随号", range(1,81), index=1)
            gs_count = Counter(gen_sui.get(sel_num_gs, []))
            st.write(f"**{sel_num_gs} 的 Top 10 跟随号 (出现次数):**")
            df_gs = pd.DataFrame(gs_count.most_common(10), columns=['号码', '次数'])
            
            # 高亮显示强跟随号
            def highlight_strong_gs(val):
                return 'background-color: #ffcccc; color: red' if val >= gen_sui_thresh else ''
            
            try:
                st.bar_chart(df_gs.set_index('号码'))
                # 展示详细数据
                st.write("**详细数据:**")
                styled_gs = df_gs.style.applymap(highlight_strong_gs, subset=['次数'])
                st.dataframe(styled_gs, use_container_width=True)
            except:
                st.bar_chart(df_gs.set_index('号码'))
                st.dataframe(df_gs, use_container_width=True)

    # 5. 双码/三码同频
    st.markdown("---")
    st.subheader("💎 双码 & 三码 同频统计")
    
    # 计算双码
    shuangma = defaultdict(int)
    for p in period_list:
        nums = get_numbers_list(df, p)
        for pair in itertools.combinations(sorted(nums), 2):
            shuangma[pair] += 1
    
    top_shuangma = sorted(shuangma.items(), key=lambda x: x[1], reverse=True)[:10]
    st.write("**Top 10 双码同频:**")
    for (a,b), cnt in top_shuangma:
        st.text(f"{a:02d} & {b:02d} : 共同出现 {cnt} 次")

    # 保存中间结果到session_state供后续Tab使用
    # 重新定义max_cycle和res_max
    max_cycle = max([c for c in results.keys()])
    res_max = results[max_cycle]
    # 保存所有周期的相随号和跟随号数据
    st.session_state['analysis_results'] = {
        'max_cycle': max_cycle,
        'res_max': res_max,
        'xiang_sui': xiang_sui_by_cycle,  # 保存所有周期的相随号
        'gen_sui': gen_sui_by_cycle,      # 保存所有周期的跟随号
        'shuangma': shuangma,
        'results': results,
        'selected_cycle': selected_cycle  # 保存当前选择的周期
    }
    
    # 6. 不同周期比例分析
    st.markdown("---")
    st.subheader("📊 不同周期参考数值比例分析")
    
    ratio_data = []
    for c in cycles:
        if c in results:
            status_count = Counter(results[c]['status'].values())
            total = sum(status_count.values())
            hot_ratio = status_count.get('热', 0) / total * 100
            warm_ratio = status_count.get('温', 0) / total * 100
            cold_ratio = status_count.get('冷', 0) / total * 100
            
            ratio_data.append({
                '周期': f'{c}期',
                '热号比例': hot_ratio,
                '温号比例': warm_ratio,
                '冷号比例': cold_ratio
            })
    
    df_ratio = pd.DataFrame(ratio_data)
    st.dataframe(df_ratio, use_container_width=True)
    
    # 可视化不同周期的比例
    st.subheader("📈 不同周期冷热号比例对比")
    fig = px.bar(df_ratio, x='周期', y=['热号比例', '温号比例', '冷号比例'], 
                 barmode='stack', title='不同周期冷热号比例')
    st.plotly_chart(fig, use_container_width=True)

def load_data():
    return load_lottery_data()

def calculate_metrics(numbers):
    if not numbers or len(numbers) != 20:
        return {'奇偶比': '-', '大小比': '-', '质合比': '-', '012路': '-', '和值': 0, '跨度': 0}
    odd = len([n for n in numbers if n % 2 == 1])
    small = len([n for n in numbers if n <= 40])
    prime = len([n for n in numbers if n in PRIME_SET])
    lu0 = len([n for n in numbers if n % 3 == 0])
    lu1 = len([n for n in numbers if n % 3 == 1])
    return {
        '奇偶比': f"{odd}:{20-odd}", 
        '大小比': f"{small}:{20-small}",
        '质合比': f"{prime}:{20-prime}", 
        '012路': f"{lu0}:{lu1}:{20-lu0-lu1}",
        '和值': sum(numbers), 
        '跨度': max(numbers)-min(numbers)
    }

def render_tab3():
    st.header("🔍 深度复盘")
    df = load_data()
    periods = sorted(df['期号'].tolist())
    
    if len(periods) < 2:
        st.warning("⚠️ 至少需要两期数据进行复盘")
        return
    
    # 选择期号 - 使用更清晰的布局
    col_sel1, col_sel2, col_sel3 = st.columns([1, 1, 2])
    with col_sel1:
        n_period = st.selectbox("📌 复盘目标期", periods, index=len(periods)-1)
    with col_sel2:
        n_idx = periods.index(n_period)
        n1_period = periods[n_idx-1] if n_idx > 0 else None
        st.write("**自动配对 N-1 期:**")
        st.success(f"{n1_period}")
    with col_sel3:
        st.write("**开奖号码:**")
        nums_n = get_numbers_list(df, n_period)
        nums_n1 = get_numbers_list(df, n1_period) if n1_period else []
        st.write(f"**{n_period}期:** {sorted(nums_n)}")
    
    if not n1_period:
        st.error("❌ 所选期号无前驱数据")
        return
    
    # 计算metrics
    metrics = calculate_metrics(nums_n)
    
    # ================== 模块一：单期深度解析 (N期) ==================
    st.markdown("---")
    st.markdown(f"### 📌 模块一：{n_period} 期 单期深度解析")
    
    # 使用卡片式布局展示主要数据
    col_card1, col_card2, col_card3 = st.columns(3)
    with col_card1:
        st.metric("奇偶比", metrics['奇偶比'])
    with col_card2:
        st.metric("大小比", metrics['大小比'])
    with col_card3:
        st.metric("质合比", metrics['质合比'])
    
    # 1. 区间分布
    with st.expander("📊 1️⃣ 区间分布 (8分区 & 4分区)", expanded=True):
        # 8分区表格
        bins_8 = [0,10,20,30,40,50,60,70,80]
        labels_8 = ["1区(01-10)","2区(11-20)","3区(21-30)","4区(31-40)",
                    "5区(41-50)","6区(51-60)","7区(61-70)","8区(71-80)"]
        
        # 计算每个区间的出号
        interval_data = []
        for i in range(8):
            start = bins_8[i] + 1
            end = bins_8[i+1]
            interval_nums = [n for n in nums_n if start <= n <= end]
            count = len(interval_nums)
            
            # 分析走势特征
            if count >= 4:
                trend = "🔥 核心出号"
            elif count == 0:
                trend = "❌ 区间断档"
            elif count <= 1:
                trend = "🔵 区间偏冷"
            else:
                trend = "✅ 平稳出号"
            
            interval_data.append({
                '区间': labels_8[i],
                '范围': f"{start:02d}-{end:02d}",
                '出号': ', '.join([f"{n:02d}" for n in interval_nums]) if interval_nums else "—",
                '个数': count,
                '特征': trend
            })
        
        df_interval = pd.DataFrame(interval_data)
        
        # 美化表格
        st.dataframe(
            df_interval.style.set_properties(**{
                'text-align': 'center'
            }).set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#4a90d9'), ('color', 'white'), ('font-weight', 'bold')]},
                {'selector': 'td', 'props': [('padding', '8px')]}
            ]),
            use_container_width=True
        )
        
        # 4分区
        bins_4 = [0,20,40,60,80]
        interval_4 = []
        labels_4 = ["1区(01-20)", "2区(21-40)", "3区(41-60)", "4区(61-80)"]
        for i in range(4):
            start = bins_4[i] + 1
            end = bins_4[i+1]
            count = len([n for n in nums_n if start <= n <= end])
            interval_4.append((labels_4[i], count))
        
        # 4分区可视化
        col_4_1, col_4_2 = st.columns([1, 1])
        with col_4_1:
            st.write("**4分区比例:**")
            for label, count in interval_4:
                percentage = count / 20 * 100
                st.progress(percentage/100, text=f"{label}: {count}个 ({percentage:.0f}%)")
        with col_4_2:
            st.write("**区间分析:**")
            counts = [count for _, count in interval_4]
            if max(counts) - min(counts) <= 3:
                st.success("✅ 各区相对均衡")
            elif sum(counts[:2]) > sum(counts[2:]):
                st.info("📈 前半区占优")
            else:
                st.info("📈 后半区占优")
    
    # 2. 基础属性
    with st.expander("2️⃣ 奇偶/大小/质合 & 012路", expanded=True):
        metrics = calculate_metrics(nums_n)
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("奇偶比", metrics['奇偶比'])
        col_m2.metric("大小比", metrics['大小比'])
        col_m3.metric("质合比", metrics['质合比'])
        st.info(f"012路分布: {metrics['012路']} | 和值: {metrics['和值']} | 跨度: {metrics['跨度']}")
        
        # 详细分析
        odd_nums = [n for n in nums_n if n % 2 == 1]
        even_nums = [n for n in nums_n if n % 2 == 0]
        small_nums = [n for n in nums_n if n <= 40]
        big_nums = [n for n in nums_n if n > 40]
        prime_nums = [n for n in nums_n if n in PRIME_SET]
        composite_nums = [n for n in nums_n if n not in PRIME_SET]
        
        st.write("**详细分析:**")
        st.write(f"- 奇数: {len(odd_nums)}个 ({', '.join([f'{n:02d}' for n in sorted(odd_nums)])})")
        st.write(f"- 偶数: {len(even_nums)}个 ({', '.join([f'{n:02d}' for n in sorted(even_nums)])})")
        st.write(f"- 小数(1-40): {len(small_nums)}个")
        st.write(f"- 大数(41-80): {len(big_nums)}个")
        st.write(f"- 质数: {len(prime_nums)}个 ({', '.join([f'{n:02d}' for n in sorted(prime_nums)])})")
        st.write(f"- 合数: {len(composite_nums)}个")
        
        # 012路详细分析
        lu0 = [n for n in nums_n if n % 3 == 0]
        lu1 = [n for n in nums_n if n % 3 == 1]
        lu2 = [n for n in nums_n if n % 3 == 2]
        st.write(f"- 0路(除3余0): {len(lu0)}个 ({', '.join([f'{n:02d}' for n in sorted(lu0)])})")
        st.write(f"- 1路(除3余1): {len(lu1)}个 ({', '.join([f'{n:02d}' for n in sorted(lu1)])})")
        st.write(f"- 2路(除3余2): {len(lu2)}个 ({', '.join([f'{n:02d}' for n in sorted(lu2)])})")
    
    # 3. 同尾号 & 连号
    with st.expander("3️⃣ 同尾号 & 连号特征", expanded=True):
        # 同尾
        tails = defaultdict(list)
        for n in nums_n:
            tails[n%10].append(n)
        st.write("**同尾号分布:**")
        
        tail_data = []
        for t, ns in sorted(tails.items()):
            count = len(ns)
            tail_data.append({
                '尾数': t,
                '本期开出号码': ', '.join([f"{n:02d}" for n in sorted(ns)]),
                '出号个数': count
            })
        
        df_tail = pd.DataFrame(tail_data)
        st.dataframe(df_tail, use_container_width=True)
        
        # 分析同尾号特征
        max_tail_count = max([len(ns) for ns in tails.values()])
        if max_tail_count >= 4:
            st.warning("⚠️ 同尾号极端集中爆发")
        elif max_tail_count >= 3:
            st.info("同尾号集中趋势明显")
        
        # 连号
        st.write("**连号组合:**")
        nums_sorted = sorted(nums_n)
        lians = []
        current = [nums_sorted[0]]
        for n in nums_sorted[1:]:
            if n == current[-1] + 1:
                current.append(n)
            else:
                if len(current)>=2:
                    lians.append(current)
                current = [n]
        if len(current)>=2:
            lians.append(current)
        
        if lians:
            for l in lians:
                st.markdown(f"- 🔗 **{len(l)}连号**: {'-'.join([f'{x:02d}' for x in l])}")
            st.write(f"总计 {len(lians)} 组连号，覆盖 {sum(len(l) for l in lians)} 个号码")
        else:
            st.text("无明显连号")
    
    # 4. 冷热遗漏分析
    with st.expander("4️⃣ 冷热遗漏与历史关联分析", expanded=True):
        # 计算遗漏值
        def calculate_omission(num, history_nums):
            for i, nums in enumerate(history_nums):
                if num in nums:
                    return i
            return len(history_nums)
        
        # 获取历史数据
        history_nums = []
        for i in range(max(0, n_idx-10), n_idx):
            history_nums.append(get_numbers_list(df, periods[i]))
        
        # 计算每个号码的遗漏值
        omission_data = []
        hot_count = 0
        warm_count = 0
        cold_count = 0
        
        for num in nums_n:
            omis = calculate_omission(num, history_nums)
            if omis <= 4:
                status = "热号"
                hot_count += 1
            elif 5 <= omis <= 9:
                status = "温码"
                warm_count += 1
            else:
                status = "冷号"
                cold_count += 1
            omission_data.append({
                '号码': num,
                '遗漏期数': omis,
                '状态': status
            })
        
        df_omission = pd.DataFrame(omission_data)
        st.write("**冷热遗漏分析:**")
        
        # 高亮显示冷热号
        def highlight_status(val):
            if val == '热号':
                return 'background-color: #ffcccc; color: red; font-weight: bold'
            elif val == '冷号':
                return 'background-color: #ccccff; color: blue; font-weight: bold'
            return ''
        
        try:
            st.dataframe(df_omission.style.applymap(highlight_status, subset=['状态']), use_container_width=True)
        except:
            # 兼容旧版pandas
            st.dataframe(df_omission, use_container_width=True)
        
        # 分析冷热分布
        total_omission = sum(df_omission['遗漏期数'])
        st.write(f"- 遗漏总值: {total_omission}")
        st.write(f"- 冷热温码比: {cold_count}:{warm_count}:{hot_count}")
        st.write(f"- 热号占比: {hot_count/20:.1%}")
        st.write(f"- 温码占比: {warm_count/20:.1%}")
        st.write(f"- 冷号占比: {cold_count/20:.1%}")
        
        if hot_count >= 10:
            st.info("热号主导型走势")
        elif cold_count >= 8:
            st.info("冷号回补型走势")
        else:
            st.info("均衡型走势")
    
    # ================== 模块二：双期对比 (N vs N-1) ==================
    st.markdown("---")
    st.subheader(f"📊 模块二：{n1_period} vs {n_period} 双期对比")
    
    # 1. 核心指标对比
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"{n1_period} 期核心数据")
        metrics_n1 = calculate_metrics(nums_n1)
        st.metric("奇偶比", metrics_n1['奇偶比'])
        st.metric("大小比", metrics_n1['大小比'])
        st.metric("质合比", metrics_n1['质合比'])
        st.info(f"012路: {metrics_n1['012路']}")
        st.info(f"和值: {metrics_n1['和值']} | 跨度: {metrics_n1['跨度']}")
    
    with col2:
        st.subheader(f"{n_period} 期核心数据")
        metrics_n = calculate_metrics(nums_n)
        st.metric("奇偶比", metrics_n['奇偶比'])
        st.metric("大小比", metrics_n['大小比'])
        st.metric("质合比", metrics_n['质合比'])
        st.info(f"012路: {metrics_n['012路']}")
        st.info(f"和值: {metrics_n['和值']} | 跨度: {metrics_n['跨度']}")
    
    # 2. 重号分析
    chonghao = set(nums_n) & set(nums_n1)
    st.markdown("---")
    st.subheader("🔗 重号与关联分析")
    st.metric(f"重号数量", f"{len(chonghao)} 个", f"重合率 {len(chonghao)/20:.0%}")
    if chonghao:
        st.write(f"重号列表: {', '.join([f'{x:02d}' for x in sorted(chonghao)])}")
        
        # 分析重号率
        if len(chonghao) >= 8:
            st.warning("重号率远超历史均值（20%-25%），热号延续性达历史高位")
        elif len(chonghao) >= 6:
            st.info("重号率高于历史均值，热号延续性强")
        elif len(chonghao) <= 3:
            st.info("重号率低于历史均值，号码更换较大")
    
    # 3. 集中形态分析
    st.markdown("---")
    st.subheader("📈 集中形态对比")
    
    # 同尾分析
    def analyze_tail_pattern(nums):
        tails = defaultdict(list)
        for n in nums:
            tails[n%10].append(n)
        max_tail_count = max([len(ns) for ns in tails.values()])
        tail_counts = [len(ns) for ns in tails.values()]
        high_tail_count = sum(1 for c in tail_counts if c >= 3)
        return max_tail_count, high_tail_count
    
    # 连号分析
    def analyze_lian_pattern(nums):
        nums_sorted = sorted(nums)
        lians = []
        current = [nums_sorted[0]]
        for n in nums_sorted[1:]:
            if n == current[-1] + 1:
                current.append(n)
            else:
                if len(current)>=2:
                    lians.append(current)
                current = [n]
        if len(current)>=2:
            lians.append(current)
        if lians:
            max_lian_length = max(len(l) for l in lians)
            total_lian_nums = sum(len(l) for l in lians)
            return max_lian_length, total_lian_nums
        else:
            return 0, 0
    
    # 分析两期的集中形态
    max_tail_n1, high_tail_n1 = analyze_tail_pattern(nums_n1)
    max_tail_n, high_tail_n = analyze_tail_pattern(nums_n)
    max_lian_n1, total_lian_n1 = analyze_lian_pattern(nums_n1)
    max_lian_n, total_lian_n = analyze_lian_pattern(nums_n)
    
    col_tail, col_lian = st.columns(2)
    with col_tail:
        st.write("**同尾分布特征对比**")
        st.write(f"{n1_period}期: 最高{max_tail_n1}同尾，{high_tail_n1}组3个以上同尾")
        st.write(f"{n_period}期: 最高{max_tail_n}同尾，{high_tail_n}组3个以上同尾")
        if max_tail_n >= 4 or max_tail_n1 >= 4:
            st.warning("⚠️ 同尾极端集中")
    
    with col_lian:
        st.write("**连号组合特征对比**")
        st.write(f"{n1_period}期: 最高{max_lian_n1}连号，覆盖{total_lian_n1}个号码")
        st.write(f"{n_period}期: 最高{max_lian_n}连号，覆盖{total_lian_n}个号码")
        if max_lian_n >= 5 or max_lian_n1 >= 5:
            st.warning("⚠️ 连号极端集中")
    
    # 4. 冷热遗漏对比
    st.markdown("---")
    st.subheader("❄️ 冷热遗漏对比")
    
    def analyze_cold_hot(nums, history_nums):
        def calculate_omission(num, history):
            for i, h_nums in enumerate(history):
                if num in h_nums:
                    return i
            return len(history)
        
        hot_count = 0
        warm_count = 0
        cold_count = 0
        total_omission = 0
        
        for num in nums:
            omis = calculate_omission(num, history_nums)
            total_omission += omis
            if omis <= 4:
                hot_count += 1
            elif 5 <= omis <= 9:
                warm_count += 1
            else:
                cold_count += 1
        
        return hot_count, warm_count, cold_count, total_omission
    
    # 获取历史数据
    history_nums = []
    for i in range(max(0, n_idx-10), n_idx):
        history_nums.append(get_numbers_list(df, periods[i]))
    
    hot_n1, warm_n1, cold_n1, omis_n1 = analyze_cold_hot(nums_n1, history_nums[:-1])
    hot_n, warm_n, cold_n, omis_n = analyze_cold_hot(nums_n, history_nums)
    
    col_hot1, col_hot2 = st.columns(2)
    with col_hot1:
        st.write(f"**{n1_period}期 冷热分布**")
        st.write(f"热号(≤4期): {hot_n1}个 ({hot_n1/20:.1%})")
        st.write(f"温码(5-9期): {warm_n1}个 ({warm_n1/20:.1%})")
        st.write(f"冷号(≥10期): {cold_n1}个 ({cold_n1/20:.1%})")
        st.write(f"遗漏总值: {omis_n1}")
    
    with col_hot2:
        st.write(f"**{n_period}期 冷热分布**")
        st.write(f"热号(≤4期): {hot_n}个 ({hot_n/20:.1%})")
        st.write(f"温码(5-9期): {warm_n}个 ({warm_n/20:.1%})")
        st.write(f"冷号(≥10期): {cold_n}个 ({cold_n/20:.1%})")
        st.write(f"遗漏总值: {omis_n}")
    
    # 5. 稳定共性规律
    st.markdown("---")
    st.subheader("🔍 稳定共性规律")
    
    st.write("**经两期数据验证的稳定共性规律：**")
    if hot_n >= 10 and hot_n1 >= 10:
        st.info("1. 热号绝对主导：两期热号占比均超50%，冷号占比不足15%，'追热不追冷'得到验证")
    
    # 质合数分析
    prime_n1 = len([n for n in nums_n1 if n in PRIME_SET])
    prime_n = len([n for n in nums_n if n in PRIME_SET])
    if prime_n <= 5 and prime_n1 <= 5:
        st.info("2. 合数持续强势：两期合数占比均超75%，远超历史均值，是出号绝对主力")
    
    # 集中化分析
    if (max_tail_n >= 3 or max_lian_n >= 3) and (max_tail_n1 >= 3 or max_lian_n1 >= 3):
        st.info("3. 集中化走势为主流：两期均出现集中形态，无一期实现全维度均衡")
    
    # 6. 核心趋势切换
    st.markdown("---")
    st.subheader("🔄 核心趋势切换")
    
    # 大小趋势分析
    small_n1 = len([n for n in nums_n1 if n <= 40])
    small_n = len([n for n in nums_n if n <= 40])
    if (small_n1 > 10 and small_n < 10) or (small_n1 < 10 and small_n > 10):
        st.info("1. 大小趋势轮动：从'小数占优'反转至'大数占优'或反之")
    
    # 集中形态切换
    if (max_tail_n1 >= 4 and max_lian_n < 3) or (max_lian_n1 >= 4 and max_tail_n < 3):
        st.info("2. 核心集中形态切换：从同尾极端集中切换为连号极端集中")
    
    # 重号率变化
    if len(chonghao) >= 6:
        st.info("3. 跨期关联逻辑强化：重号率高于历史均值，热号延续性强")
    
    # 7. 投注策略核心启示
    st.markdown("---")
    st.subheader("💡 投注策略核心启示")
    
    st.write("**选号优先级建议：**")
    st.info("1. 重号 > 相随号 > 热尾跟随号 > 冷号")
    
    st.write("**策略建议：**")
    st.info("2. 对同尾、连号两种核心集中形态双向防守")
    st.info("3. 放弃均衡化执念，顺应集中化与断档趋势")
    st.info("4. 热号追号需保持连续性，不可随意切换冷号")
    st.info("5. 极端形态是头奖轮空的核心诱因，需做好风险防控")
    
    # 走势复盘启示
    st.markdown("---")
    st.subheader("💡 走势复盘启示与策略总结")
    
    # 分析当前走势类型
    if hot_count >= 12:
        st.success("本期走势核心本质：热号主导型集中化走势")
        st.info("有效策略：热号追号 + 同尾跟进 + 连号防守 + 重号/相随号优先")
        st.warning("失效策略：冷号回补 + 尾数均衡分布 + 断尾防守")
    elif cold_count >= 8:
        st.success("本期走势核心本质：冷号回补型走势")
        st.info("有效策略：冷号回补 + 断尾防守 + 区间均衡分布")
        st.warning("失效策略：热号追号 + 同尾跟进")
    else:
        st.success("本期走势核心本质：均衡型走势")
        st.info("有效策略：区间均衡 + 尾数均衡 + 冷热搭配")
    
    st.info("核心规律启示：快乐8走势中，'集中化走势'的出现概率远高于'完美均衡化走势'，投注分析需优先关注近期热尾、热号的延续性，而非强行追求尾数、区间的理论均衡分布。")

# ==================== Tab 4: 三流派智能选号 (完全修复版) ====================
def render_tab4():
    st.header("🎯 智能选号")
    try:
        df = load_lottery_data()
        periods = sorted(df['期号'].tolist())
        if len(periods) < 10:
            st.warning("⚠️ 至少需要10期数据")
            return

        if 'analysis_results' not in st.session_state:
            st.warning("⚠️ 请先运行【Tab 2: 周期分析】生成分析数据")
            return
        
        analysis_data = st.session_state['analysis_results']
        results = analysis_data.get('results', {})
        
        # 期号选择 - 使用更清晰的布局
        col_sel1, col_sel2, col_sel3 = st.columns([1, 1, 2])
        with col_sel1:
            selected_period = st.selectbox("📌 分析基准期", periods, index=len(periods)-1)
        with col_sel2:
            # 计算下一期期号
            try:
                year_part = str(selected_period)[:4]
                seq_part = str(selected_period)[4:]
                next_seq = int(seq_part) + 1
                next_period = f"{year_part}{str(next_seq).zfill(3)}"
            except:
                next_period = f"2026{str(len(periods)+1).zfill(3)}"
            st.write("**🎯 预测目标期:**")
            st.success(f"{next_period}")
        
        with col_sel3:
            # 显示开奖号码
            selected_nums = get_numbers_list(df, selected_period)
            st.write(f"**📅 {selected_period}期号码:**")
            st.write(f"{sorted(selected_nums)}")

        # ==================================
        # N期开奖号码相随号、跟随号分析
        # ==================================
        st.markdown("---")
        st.subheader("📊 相随号/跟随号分析")
        
        if selected_nums:
            # 从analysis_data中获取相随号和跟随号数据
            xiang_sui_by_cycle = analysis_data.get('xiang_sui', {})
            gen_sui_by_cycle = analysis_data.get('gen_sui', {})
            
            # 获取用户选择的分析周期
            available_cycles = sorted(xiang_sui_by_cycle.keys()) if xiang_sui_by_cycle else []
            
            if not available_cycles:
                st.warning("⚠️ 未找到相随号/跟随号数据，请先在Tab 2中运行周期分析")
            else:
                # 默认使用最长的周期
                default_cycle_idx = len(available_cycles) - 1
                analysis_cycle = st.selectbox("选择分析周期", available_cycles, index=default_cycle_idx)
                
                # 获取对应周期的相随号和跟随号
                xiang_sui = xiang_sui_by_cycle.get(analysis_cycle, {})
                gen_sui = gen_sui_by_cycle.get(analysis_cycle, {})
                
                # 分析每个号码的相随号和跟随号
                all_related_nums = {}
                
                for num in selected_nums:
                    # 收集相随号
                    xs_nums = xiang_sui.get(num, [])
                    # 收集跟随号
                    gs_nums = gen_sui.get(num, [])
                    # 合并去重
                    related_nums = list(set(xs_nums + gs_nums))
                    # 统计出现次数
                    count = Counter(xs_nums + gs_nums)
                    # 按出现次数排序
                    sorted_related = sorted(count.items(), key=lambda x: x[1], reverse=True)
                    
                    all_related_nums[num] = sorted_related[:10]  # 取前10个
                
                # 展示结果
                st.info(f"📊 基于 {analysis_cycle} 期数据分析")
                for num in selected_nums:
                    with st.expander(f"号码 {num} 的相随号/跟随号分析"):
                        if all_related_nums.get(num):
                            # 提取数据
                            data = [(n, c) for n, c in all_related_nums[num]]
                            df_related = pd.DataFrame(data, columns=['关联号码', '出现次数'])
                            
                            # 高亮显示高频关联号码
                            def highlight_high_freq(val):
                                return 'background-color: #ffcccc; color: red' if val >= 3 else ''
                            
                            try:
                                styled_df = df_related.style.applymap(highlight_high_freq, subset=['出现次数'])
                                st.dataframe(styled_df, use_container_width=True)
                            except:
                                st.dataframe(df_related, use_container_width=True)
                        else:
                            st.write("暂无关联数据")
        else:
            st.warning(f"未找到 {selected_period} 期的开奖数据")

        # 检查是否已有存档
        df_predict_exist = load_predict_record(next_period)
        if not df_predict_exist.empty:
            st.info(f"📂 已找到 {next_period} 期的历史预测数据，正在加载...")
            row = df_predict_exist.iloc[0]
            final_25 = str2list(row['25码核心池'])
            final_hot = str2list(row['热号池'])
            final_cold = str2list(row['冷号池'])
            final_wen = str2list(row['稳胆池'])
            final_combo = {
                '11码': str2combo(row['11码组合']),
                '8码': str2combo(row['8码组合']),
                '6码': str2combo(row['6码组合']),
                '3码': str2combo(row['3码组合'])
            }
            
            # 直接展示存档数据
            st.markdown("---")
            st.subheader(f"✅ {next_period}期 25码核心选号池 (存档数据)")
            st.success(f"**{final_25}**")
            
            st.markdown("---")
            st.subheader(f"💎 {next_period}期 最终打号组合 (存档数据)")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write("**三组11码**")
                for i, z in enumerate(final_combo['11码'][:3]):
                    st.text(f"组{i+1}: {z}")
            with col2:
                st.write("**五组8码**")
                for i, z in enumerate(final_combo['8码']):
                    st.text(f"组{i+1}: {z}")
            with col3:
                st.write("**十组6码**")
                for i, z in enumerate(final_combo['6码']):
                    st.text(f"组{i+1}: {z}")
            with col4:
                st.write("**十组3码**")
                for i, z in enumerate(final_combo['3码']):
                    st.text(f"组{i+1}: {z}")
            return

        # ==================================
        # 一、简化版三流派筛选 (确保快速生成)
        # ==================================
        st.markdown("---")
        st.subheader("一、三流派号码筛选")
        
        # 简化筛选逻辑，确保有数据
        hot_pool = []
        cold_pool = []
        wen_pool = []
        
        if 20 in results:
            res_20 = results[20]
            # 热号：近20期出现≥4次
            hot_pool = sorted([n for n in range(1,81) if res_20['count'].get(n,0)>=4])[:15]
            # 冷号：遗漏≥10期
            cold_pool = sorted([n for n in range(1,81) if res_20['miss'][n]>=10])[:10]
            # 温号：中间状态
            wen_pool = sorted([n for n in range(1,81) if 5<=res_20['miss'][n]<=9])[:15]
        else:
            # 兜底
            hot_pool = list(range(1,26))
            cold_pool = list(range(26,51))
            wen_pool = list(range(51,81))
        
        # 高亮显示热号和冷号
        def highlight_numbers(nums, pool_type):
            highlighted = []
            for num in nums:
                if pool_type == 'hot':
                    highlighted.append(f"<span style='color: red; font-weight: bold'>{num}</span>")
                elif pool_type == 'cold':
                    highlighted.append(f"<span style='color: blue; font-weight: bold'>{num}</span>")
                else:
                    highlighted.append(str(num))
            return " ".join(highlighted)
        
        st.markdown(f"✅ 真热号池：{highlight_numbers(hot_pool[:10], 'hot')}", unsafe_allow_html=True)
        st.markdown(f"✅ 真冷号池：{highlight_numbers(cold_pool[:7], 'cold')}", unsafe_allow_html=True)
        st.markdown(f"✅ 稳胆温号池：{highlight_numbers(wen_pool[:8], 'warm')}", unsafe_allow_html=True)

        # ==================================
        # 二、25码核心选号池生成
        # ==================================
        st.markdown("---")
        st.subheader("二、25码核心选号池")
        
        if st.button("🎲 生成25码核心选号池", type="primary", use_container_width=True):
            with st.spinner("生成中..."):
                final_hot = hot_pool[:10]
                final_cold = [n for n in cold_pool if n not in final_hot][:7]
                final_wen = [n for n in wen_pool if n not in final_hot+final_cold][:8]
                
                final_25 = final_hot + final_cold + final_wen
                if len(final_25) <25:
                    supplement = [n for n in range(1,81) if n not in final_25][:25-len(final_25)]
                    final_25 += supplement
                final_25 = sorted(final_25[:25])
                
                # 保存到session
                st.session_state[f'{next_period}_25码选号池'] = final_25
                st.session_state[f'{next_period}_热号池'] = final_hot
                st.session_state[f'{next_period}_冷号池'] = final_cold
                st.session_state[f'{next_period}_稳胆池'] = final_wen
                st.session_state['current_predict_period'] = next_period
                
                st.success(f"**{final_25}**")

        # ==================================
        # 三、打号组合生成 (100%修复显示+存档)
        # ==================================
        st.markdown("---")
        st.subheader("三、打号组合生成")
        
        pool_key = f'{next_period}_25码选号池'
        if pool_key not in st.session_state:
            st.info("ℹ️ 请先点击【生成25码核心选号池】")
            return
        
        final_25 = st.session_state[pool_key]
        final_hot = st.session_state.get(f'{next_period}_热号池', [])
        final_cold = st.session_state.get(f'{next_period}_冷号池', [])
        final_wen = st.session_state.get(f'{next_period}_稳胆池', [])
        
        if st.button("📋 生成最终打号组合", type="primary", use_container_width=True):
            with st.spinner("正在生成组合..."):
                import random
                random.seed(42)
                
                # 【完全修复】组合生成函数
                def generate_combo(source, length, count):
                    combos = []
                    source_extend = source + [n for n in range(1,81) if n not in source]
                    attempts = 0
                    while len(combos) < count and attempts < 3000:
                        c = tuple(sorted(random.sample(source_extend, length)))
                        if c not in combos:
                            combos.append(c)
                        attempts +=1
                    # 兜底：如果还不够，生成简单组合
                    while len(combos) < count:
                        start = (len(combos) * length) % 80 + 1
                        end = start + length
                        if end > 80:
                            start = 1
                            end = start + length
                        c = tuple(range(start, end))
                        if c not in combos:
                            combos.append(c)
                    return combos
                
                # 生成各类型组合
                combo_11 = generate_combo(final_25, 11, 8)
                combo_8 = generate_combo(final_25, 8, 5)
                combo_6 = generate_combo(final_25, 6, 10)
                combo_3 = generate_combo(final_25, 3, 10)
                
                final_combo = {
                    '11码': combo_11,
                    '8码': combo_8,
                    '6码': combo_6,
                    '3码': combo_3
                }
                
                # 【完全修复】存档
                save_success = save_predict_record(
                    next_period,
                    final_hot,
                    final_cold,
                    final_wen,
                    final_25,
                    final_combo
                )
                
                if save_success:
                    st.success("✅ 预测数据已永久存档到本地！")
                
                # 【完全修复】显示组合
                st.markdown("---")
                st.subheader(f"💎 {next_period}期 最终打号组合")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write("**八组11码**")
                    for i, z in enumerate(final_combo['11码']):
                        st.text(f"组{i+1}: {z}")
                with col2:
                    st.write("**五组8码**")
                    for i, z in enumerate(final_combo['8码']):
                        st.text(f"组{i+1}: {z}")
                with col3:
                    st.write("**十组6码**")
                    for i, z in enumerate(final_combo['6码']):
                        st.text(f"组{i+1}: {z}")
                with col4:
                    st.write("**十组3码**")
                    for i, z in enumerate(final_combo['3码']):
                        st.text(f"组{i+1}: {z}")
                
                # 保存到session
                st.session_state[f'{next_period}_打号组合'] = final_combo

    except Exception as e:
        st.error(f"页面出错: {str(e)}")
        import traceback
        st.text(traceback.format_exc())

# ==================== Tab 5: 正确率验证 (永久存档版) ====================
def render_tab5():
    st.header("✅ 正确率验证")
    try:
        df = load_lottery_data()
        periods = sorted(df['期号'].tolist())
        if len(periods) < 1:
            st.warning("⚠️ 无开奖数据")
            return

        # 先检查是否有历史复盘存档
        df_verify_history = load_verify_record()
        if not df_verify_history.empty:
            with st.expander("📂 历史复盘存档", expanded=False):
                st.dataframe(df_verify_history.sort_values('期号', ascending=False), use_container_width=True)

        # 选择期号
        verify_period = st.selectbox("📌 选择已开奖的期号进行验证", periods, index=len(periods)-1)
        verify_period_str = str(verify_period)

        # 检查是否有该期的预测存档
        df_predict = load_predict_record(verify_period_str)
        if df_predict.empty:
            st.warning(f"⚠️ 未找到 {verify_period_str} 期的预测存档数据，请在开奖前于【Tab4智能选号】生成预测")
            return

        # 加载预测数据
        row = df_predict.iloc[0]
        core_pool = str2list(row['25码核心池'])
        combo_dict = {
            '11码': str2combo(row['11码组合']),
            '8码': str2combo(row['8码组合']),
            '6码': str2combo(row['6码组合']),
            '3码': str2combo(row['3码组合'])
        }

        # 获取开奖号码
        open_nums = set(get_period_numbers(df, verify_period_str))

        # 计算指标
        hit_pool = set(core_pool) & open_nums
        core_pool_hit = len(hit_pool)
        core_pool_hit_rate = core_pool_hit / 20 if core_pool else 0

        best_hit, best_combo, best_type = 0, None, ""
        combo_detail = {}
        for typ, z_list in combo_dict.items():
            for z in z_list:
                h = len(set(z) & open_nums)
                combo_detail[f"{typ}_{z}"] = h
                if h > best_hit:
                    best_hit, best_combo, best_type = h, z, typ

        # 展示结果
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("🎰 开奖号码")
            st.success(f"{sorted(open_nums)}")
        with col2:
            st.subheader("📂 筛选池命中")
            st.metric("命中个数", f"{core_pool_hit}/20", f"命中率 {core_pool_hit_rate:.0%}")
            # 高亮显示命中号码
            hit_str = "命中号码: "
            for num in sorted(core_pool):
                if num in hit_pool:
                    hit_str += f"<span style='color: red; font-weight: bold'>{num}</span> "
                else:
                    hit_str += f"{num} "
            st.markdown(hit_str, unsafe_allow_html=True)
        with col3:
            st.subheader("💎 最佳组合表现")
            st.metric(f"最佳表现({best_type})", f"中{best_hit}码")
            # 高亮显示最佳组合中的命中号码
            combo_str = "组合: "
            for num in best_combo:
                if num in open_nums:
                    combo_str += f"<span style='color: red; font-weight: bold'>{num}</span> "
                else:
                    combo_str += f"{num} "
            st.markdown(combo_str, unsafe_allow_html=True)

        # 自动存档复盘结果
        if st.button("💾 保存本次复盘结果", type="primary"):
            save_success = save_verify_record(
                verify_period_str,
                sorted(open_nums),
                core_pool_hit,
                core_pool_hit_rate,
                best_hit,
                best_combo,
                best_type,
                combo_detail
            )
            if save_success:
                st.success("✅ 复盘结果已永久存档！")
                st.rerun()

        # 全组合详细对比 - 四种组合并排显示
        st.markdown("---")
        st.markdown("### 📋 全组合详细命中对比")

        # 使用4列布局将四种组合并排显示
        col1, col2, col3, col4 = st.columns(4)
        
        combo_types = list(combo_dict.items())
        
        # 为每种组合类型创建表格
        for idx, (typ, z_list) in enumerate(combo_types):
            # 选择对应的列
            if idx == 0:
                col = col1
            elif idx == 1:
                col = col2
            elif idx == 2:
                col = col3
            else:
                col = col4
            
            with col:
                st.markdown(f"**📊 {typ}组合**")
                
                # 构建表格数据
                table_data = []
                for i, z in enumerate(z_list):
                    h = len(set(z) & open_nums)
                    # 标记命中的号码
                    hit_nums = []
                    miss_nums = []
                    for num in z:
                        if num in open_nums:
                            hit_nums.append(f"<span style='color: red; font-weight: bold'>{num:02d}</span>")
                        else:
                            miss_nums.append(f"{num:02d}")
                    
                    # 组合显示格式
                    all_nums_display = " ".join(hit_nums + miss_nums)
                    
                    table_data.append({
                        "编号": i + 1,
                        "号码": all_nums_display,
                        "重": h
                    })
                
                # 创建DataFrame并显示
                df_combo = pd.DataFrame(table_data)
                st.markdown(df_combo.to_html(escape=False, index=False), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"页面出错: {str(e)}")
        import traceback
        st.text(traceback.format_exc())

# ==================== 主程序入口 ====================
def main():
    try:
        init_base_data()
        
        # 顶部标题和简介
        st.title("🎰 快乐8智能分析系统 V4.0")
        st.markdown("---")
        
        # 上方标签框
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📚 号码库", 
            "📊 周期分析", 
            "🔍 深度复盘", 
            "🎯 智能选号", 
            "✅ 正确率验证"
        ])
        
        st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            font-size: 16px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 24px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.caption("💡 核心升级：预测永久存档 | 复盘永久存档 | 组合100%生成 | 多周期智能分析")

        with tab1:
            render_tab1()
        with tab2:
            render_tab2()
        with tab3:
            render_tab3()
        with tab4:
            render_tab4()
        with tab5:
            render_tab5()
    except Exception as e:
        st.error(f"系统严重错误: {str(e)}")
        import traceback
        st.text(traceback.format_exc())

if __name__ == "__main__":
    main()