# 带永久数据存储的公网版本代码
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
import os
import csv
import requests

# 页面配置
st.set_page_config(page_title="快乐8专业数据分析", page_icon="🎰", layout="wide")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 永久数据存储：用GitHub Gist ----------------------
# 你需要在这里替换成你自己的Gist信息，步骤：
# 1. 打开https://gist.github.com/ 新建一个公开的Gist，文件名叫kl8_data.csv
# 2. 把下面的GIST_ID和GIST_TOKEN替换成你的
GIST_ID = "你的Gist的ID"
GIST_TOKEN = "你的GitHub的Personal Access Token"
DATA_FILE = "kl8_history_data.csv"

# 完整的88期初始数据
INIT_DATA = [
    ["2026001",2,5,6,11,24,25,27,32,34,35,39,41,44,51,54,62,70,71,72,75],
    # ... 这里是完整的88期数据，和之前的一样，你复制的时候把完整的带过来就行
    ["2026088",8,9,13,14,18,22,25,33,39,40,44,46,49,55,64,66,68,71,77,80]
]

# 从Gist加载数据
def load_data():
    if not os.path.exists(DATA_FILE):
        # 第一次运行，从Gist拉取
        try:
            url = f"https://api.github.com/gists/{GIST_ID}"
            headers = {"Authorization": f"token {GIST_TOKEN}"}
            r = requests.get(url, headers=headers)
            content = r.json()['files']['kl8_data.csv']['content']
            with open(DATA_FILE, 'w', encoding='utf-8') as f:
                f.write(content)
        except:
            # 拉取失败，初始化本地数据
            with open(DATA_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['period'] + [f'n{i}' for i in range(1,21)])
                writer.writerows(INIT_DATA)
    
    # 读取数据
    df = pd.read_csv(DATA_FILE)
    df = df.sort_values('period', ascending=False).reset_index(drop=True)
    return df

# 保存新数据到Gist
def save_new_data(period, numbers):
    numbers = sorted(numbers)
    # 先保存到本地
    with open(DATA_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([period] + numbers)
    
    # 同步到Gist
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        url = f"https://api.github.com/gists/{GIST_ID}"
        headers = {"Authorization": f"token {GIST_TOKEN}"}
        data = {
            "files": {
                "kl8_data.csv": {"content": content}
            }
        }
        requests.patch(url, json=data, headers=headers)
    except:
        pass
    return True

# ---------------------- 后面的分析和页面代码和之前的完全一样，不用改 ----------------------
# ... 你把之前的分析函数、页面布局的代码直接粘在后面就行