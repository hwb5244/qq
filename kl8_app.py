
        
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

# 全局Session初始化
init_session_keys = [
    "raw_original_db", "26_year_db", "tongqi_db", "re_leng_data_db",
    "feature_columns", "feature_engineered_db", "model_dict", "predict_result",
    "hit_summary", "accuracy_detail", "multi_model_dict", "model_metrics",
    "final_predict_result", "multi_model_hit_summary"
]
for key in init_session_keys:
    if key not in st.session_state:
        st.session_state[key] = pd.DataFrame()

# 全局配置
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
st.set_page_config(page_title="快乐8数据分析预测系统V2.0", layout="wide", initial_sidebar_state="expanded")

# 全局常量
LOTTERY_RULE = {"total_numbers":80,"draw_per_period":20,"number_range":range(1,81),"interval_count":8,"tail_number_count":10}

# 数据加载
@st.cache_data(ttl=3600)
def load_standard_data(uploaded_file=None):
    if uploaded_file is None:
        st.info("使用测试数据，上传CSV/Excel切换真实数据")
        mock_data = [[f"2026{p:03d}"]+sorted(np.random.choice(range(1,81),20,replace=False)) for p in range(1,201)]
        raw_df = pd.DataFrame(mock_data, columns=["期号"]+[f"开奖号码{i}" for i in range(1,21)])
    else:
        raw_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.session_state["raw_original_db"] = raw_df.copy()
    df = raw_df.copy().sort_values("期号").reset_index(drop=True)
    df["开奖号码集合"] = df[[f"开奖号码{i}" for i in range(1,21)]].apply(lambda x: set(int(i) for i in x if pd.notna(i)), axis=1)
    return df

# 特征工程
def build_feature_engineer(df, rolling_window_list=[5,10,30]):
    if df.empty: return pd.DataFrame(),[]
    feature_df = df.copy()
    total = len(feature_df)
    # 热度遗漏
    for w in rolling_window_list:
        hot,miss = np.zeros((total,80)),np.zeros((total,80))
        for i in range(total):
            s = max(0,i-w)
            hist = feature_df.loc[s:i-1,"开奖号码集合"].tolist()
            if not hist: continue
            for n in range(1,81):
                hot[i,n-1] = sum(1 for d in hist if n in d)
                for idx,d in enumerate(reversed(hist)):
                    if n in d: miss[i,n-1]=idx+1;break
        for n in range(1,81):
            feature_df[f"近{w}期_热度_{n}"]=hot[:,n-1]
            feature_df[f"近{w}期_遗漏值_{n}"]=miss[:,n-1]
    # 规则特征
    repeat,accompany,follow = np.zeros((total,80)),np.zeros((total,80)),np.zeros((total,80))
    for i in range(1,total):
        h = feature_df.loc[0:i-1]
        if len(h)<2:continue
        # 重号
        rc=defaultdict(int)
        for j in range(len(h)-1):
            l,c=h.iloc[j]["开奖号码集合"],h.iloc[j+1]["开奖号码集合"]
            for n in l:
                if n in c:rc[n]+=1
        for n in range(1,81):repeat[i,n-1]=rc.get(n,0)/(len(h)-1) if len(h)>1 else 0
        # 相随号
        ac=defaultdict(lambda:defaultdict(int))
        for j in range(len(h)-1):
            l,c=h.iloc[j]["开奖号码集合"],h.iloc[j+1]["开奖号码集合"]
            for a in l:
                for b in c:ac[a][b]+=1
        last = feature_df.iloc[i-1]["开奖号码集合"]
        for m in range(1,81):
            total_ac = sum(ac[n].get(m,0) for n in last)
            accompany[i,m-1]=total_ac/(len(last)*(len(h)-1)) if len(last) and len(h)>1 else 0
        # 跟随号
        fc=defaultdict(lambda:defaultdict(int))
        for j in range(len(h)):
            d=list(h.iloc[j]["开奖号码集合"])
            for x in range(len(d)):
                for y in range(x+1,len(d)):
                    fc[d[x]][d[y]]+=1;fc[d[y]][d[x]]+=1
        for n in range(1,81):follow[i,n-1]=sum(fc[n].values())/(20*len(h)) if len(h) else 0
    for n in range(1,81):
        feature_df[f"重号概率_{n}"]=repeat[:,n-1]
        feature_df[f"相随号概率_{n}"]=accompany[:,n-1]
        feature_df[f"跟随号共现度_{n}"]=follow[:,n-1]
    # 高级特征+标签
    def iv(x):return (x-1)//10+1
    for w in rolling_window_list:
        im=np.zeros((total,8))
        for i in range(total):
            s=max(0,i-w)
            h=feature_df.loc[s:i-1,"开奖号码集合"].tolist()
            if not h:continue
            for ivl in range(1,9):im[i,ivl-1]=sum(1 for d in h for num in d if iv(num)==ivl)
        for ivl in range(1,9):feature_df[f"近{w}期_区间{ivl}_热度"]=im[:,ivl-1]
    # 标签
    label=np.zeros((total,80))
    for i in range(total-1):
        nxt=feature_df.iloc[i+1]["开奖号码集合"]
        for num in nxt:label[i,num-1]=1
    for num in range(1,81):feature_df[f"下期是否开出_{num}"]=label[:,num-1]
    feature_df=feature_df.iloc[1:-1].reset_index(drop=True)
    cols=[c for c in feature_df.columns if any(k in c for k in ["热度","遗漏","概率","共现","区间","奇偶","尾号","斜连"])]
    if cols:
        vt=VarianceThreshold(0.01)
        vt.fit(feature_df[cols])
        valid=feature_df[cols].columns[vt.get_support()].tolist()
    else:valid=[]
    st.session_state["feature_columns"]=valid
    st.session_state["feature_engineered_db"]=feature_df.copy()
    return feature_df,valid

# 特征分析
def feature_analysis(fdf,cols):
    if fdf.empty or not cols:return pd.DataFrame(),pd.Series()
    return fdf[cols].corr(),fdf[cols].var().sort_values(ascending=False)

# 多模型训练
def train_multi_model(fdf,cols,weight=None):
    if fdf.empty or not cols:return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    if weight is None:weight={"LogisticRegression":0.15,"RandomForest":0.2,"XGBoost":0.35,"LightGBM":0.3}
    train,val,test = fdf.iloc[:int(len(fdf)*0.7)],fdf.iloc[int(len(fdf)*0.7):int(len(fdf)*0.85)],fdf.iloc[int(len(fdf)*0.85):]
    models={
        "LogisticRegression":LogisticRegression(max_iter=1000,class_weight="balanced"),
        "RandomForest":RandomForestClassifier(50,3),
        "XGBoost":XGBClassifier(50,3,0.1,eval_metric="logloss"),
        "LightGBM":LGBMClassifier(50,3,0.1,verbose=-1)
    }
    st.info("多模型训练中")
    bar=st.progress(0)
    res,metrics,test_pred,latest_pred=defaultdict(dict),defaultdict(list),defaultdict(dict),defaultdict(dict)
    for idx,n in enumerate(range(1,81)):
        bar.progress((idx+1)/80)
        y=fdf[f"下期是否开出_{n}"]
        X_tr,X_va,X_te=train[cols],val[cols],test[cols]
        y_tr,y_va,y_te=train[f"下期是否开出_{n}"],val[f"下期是否开出_{n}"],test[f"下期是否开出_{n}"]
        for name,m in models.items():
            if name in ["XGBoost","LightGBM"]:m.fit(X_tr,y_tr,eval_set=[(X_va,y_va)],verbose=False)
            else:m.fit(X_tr,y_tr)
            test_pred[n][name]=m.predict_proba(X_te)[:,1]
            latest_pred[n][name]=m.predict_proba(fdf.iloc[-1:][cols])[:,1][0]
            try:metrics[name].append(roc_auc_score(y_te,test_pred[n][name]))
            except:metrics[name].append(0.5)
    bar.empty()
    # 结果整理
    final=[]
    for n in range(1,81):
        wp=latest_pred[n]["LogisticRegression"]*0.15+latest_pred[n]["RandomForest"]*0.2+latest_pred[n]["XGBoost"]*0.35+latest_pred[n]["LightGBM"]*0.3
        rw=0.3*fdf.iloc[-1][f"重号概率_{n}"]+0.4*fdf.iloc[-1][f"相随号概率_{n}"]+0.3*fdf.iloc[-1][f"跟随号共现度_{n}"]
        final.append({"号码":n,"最终融合概率":0.6*wp+0.4*rw})
    final_df=pd.DataFrame(final).sort_values("最终融合概率",ascending=False)
    st.session_state["final_predict_result"]=final_df
    return final_df,pd.DataFrame([{"模型名称":k,"平均AUC":np.mean(v)} for k,v in metrics.items()]),pd.DataFrame()

# 单模型训练
def train_predict_model(fdf,cols):
    if fdf.empty or not cols:return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    train,val,test = fdf.iloc[:int(len(fdf)*0.7)],fdf.iloc[int(len(fdf)*0.7):int(len(fdf)*0.85)],fdf.iloc[int(len(fdf)*0.85):]
    st.info("单模型训练中")
    bar=st.progress(0)
    pred,detail=[],[]
    for idx,n in enumerate(range(1,81)):
        bar.progress((idx+1)/80)
        m=XGBClassifier(50,3,0.1,eval_metric="logloss")
        m.fit(train[cols],train[f"下期是否开出_{n}"],eval_set=[(val[cols],val[f"下期是否开出_{n}"])],verbose=False)
        pred.append({"号码":n,"模型预测概率":m.predict_proba(fdf.iloc[-1:][cols])[:,1][0]})
    bar.empty()
    pred_df=pd.DataFrame(pred).sort_values("模型预测概率",ascending=False)
    st.session_state["predict_result"]=pred_df
    return pred_df,pd.DataFrame(),pd.DataFrame()

# 组合生成
def generate_low_repeat_combinations(df,sel=8,grp=5,rate=0.3):
    if df.empty:return pd.DataFrame()
    pool=df.head(sel*3)["号码"].tolist()
    combos=[]
    for _ in range(grp):
        cand=pool[:sel] if not combos else [n for n in pool if sum(n in c for c in combos)/len(combos)<=rate][:sel]
        combo=sorted(cand if cand else pool[:sel])
        if combo not in combos:combos.append(combo)
    while len(combos)<grp:combos.append(sorted(np.random.choice(pool,sel,replace=False)))
    return pd.DataFrame(combos,columns=[f"号码{i+1}" for i in range(sel)],index=[f"第{i+1}组" for i in range(len(combos))])

def generate_dantuo_combinations(df,dan=5,tuo=8,grp=5):
    if df.empty:return pd.DataFrame(),[],[]
    d_pool=df.head(dan*3)["号码"].tolist()
    t_pool=df[~df["号码"].isin(d_pool)].head(tuo*3)["号码"].tolist()
    combos=[(sorted(np.random.choice(d_pool,dan,replace=False)),sorted(np.random.choice([x for x in t_pool if x not in d],tuo,replace=False))) for _ in range(grp)]
    dt=[{"组别":f"第{i+1}组","胆码":"、".join(map(str,d)),"拖码":"、".join(map(str,t))} for i,(d,t) in enumerate(combos)]
    return pd.DataFrame(dt).set_index("组别"),d_pool,t_pool

# 页面主函数（唯一、无重复、全唯一key）
def build_streamlit_page():
    st.title("快乐8数据分析&预测系统V2.0")
    st.markdown("---")
    # 侧边栏
    with st.sidebar:
        uploaded_file=st.file_uploader("上传开奖数据",type=["csv","xlsx"],key="uploader_1")
        rolling=st.multiselect("滚动周期",[5,10,30],[5,10,30],key="ms_1")
        lr=st.slider("LR权重",0.0,1.0,0.15,key="slider_1")
        rf=st.slider("RF权重",0.0,1.0,0.2,key="slider_2")
        xgb=st.slider("XGB权重",0.0,1.0,0.35,key="slider_3")
        lgb=st.slider("LGB权重",0.0,1.0,0.3,key="slider_4")
        sel_cnt=st.slider("单组号码数",5,20,8,key="slider_5")
        grp_cnt=st.slider("组合组数",1,20,5,key="slider_6")
        dan_cnt=st.slider("胆码数",1,10,5,key="slider_7")
        tuo_cnt=st.slider("拖码数",5,20,8,key="slider_8")
    # 加载数据
    df=load_standard_data(uploaded_file)
    # Tab页
    tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8=st.tabs(["数据底层库","特征工程","高级特征","单模型","多模型","命中率","普通组合","胆拖组合"])
    # Tab1
    with tab1:
        st.subheader("原始数据库")
        st.dataframe(st.session_state.raw_original_db,use_container_width=True,key="df_1")
    # Tab2
    with tab2:
        if st.button("一键生成特征",type="primary",key="btn_1"):
            build_feature_engineer(df,rolling)
            st.success("特征生成完成")
    # Tab3
    with tab3:
        fdf=st.session_state.get("feature_engineered_db",pd.DataFrame())
        fcols=st.session_state.get("feature_columns",[])
        if fdf.empty or not fcols:st.warning("先生成特征")
        else:
            corr,fvar=feature_analysis(fdf,fcols)
            st.dataframe(fvar,use_container_width=True,key="df_2")
    # Tab4
    with tab4:
        if st.button("训练单模型",type="primary",key="btn_2"):
            train_predict_model(st.session_state.feature_engineered_db,st.session_state.feature_columns)
            st.success("训练完成")
            st.dataframe(st.session_state.predict_result,use_container_width=True,key="df_3")
    # Tab5
    with tab5:
        if st.button("训练多模型",type="primary",key="btn_3"):
            train_multi_model(st.session_state.feature_engineered_db,st.session_state.feature_columns)
            st.success("训练完成")
            st.dataframe(st.session_state.final_predict_result,use_container_width=True,key="df_4")
    # Tab6
    with tab6:
        st.subheader("训练后查看命中率")
    # Tab7
    with tab7:
        st.subheader("普通组合")
        src=st.radio("选择数据源",["单模型","多模型"],horizontal=True,key="radio_1")
        pred=st.session_state.predict_result if src=="单模型" else st.session_state.final_predict_result
        if not pred.empty and st.button("生成普通组合",type="primary",key="btn_4"):
            cdf=generate_low_repeat_combinations(pred,sel_cnt,grp_cnt)
            st.dataframe(cdf,use_container_width=True,key="df_5")
    # Tab8
    with tab8:
        st.subheader("胆拖组合")
        src=st.radio("选择胆拖数据源",["多模型","单模型"],horizontal=True,key="radio_2")
        pred=st.session_state.final_predict_result if src=="多模型" else st.session_state.predict_result
        if not pred.empty and st.button("生成胆拖组合",type="primary",key="btn_5"):
            dtdf,_,_=generate_dantuo_combinations(pred,dan_cnt,tuo_cnt,grp_cnt)
            st.dataframe(dtdf,use_container_width=True,key="df_6")

# 顶格执行，无缩进
if __name__ == "__main__":
    build_streamlit_page()
