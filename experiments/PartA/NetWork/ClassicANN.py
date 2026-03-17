# ClassicANN.py
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import warnings

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, adjusted_rand_score,
    adjusted_mutual_info_score, normalized_mutual_info_score
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# ======================= 模型定义 (采用官方默认，并修复序列化与类型问题) =======================

def knn_cell_annotation(X_ref, labels_ref, X_query):
    model = KNeighborsClassifier(n_neighbors=15, metric='euclidean', n_jobs=-1)
    model.fit(X_ref, labels_ref)
    return model.predict(X_query)

def svm_cell_annotation(X_ref, labels_ref, X_query):
    model = SVC(kernel='rbf', probability=False, random_state=42)
    model.fit(X_ref, labels_ref)
    return model.predict(X_query)

def random_forest_cell_annotation(X_ref, labels_ref, X_query):
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_ref, labels_ref)
    return model.predict(X_query)

def logistic_regression_annotation(X_ref, labels_ref, X_query):
    # 【修复点 1】：移除 n_jobs=-1。避免 object 类型字符串数组在多进程 pickling 时序列化失败
    model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000, random_state=42)
    model.fit(X_ref, labels_ref)
    return model.predict(X_query)

def naive_bayes_annotation(X_ref, labels_ref, X_query):
    model = GaussianNB()
    model.fit(X_ref, labels_ref)
    return model.predict(X_query)

def xgboost_annotation(X_ref, labels_ref, X_query):
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels_ref)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1)
    model.fit(X_ref, y_encoded)
    pred_encoded = model.predict(X_query)
    return le.inverse_transform(pred_encoded)

def mlp_annotation(X_ref, labels_ref, X_query):
    # 【修复点 2】：引入 LabelEncoder 隔离文本标签，防止 MLP 底层调用 np.isnan 时引发类型不支持错误
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels_ref)
    model = MLPClassifier(early_stopping=True, random_state=42)
    model.fit(X_ref, y_encoded)
    pred_encoded = model.predict(X_query)
    return le.inverse_transform(pred_encoded)

# ======================= 执行与评估总控 =======================

def run_and_evaluate_all_classic_methods(
    X_ref_features: np.ndarray,
    X_query_features: np.ndarray,
    labels_ref_str: np.ndarray,
    true_labels_query_str: np.ndarray,
    fold_info: Dict,
    methods_to_run: List[str]
) -> List[Dict]:
    
    results_this_fold =[]

    method_function_map = {
        'kNN': knn_cell_annotation,
        'SVM': svm_cell_annotation,
        'RandomForest': random_forest_cell_annotation,
        'LogisticRegression': logistic_regression_annotation,
        'NaiveBayes': naive_bayes_annotation,
        'XGBoost': xgboost_annotation,
        'MLP': mlp_annotation
    }

    for method_name in methods_to_run:
        if method_name not in method_function_map:
            continue

        print(f"      正在训练与评估: {method_name} ...")
        func_to_call = method_function_map[method_name]

        try:
            # 执行预测 (纯文本标签出入)
            pred_labels_str = func_to_call(X_ref_features, labels_ref_str, X_query_features)
            
            # 捕获异常输出，将其强行赋值为 Unknown 接受性能惩罚
            pred_labels_str = np.array(pred_labels_str, dtype=str)
            pred_labels_str[pd.isna(pred_labels_str) | (pred_labels_str == 'nan')] = "Unknown"

            # 七大指标统一计算
            all_labels = list(set(true_labels_query_str) | set(pred_labels_str))
            
            acc = accuracy_score(true_labels_query_str, pred_labels_str)
            f1_mi = f1_score(true_labels_query_str, pred_labels_str, labels=all_labels, average='micro', zero_division=0)
            f1_ma = f1_score(true_labels_query_str, pred_labels_str, labels=all_labels, average='macro', zero_division=0)
            f1_we = f1_score(true_labels_query_str, pred_labels_str, labels=all_labels, average='weighted', zero_division=0)
            ari = adjusted_rand_score(true_labels_query_str, pred_labels_str)
            ami = adjusted_mutual_info_score(true_labels_query_str, pred_labels_str)
            nmi = normalized_mutual_info_score(true_labels_query_str, pred_labels_str)

            print(f"        -> {method_name} 完毕: Macro-F1 = {f1_ma:.4f}")

            results_this_fold.append({
                **fold_info,
                'method': method_name,
                'accuracy': acc, 'f1_micro': f1_mi, 'f1_macro': f1_ma, 'f1_weighted': f1_we,
                'ari': ari, 'ami': ami, 'nmi': nmi
            })

        except Exception as e:
            print(f"        -> {method_name} 失败: {str(e)}")
            results_this_fold.append({
                **fold_info, 'method': method_name, 
                'accuracy': np.nan, 'f1_micro': np.nan, 'f1_macro': np.nan, 'f1_weighted': np.nan,
                'ari': np.nan, 'ami': np.nan, 'nmi': np.nan
            })

    return results_this_fold