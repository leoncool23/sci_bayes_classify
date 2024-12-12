import base64

import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    accuracy_score
)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

# Title and introduction
st.title('朴素贝叶斯分类结局预测模型 (二分类)')
st.markdown("""
    朴素贝叶斯(naivebayes)对二分类结局的资料建立预测模型，包括变量重要性评价、建立模型和内外部验证，计算错误分类百分比和绘制ROC曲线——科研之心提供（https://ai.linkagi.top)。
   
""")

# File upload
st.subheader('数据上传')
uploaded_file = st.file_uploader("上传数据文件", type=["csv", "xls", "xlsx", "sav", "dta", "txt"])

# Model Parameters
st.subheader('模型参数')
target_variable = None
predictors = []

validation_mode = st.radio('验证集产生方式',
                           options=['按数据集顺序产生验证集', '随机选取验证集', 'K折交叉验证'],
                           index=1)

seed = st.number_input('种子数', value=1, min_value=1)
test_size = st.slider('验证集的比例(0-1之间)', 0.0, 1.0, 0.3)
classification_threshold = st.slider('变量取值是几类时可界定为分类变量', 2, 20, 5)
decimal_places = st.selectbox('结果保留小数位数', [0, 1, 2, 3, 4, 5], index=2)

# Function Definitions
def preprocess_data(df, target_variable, predictors, classification_threshold):
    """数据预处理"""
    X = df[predictors]
    y = df[target_variable]

    # Convert categorical columns to numerical using one-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Handle non-numeric values in y by label encoding
    if not np.issubdtype(y.dtype, np.number) or y.nunique() > classification_threshold :
       y = pd.factorize(y)[0]

    return X, y

def train_naive_bayes(X, y):
    """训练朴素贝叶斯模型"""
    if validation_mode == '按数据集顺序产生验证集':
        split_type = False
    elif validation_mode == '随机选取验证集':
        split_type = True
    else:  # K折交叉验证
        return None, None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        shuffle=split_type,
        stratify=y if split_type else None
    )

    model = GaussianNB()
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """模型评估"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    # 错误分类百分比
    misclassification_rate = 1 - accuracy_score(y_test, y_pred)

    # 获取分类报告
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        'roc_curve': (fpr, tpr, roc_auc),
        'confusion_matrix': cm,
        'misclassification_rate': misclassification_rate,
        'classification_report': report,
    }

def plot_roc_curve(fpr, tpr, roc_auc):
    """绘制ROC曲线"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假阳性率')
    ax.set_ylabel('真阳性率')
    ax.set_title('ROC 曲线')
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('混淆矩阵')
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')
    plt.tight_layout()
    return fig

def calculate_permutation_importance(model, X, y, seed):
    """计算排列重要性"""
    result = permutation_importance(model, X, y, n_repeats=10, random_state=seed)
    return result.importances_mean

def plot_permutation_importance(importances, feature_names):
    """绘制排列重要性图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    sorted_idx = importances.argsort()
    ax.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
    ax.set_xlabel("Permutation Importance")
    ax.set_title("变量重要性")
    plt.tight_layout()
    return fig

def download_image(fig, filename):
    """生成图片下载链接"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">下载 {filename}</a>'
    return href

# Main Logic
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("不支持的文件格式")
        st.stop()

    st.write(f"已上传文件: {uploaded_file.name}")
    st.dataframe(df.head())

    target_variable = st.selectbox('选择因变量', options=df.columns.tolist())
    predictors = st.multiselect('选择预测变量', [col for col in df.columns if col != target_variable])

    if target_variable and predictors:
        start_analysis = st.button('开始建模与分析')

        if start_analysis:
            # 数据预处理
            X, y = preprocess_data(df, target_variable, predictors, classification_threshold)

            # 根据选择的验证集方式进行处理
            if validation_mode == 'K折交叉验证':
                # K折交叉验证
                kf = KFold(n_splits=5, shuffle=True, random_state=seed)
                model = GaussianNB()
                scores = cross_val_score(model, X, y, cv=kf, scoring='roc_auc')
                st.write(f'交叉验证的平均AUC：{scores.mean():.{decimal_places}f}')
                st.write(f'交叉验证的标准差：{scores.std():.{decimal_places}f}')

                # 计算排列重要性
                importances = calculate_permutation_importance(model.fit(X,y), X, y, seed)
                st.subheader('变量重要性')
                importance_fig = plot_permutation_importance(importances, X.columns.tolist())
                st.pyplot(importance_fig)
                st.markdown(download_image(importance_fig, "permutation_importance.png"), unsafe_allow_html=True)
                st.stop()

            # 正常的训练与评估过程
            model, X_train, X_test, y_train, y_test = train_naive_bayes(X, y)

            if model is not None:
                # 模型评估
                results = evaluate_model(model, X_test, y_test)

                # 显示错误分类百分比
                st.subheader('模型性能指标')
                st.write(f'错误分类百分比：{results["misclassification_rate"]:.{decimal_places}f}')

                # 保存 ROC 曲线并显示
                st.subheader('ROC 曲线')
                roc_fig = plot_roc_curve(*results['roc_curve'])
                st.pyplot(roc_fig)
                st.markdown(download_image(roc_fig, "roc_curve.png"), unsafe_allow_html=True)

                # 保存混淆矩阵并显示
                st.subheader('混淆矩阵')
                cm_fig = plot_confusion_matrix(results['confusion_matrix'])
                st.pyplot(cm_fig)
                st.markdown(download_image(cm_fig, "confusion_matrix.png"), unsafe_allow_html=True)

                # 显示分类报告
                st.subheader('分类报告')
                report_dict = results['classification_report']
                report_df = pd.DataFrame(report_dict).transpose().round(decimal_places)
                st.dataframe(report_df)

                # 计算排列重要性
                importances = calculate_permutation_importance(model, X_test, y_test, seed)
                st.subheader('变量重要性')
                importance_fig = plot_permutation_importance(importances, X_test.columns.tolist())
                st.pyplot(importance_fig)
                st.markdown(download_image(importance_fig, "permutation_importance.png"), unsafe_allow_html=True)

            else:
                st.error("模型训练失败")

else:
    st.info("请上传数据文件以继续")
