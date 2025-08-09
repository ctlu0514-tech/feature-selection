import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif

def build_mi_feature(X, bins=10, discrete_threshold=20):
    """
    自适应计算混合特征的互信息矩阵
    Args:
        X (np.ndarray): 特征矩阵 (样本数, 特征数)
        bins (int): 连续特征分箱数
        discrete_threshold (int): 离散特征判定阈值（唯一值数 < 阈值）
    Returns:
        np.ndarray: 互信息矩阵 (特征数, 特征数)
    """
    n_samples, n_features = X.shape
    mi_matrix = np.zeros((n_features, n_features))
  
    # 转换为DataFrame处理缺失值和类型
    df = pd.DataFrame(X)
  
    # 离散化连续特征
    X_discrete = df.copy()
    for i in range(n_features):
        unique_vals = df[i].nunique()
        if unique_vals >= discrete_threshold:  # 连续特征需分箱
            try:
                # 分箱并直接获取编码（无需cat.codes）
                X_discrete[i] = pd.cut(df[i], bins=bins, labels=False, duplicates='drop')
            except (ValueError, TypeError):
                # 分箱失败时视为单箱（如全相同值）
                X_discrete[i] = 0
        # 转换为整数（无论连续/离散）
        X_discrete[i] = X_discrete[i].astype(int)
  
    # 计算互信息矩阵
    X_np = X_discrete.to_numpy()
    for i in range(n_features):
        for j in range(i, n_features):
            mi = mutual_info_score(X_np[:, i], X_np[:, j])
            mi_matrix[i, j] = mi_matrix[j, i] = mi
  
    return mi_matrix

def build_mi_label(X: np.ndarray, y: np.ndarray, method: str = 'average') -> np.ndarray:
    """
    构建基因-标签互信息网络的邻接矩阵。
    
    参数:
        X (np.ndarray): 形状为 (n_samples, n_features) 的特征矩阵。
        y (np.ndarray): 标签向量，形状为 (n_samples,)。
        method (str): 构建权重的方式，可选 'average'、'min'、'product'。
    
    返回:
        A (np.ndarray): 形状为 (n_features, n_features) 的邻接矩阵。
    """
    # Step 1: 计算每个特征与标签的互信息
    mi_scores = mutual_info_classif(X, y, discrete_features='auto')

    # Step 2: 构建邻接矩阵
    n_features = X.shape[1]
    A = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(i+1, n_features):
            if method == 'average':
                weight = (mi_scores[i] + mi_scores[j]) / 2
            elif method == 'min':
                weight = min(mi_scores[i], mi_scores[j])
            elif method == 'product':
                weight = mi_scores[i] * mi_scores[j]
            else:
                raise ValueError("method 只能是 'average'、'min' 或 'product'")

            A[i, j] = weight
            A[j, i] = weight  # 保证对称

    return A
