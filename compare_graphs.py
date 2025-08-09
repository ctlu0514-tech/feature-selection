# -*- coding: utf-8 -*-
# 文件名: compare_graphs.py
# 描述: 用于对比仅使用皮尔逊相关性构建的图 vs 融合背景知识构建的图。
#      该脚本会计算并打印两种图的结构指标，并生成可视化图片。
# 版本: v2 - 修复了模块度计算中的 NotAPartition 错误。

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# --- 从你的项目中导入必要的函数 ---
from fisher_score import compute_fisher_score
from normalize_scores import normalize_scores
from construct_feature_graph import construct_feature_graph
from community_detection import iscd_algorithm_auto_k

# --- 辅助函数：专门用于构建一个仅基于皮尔逊的图 ---
def construct_pearson_only_graph(X, theta):
    """
    仅使用皮尔逊相关系数构建特征图。
    """
    print("\n--- 构建仅基于皮尔逊的图 ---")
    pearson_matrix = np.corrcoef(X, rowvar=False)
    abs_corr = np.abs(pearson_matrix)
    pearson_norm = normalize_scores(abs_corr)
    
    graph_matrix = pearson_norm.copy()
    
    num_edges_before = np.count_nonzero(graph_matrix) // 2
    graph_matrix[graph_matrix < theta] = 0
    num_edges_after = np.count_nonzero(graph_matrix) // 2
    print(f"根据阈值 {theta} 过滤的边数: {num_edges_before - num_edges_after}")

    np.fill_diagonal(graph_matrix, 0)
    graph_matrix = np.nan_to_num(graph_matrix, nan=0)
    
    G = nx.from_numpy_array(graph_matrix)
    print("皮尔逊图构建完成:")
    print(G)
    return G

# --- 辅助函数：用于分析和可视化图 (已修复) ---
def analyze_and_visualize(G, graph_name, gene_names):
    """
    计算图的各项指标并进行可视化。
    """
    print(f"\n--- 分析图: {graph_name} ---")

    # 1. 计算量化指标
    # 模块度（需要先进行社区划分）
    print("正在进行社区检测以计算模块度...")
    partition = iscd_algorithm_auto_k(G)
    
    # ========================> 错误修复逻辑 <========================
    if not partition: # 如果图中没有边，partition可能为空
        print("图中没有社区可以被检测到。")
        modularity = 0.0
    else:
        # 将社区划分转换为 networkx 需要的格式 (列表的列表)
        communities_dict = defaultdict(list)
        for node, comm_id in partition.items():
            communities_dict[comm_id].append(node)
        communities = [set(nodes) for nodes in communities_dict.values()]
        
        # 处理孤立节点，将每个孤立节点视为一个独立的社区
        all_nodes_in_graph = set(G.nodes())
        partitioned_nodes = set(partition.keys())
        isolated_nodes = all_nodes_in_graph - partitioned_nodes
        for iso_node in isolated_nodes:
            communities.append({iso_node}) # 每个孤立点自成一派
            
        # 现在 communities 是一个覆盖所有节点的有效划分
        modularity = nx.community.modularity(G, communities)
    # =================================================================
    
    print(f"模块度 (Modularity): {modularity:.4f}")

    # 平均聚类系数
    avg_clustering = nx.average_clustering(G)
    print(f"平均聚类系数 (Average Clustering Coefficient): {avg_clustering:.4f}")

    # 连通分量
    num_components = nx.number_connected_components(G)
    print(f"连通分量数量 (Number of Connected Components): {num_components}")

    # 2. 可视化
    print("正在生成网络可视化图...")
    plt.figure(figsize=(16, 16))
    
    pos = nx.spring_layout(G, seed=42)
    
    nx.draw_networkx_nodes(G, pos, node_size=10, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)
    
    plt.title(f"Feature Graph Visualization: {graph_name}", fontsize=20)
    plt.savefig(f"{graph_name}.png", dpi=150)
    plt.close()
    print(f"可视化图已保存为: {graph_name}.png")


if __name__ == '__main__':
    # --- 参数设置 ---
    PRE_FILTER_TOP_N = 4000
    THETA = 0.8
    FILE_NAME = 'Brain_GSE50161_byGene.csv'
    LABEL_COL_NAME = 'type'

    # --- 数据加载与预处理 (与 main.py 逻辑一致) ---
    print("--- 1. 加载和预处理数据 ---")
    total_data = pd.read_csv('/data/qh_20T_share_file/lct/CuMiDa/' + FILE_NAME)
    col_name = total_data.columns
    patient_id_col = col_name[0]
    
    feature_columns = [col for col in col_name if col not in [LABEL_COL_NAME, patient_id_col]]
    variances = total_data[feature_columns].var()
    non_zero_var_features = variances[variances > 0].index.tolist()
    
    total_data_clean = total_data.dropna(subset=non_zero_var_features)
    
    labels = total_data_clean[LABEL_COL_NAME].values
    
    feature_min = total_data_clean[non_zero_var_features].min()
    feature_max = total_data_clean[non_zero_var_features].max()
    normalized_features = (total_data_clean[non_zero_var_features] - feature_min) / (feature_max - feature_min + 1e-10)
    
    gene_list_full = normalized_features.columns.tolist()
    feature_data_full = normalized_features.values

    # --- 预筛选 (与 CDGAFS.py 逻辑一致) ---
    print("\n--- 2. 执行特征预筛选 ---")
    fisher_scores = compute_fisher_score(feature_data_full, labels)
    top_indices = np.argsort(fisher_scores)[-PRE_FILTER_TOP_N:]
    
    X_subset = feature_data_full[:, top_indices]
    gene_list_subset = [gene_list_full[i] for i in top_indices]
    print(f"预筛选完成，保留 {len(gene_list_subset)} 个特征进行图构建。")

    # --- 构建并分析两个图 ---
    # 1. 构建并分析“基础图”（仅皮尔逊）
    G_pearson = construct_pearson_only_graph(X_subset, theta=THETA)
    analyze_and_visualize(G_pearson, "Pearson_Only_Graph", gene_list_subset)

    # 2. 构建并分析“融合图”（你的原始方法）
    print("\n--- 构建融合背景知识的图 ---")
    G_fused = construct_feature_graph(X_subset, labels, gene_list_subset, theta=THETA)
    analyze_and_visualize(G_fused, "Fused_Knowledge_Graph", gene_list_subset)

    print("\n--- 对比分析完成 ---")
    print("请查看生成的 .png 图片，并对比控制台输出的量化指标。")
