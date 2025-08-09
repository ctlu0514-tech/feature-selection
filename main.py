# =================================================================
# 文件: main.py (修改后)
# 主要改动：
# 1. 在调用 cdgafs_feature_selection 时增加了 pre_filter_top_n 参数。
# 2. 在评估阶段，使用原始的 feature_data，但传入的是最终筛选出的原始索引。
# 3. 添加了将最终选择的特征索引映射回基因名的功能。
# =================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # 数据集划分
from collections import defaultdict  # 用于存储和管理簇信息
from CDGAFS import cdgafs_feature_selection
from evaluate import evaluate_classifiers, evaluate_with_repeats  # 自定义的评估函数

file_name='Brain_GSE50161_byGene.csv'#数据集名称
label_line_name='type'  #哪一列作为最终的标签

# 读取并预处理数据
def read_data():
    total_data = pd.read_csv('/data/qh_20T_share_file/lct/CuMiDa/'+file_name)
    col_name = total_data.columns
    patient_id_col = col_name[0]
    label_col = label_line_name

    # 提取特征列（排除患者ID和标签列）
    feature_columns = [col for col in col_name if col not in [label_col, patient_id_col]]
    
    print(f"原始特征数量: {len(feature_columns)}")
    # 移除方差为0的特征
    variances = total_data[feature_columns].var()
    non_zero_var_features = variances[variances > 0].index.tolist()
    cal_feature_list = non_zero_var_features
    print(f"移除常数特征后特征数量: {len(non_zero_var_features)}")
    
    # 筛选没有缺失值的样本  
    total_data_clean = total_data.dropna(subset=cal_feature_list)
    lack_person_indices = total_data.index.difference(total_data_clean.index).tolist()
    lack_person_num = [str(idx) for idx in lack_person_indices]
    print('哪一行有空缺值:', lack_person_num)

    # 处理标签
    labels = total_data_clean[label_col].values
    label_list = labels.tolist()

    # 归一化特征
    feature_min = total_data_clean[cal_feature_list].min()
    feature_max = total_data_clean[cal_feature_list].max()
    normalized_features = (total_data_clean[cal_feature_list] - feature_min) / (feature_max - feature_min + 1e-10)
    
    # 获取完整的基因列表和特征矩阵
    gene_list_full = normalized_features.columns.tolist()
    feature_data_full = normalized_features.values
    label_data = np.array(label_list)

    # 调用CDGAFS特征选择，并加入预筛选步骤
    # ========================> 新增关键参数 <========================
    # pre_filter_top_n: 设置为 4000，意味着只对 Fisher Score 最高的 4000 个特征进行复杂分析。
    # 如果你的内存依然紧张，可以适当调低这个数值，比如 2000。如果资源充足，可以调高。
    selected_original_indices = cdgafs_feature_selection(
        X=feature_data_full, 
        y=label_data,
        gene_list=gene_list_full,
        theta=0.5, 
        omega=0.5, 
        population_size=100,
        pre_filter_top_n=4000  # <--- 核心改动：引入预筛选
    )
    
    # 将筛选出的原始索引映射到基因名
    selected_gene_names = [gene_list_full[i] for i in selected_original_indices]
    print(f"\n最终选择的 {len(selected_gene_names)} 个特征（基因名）: {selected_gene_names}")

    # 保存选中的特征到文件
    selected_df = total_data_clean[selected_gene_names]
    output_path = '/data/qh_20T_share_file/lct/CDGAFS/Selected_Features.csv'
    selected_df.to_csv(output_path, index=False)
    print(f"选中的特征已保存到: {output_path}")

    return feature_data_full, selected_original_indices, label_data

# 模型训练和评估 
if __name__ == "__main__":
    feature_data, selected_features, label_data = read_data()
    
    print("\n--- 开始评估分类性能 ---")
    # 原始数据评估
    print("\n1. 使用完整特征集进行评估:")
    evaluate_classifiers(feature_data, label_data, "完整数据集")
    
    # 评估CDGAFS特征子集
    print("\n2. 使用 CDGAFS 选择的特征集进行评估:")
    evaluate_with_repeats(feature_data, label_data, selected_features, "CDGAFS特征选择数据")
