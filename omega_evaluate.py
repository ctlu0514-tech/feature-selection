import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import defaultdict

# 自定义模块
from fisher_score import compute_fisher_score
from softmax_scaling import softmax_scaling
from normalize_scores import normalize_scores
from construct_feature_graph import construct_feature_graph
from community_detection import iscd_algorithm_auto_k
from initialize_population import initialize_population
from calculate_fitness import calculate_fitness
from genetic_algorithm_utils import genetic_algorithm
from final_subset_selection import final_subset_selection
from evaluate import evaluate_with_repeats

def feature_selection_pipeline(X_train, y_train, X_normalized, y, clusters, fisher_scores, normalized_fisher_scores, omega, population_size=100):
    """
    执行特征选择流程并返回最终分类性能。
    """
    # 初始化种群
    num_features = X_train.shape[1]
    population = initialize_population(num_features, clusters, omega, population_size)

    # 计算适应度
    fitness_values = calculate_fitness(population, X_train, y_train)

    # 遗传算法优化
    population, fitness_values = genetic_algorithm(
        population, fitness_values, X_train, y_train, clusters, omega, population_size,
        num_features, fisher_scores, normalized_fisher_scores
    )

    # 最终子集选择
    _, selected_features = final_subset_selection(population, fitness_values)

    # 评估分类器性能
    accuracy = evaluate_with_repeats(X_normalized, y, selected_features, f"Omega={omega}")

    return accuracy

if __name__ == "__main__":
    # 数据集路径
    data_paths = {
        "Dataset 1": "/data/qh_20T_share_file/lct/CDGAFS/wbdc/wbdc.csv",
        "Dataset 2": "/data/qh_20T_share_file/lct/CDGAFS/arrhythmia/arrhythmia.csv",
        # "Dataset 3": "/data/qh_20T_share_file/lct/CDGAFS/dataset3.csv"
    }

    # 设置不同的 omega 值范围
    omega_values = list(range(1, 7))

    # 初始化结果字典
    results = {name: [] for name in data_paths.keys()}

    # 遍历不同数据集
    for dataset_name, file_path in data_paths.items():
        print(f"\n=== 开始评估 {dataset_name} ===")

        # 数据加载和预处理
        if dataset_name == "Dataset 1":
            data = pd.read_csv(file_path, header=None)
            X = data.iloc[:, 2:].values  # 第3列到最后为特征
            y = data.iloc[:, 1].values   # 第2列为标签（M/B）
            y = np.where(y == 'M', 1, 0)  # 将 M 转换为 1，B 转换为 0

        elif dataset_name == "Dataset 2":
            data = pd.read_csv(file_path, header=None)
            data = data.apply(pd.to_numeric, errors="coerce")  # 将所有列转换为数值型，强制非数值变为 NaN
            data.fillna(data.mean(), inplace=True)  # 用均值填充缺失值
            X = data.iloc[:, :-1].values  # 除最后一列外为特征
            y = data.iloc[:, -1].values   # 最后一列为标签

        # 数据归一化（Softmax 缩放）
        X_normalized = np.apply_along_axis(softmax_scaling, axis=0, arr=X)

        # 数据划分
        X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

        # Fisher Scores（只需计算一次）
        fisher_scores = compute_fisher_score(X_train, y_train)
        normalized_fisher_scores = normalize_scores(fisher_scores)

        # 特征图构建（只需计算一次）
        feature_graph = construct_feature_graph(X_train, theta=0.2)

        # 社区检测（只需计算一次）
        partition = iscd_algorithm_auto_k(feature_graph)
        clusters = defaultdict(list)
        for node, community in partition.items():
            clusters[community].append(node)
        clusters = [cluster for cluster in clusters.values()]

        # 遍历不同的 omega 值
        for omega in omega_values:
            print(f"评估 {dataset_name}, Omega = {omega}")
            accuracy = feature_selection_pipeline(
                X_train, y_train, X_normalized, y, clusters, fisher_scores, normalized_fisher_scores, omega
            )
            results[dataset_name].append(accuracy)

            # 打印所有分类器及其对应的准确率
            for clf_name, clf_accuracy in accuracy.items():
                print(f"{dataset_name}, Omega = {omega}, {clf_name} 分类准确率 = {clf_accuracy:.4f}")

    # 绘制不同数据集、分类器的对比曲线
    plt.figure(figsize=(12, 8))
    for dataset_name, accuracies in results.items():
        clf_results = defaultdict(list)
        for accuracy_dict in accuracies:
            for clf_name, clf_accuracy in accuracy_dict.items():
                clf_results[clf_name].append(clf_accuracy)

        # 针对每个分类器绘制曲线
        for clf_name, clf_accuracies in clf_results.items():
            plt.plot(omega_values, clf_accuracies, marker='o', linestyle='-', label=f"{dataset_name} - {clf_name}")

    plt.title("Effect of Omega on Classification Accuracy for Multiple Datasets and Classifiers")
    plt.xlabel("Omega (Number of Features per Cluster)")
    plt.ylabel("Classification Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig("omega_vs_accuracy_classifiers.png")


    # 输出最终结果
    for dataset_name, accuracies in results.items():
        print(f"\n=== {dataset_name} 最终结果 ===")
        for omega, accuracy_dict in zip(omega_values, accuracies):
            print(f"Omega = {omega}:")
            for clf_name, clf_accuracy in accuracy_dict.items():
                print(f"    {clf_name} 准确率 = {clf_accuracy:.4f}")
