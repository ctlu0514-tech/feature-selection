import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from CDGAFS import cdgafs_feature_selection
from evaluate import evaluate_classifiers, evaluate_with_repeats
from normalize_scores import normalize_scores  # 归一化函数

# 定义数据集加载函数
def load_dataset(dataset_id):
    if dataset_id == 1:
        data = pd.read_csv("/data/qh_20T_share_file/lct/CDGAFS/data_spambase/spambase.data", header=None)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
    elif dataset_id == 2:
        data = pd.read_csv("/data/qh_20T_share_file/lct/CDGAFS/data_arrhythmia/arrhythmia.csv", header=None)
        data = data.apply(pd.to_numeric, errors="coerce")
        data.fillna(data.mean(), inplace=True)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
    elif dataset_id == 3:
        data = pd.read_csv("/data/qh_20T_share_file/lct/CDGAFS/data_wbdc/wbdc.csv", header=None)
        X = data.iloc[:, 2:].values
        y = data.iloc[:, 1].values
        y = np.where(y == 'M', 1, 0)
    elif dataset_id == 4:
        data = pd.read_csv('/data/qh_20T_share_file/lct/CDGAFS/data_arcene/ARCENE/processed_data.csv')
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
    return X, y

# 参数敏感性测试函数
def parameter_sensitivity_test(dataset_id, dataset_name):
    X, y = load_dataset(dataset_id)
    # 打印移除前的特征数量
    print(f"移除常数特征前，特征数量: {X.shape[1]}")
    # 计算每列（特征）的方差
    non_zero_variance_indices = np.var(X, axis=0) != 0
    # 移除方差为零的列（常数特征）
    X= X[:, non_zero_variance_indices]
    # 打印移除后的特征数
    print(f"移除常数特征后，特征数量: {X.shape[1]}")

    # 对每个特征进行Softmax缩放
    X = np.apply_along_axis(normalize_scores, axis=0, arr=X)  # 特征归一化

    # 新增基准测试
    print(f"\n评估完整数据集性能: {dataset_name}")
    baseline = evaluate_classifiers(X, y, f"{dataset_name}-完整特征")
  
    results = {
        'Baseline': baseline,
        'Theta': [],
        'Omega': []
    }
  
    # 测试Theta敏感性（固定omega=0.1）
    for theta in [0.5, 0.6, 0.7, 0.8]:
        print(f"\nTheta={theta} 测试中...")
        selected_features = cdgafs_feature_selection(
            X=X, y=y, theta=theta, omega=0.3, population_size=100
        )
        eval_results = evaluate_with_repeats(X, y, selected_features, f"{dataset_name}-Theta={theta}")
        results['Theta'].append({
            'theta': theta,
            'results': eval_results
        })
  
    # 测试Omega敏感性（固定theta=0.9）
    for omega in [0.1, 0.2, 0.3, 0.4, 0.5]:
        print(f"\nOmega={omega} 测试中...")
        selected_features = cdgafs_feature_selection(
            X=X, y=y, theta=0.8, omega=omega, population_size=100
        )
        eval_results = evaluate_with_repeats(X, y, selected_features, f"{dataset_name}-Omega={omega}")
        results['Omega'].append({
            'omega': omega,
            'results': eval_results
        })
  
    return results

# 可视化函数
def plot_sensitivity(full_results, parameter_type, dataset_name):
    """修改后的可视化函数"""
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    colors = plt.cm.tab10.colors
  
    # 从完整结果中获取基准数据
    baseline = full_results['Baseline']
  
    # 获取参数测试数据
    param_data = full_results[parameter_type]
  
    # 绘制基准线（完整特征结果）
    for idx, model in enumerate(['KNN', 'SVM', 'AdaBoost']):
        ax.axhline(y=baseline[model], color=colors[idx], 
                  linestyle='--', linewidth=2,
                  label=f'{model} Baseline')
  
    # 绘制参数曲线（特征选择结果）
    for idx, model in enumerate(['KNN', 'SVM', 'AdaBoost']):
        if parameter_type == 'Theta':
            x = [d['theta'] for d in param_data]
        else:
            x = [d['omega'] for d in param_data]
      
        y = [d['results'][model] for d in param_data]  # 注意此处访问层级
        ax.plot(x, y, color=colors[idx], 
               marker='o', linewidth=2.5,
               label=f'{model} Selected Features')
  
    # 图表装饰（保持原有逻辑）
    plt.title(f'{dataset_name} - {parameter_type}敏感性分析')
    plt.xlabel('Theta值' if parameter_type == 'Theta' else 'Omega值')
    plt.ylabel('分类准确率')

    plt.ylim(0, 1)  # 添加y轴限制设置

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_{parameter_type}_sensitivity.png')
    plt.close()

# 主程序
if __name__ == "__main__":
    datasets = [
        # (1, "Spambase"),
        # (2, "Arrhythmia"),
        # (3, "WDBC"),
        (4, "ARCENE")
    ]
  
    for dataset_id, dataset_name in datasets:
        print(f"\n{'='*30}\nProcessing {dataset_name}\n{'='*30}")
        # 获取完整结果字典
        full_results = parameter_sensitivity_test(dataset_id, dataset_name)
      
        # 分别传递完整结果和参数类型
        plot_sensitivity(full_results, 'Theta', dataset_name)
        plot_sensitivity(full_results, 'Omega', dataset_name)
        print(f"Saved plots for {dataset_name}")

    print("\nAll sensitivity analysis completed. Check generated PNG files.")