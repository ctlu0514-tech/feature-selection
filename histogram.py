import os
import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(data, data_name, save_path="./figures", 
                   bins=30, color='skyblue', alpha=0.7, edgecolor='k',
                   figsize=(8, 6), dpi=100, format='png'):
    """
    绘制并保存直方图的函数
  
    参数:
    - data: np.array或列表，输入数据
    - data_name: str，数据名称（用于标题和文件名）
    - save_path: str，图片保存路径，默认'./figures'
    - bins: int，直方图分箱数量，默认30
    - color: str，直方图填充颜色，默认'skyblue'
    - alpha: float，透明度，默认0.7
    - edgecolor: str，边缘颜色，默认'k'（黑色）
    - figsize: tuple，图像尺寸，默认(8,6)
    - dpi: int，分辨率，默认100
    - format: str，保存格式，默认'png'
    """
    # 创建画布
    plt.figure(figsize=figsize, dpi=dpi)
  
    # 绘制直方图
    plt.hist(data, bins=bins, color=color, alpha=alpha, edgecolor=edgecolor)
  
    # 设置标题和标签
    plt.title(f'Distribution of {data_name}', fontsize=12)
    plt.xlabel('Value', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
  
    # 创建保存目录（如果不存在）
    os.makedirs(save_path, exist_ok=True)
  
    # 生成合法文件名（替换空格和特殊字符）
    safe_name = data_name.replace(" ", "_").lower()
    filename = os.path.join(save_path, f"{safe_name}_histogram.{format}")
  
    # 保存并关闭
    plt.savefig(filename, bbox_inches='tight', dpi=dpi)
    plt.close()
    print(f"图片已保存至: {filename}")

# 示例用法
if __name__ == "__main__":
    # 生成测试数据
    normal_data = np.random.normal(0, 1, 1000)
    skewed_data = np.random.exponential(1, 1000)
  
    # 调用函数
    plot_histogram(normal_data, "Normal Distribution", 
                   bins=20, color='salmon', save_path="/data/qh_20T_share_file/lct/CDGAFS")
  
    plot_histogram(skewed_data, "Skewed Data", 
                   bins=50, edgecolor='darkred', format='pdf')