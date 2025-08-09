import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def draw_graph(G):
    """
    绘制图：增大画布尺寸，缩小节点尺寸，将孤立节点均匀随机分布在画布上。
    Args:
        G (networkx.Graph): 要绘制的图对象。
    """
    # 设置较大的画布尺寸
    plt.figure(figsize=(20, 20))
    
    # 使用 spring_layout 布局计算所有节点的位置
    pos = nx.spring_layout(G, seed=42, k=0.15, iterations=20)
    
    # 定义画布的边界范围（可根据需要修改）
    canvas_min, canvas_max = -2, 2
    
    # 对孤立节点（没有边的节点）采用均匀随机分布在整个画布内
    isolated_nodes = list(nx.isolates(G))
    for node in isolated_nodes:
        pos[node] = np.array([
            np.random.uniform(canvas_min, canvas_max),
            np.random.uniform(canvas_min, canvas_max)
        ])
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    # 绘制节点：减小节点尺寸（例如 node_size=5），颜色和透明度可调
    nx.draw_networkx_nodes(G, pos, node_size=5, node_color="blue", alpha=0.7)
    
    # 如果节点太多，不建议显示所有标签；如果需要可取消注释以下代码
    # nx.draw_networkx_labels(G, pos, font_size=5, font_color='black')
    
    # 设置坐标轴范围与画布边界一致
    plt.xlim(canvas_min, canvas_max)
    plt.ylim(canvas_min, canvas_max)
    
    # 去除坐标轴显示
    plt.axis('off')
    plt.tight_layout()
    plt.show()


    plt.savefig('/data/qh_20T_share_file/lct/CDGAFS/网络图.png', format='png')
