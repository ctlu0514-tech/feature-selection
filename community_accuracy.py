import networkx as nx
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from community_detection import iscd_algorithm_auto_k  # 请确保该模块已正确导入

# 1. 加载图数据
def load_graph(path):
    """从GML文件中加载图"""
    G = nx.read_gml(path)  # 读取 GML 文件

    # 获取每个节点的真实社区标签
    true_partition = {}
    for node, data in G.nodes(data=True):
        true_partition[node] = data.get('value', -1)  # 使用 'value' 属性作为社区标签
    
    # 组织社区节点列表
    community_nodes = {}
    for node, community in true_partition.items():
        if community not in community_nodes:
            community_nodes[community] = []
        community_nodes[community].append(node)
    
    # 按社区编号排序
    sorted_communities = sorted(community_nodes.keys())
    
    # 打印每个社区的节点列表
    print("真实社区节点列表：")
    for community in sorted_communities:
        print(f"社区 {community}: {community_nodes[community]}")
    
    return G

# 2. 获取真实社区标签（从 GML 文件获取 'value'）
def get_true_partition(G):
    """获取每个节点的真实社区标签（基于 value 属性）"""
    true_partition = {}
    for node, data in G.nodes(data=True):
        true_partition[node] = data.get('value', -1)  # 使用 'value' 属性作为社区标签
    return true_partition

# 3. 计算调整兰德指数 (ARI)
def compute_ARI(partition, true_partition):
    """计算调整兰德指数"""
    true_labels = [true_partition[node] for node in partition.keys()]
    predicted_labels = [partition[node] for node in partition.keys()]
    ari = adjusted_rand_score(true_labels, predicted_labels)
    return ari

# 4. 计算标准化互信息 (NMI)
def compute_NMI(partition, true_partition):
    """计算标准化互信息"""
    true_labels = [true_partition[node] for node in partition.keys()]
    predicted_labels = [partition[node] for node in partition.keys()]
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    return nmi

# 5. 使用ISCD+算法进行社区检测
def run_iscd_plus(G):
    """运行 ISCD+ 算法进行社区检测"""
    # 此处你可以调用已有的 ISCD+ 算法，假设你已经有 partition 结果
    partition = iscd_algorithm_auto_k(G)  # 返回格式为 {节点: 社区编号}
    return partition

# 6. 计算并打印检测精度结果（ARI和NMI）
def compute_detection_accuracy(G):
    """计算并打印ARI和NMI"""
    # 获取真实的分区
    true_partition = get_true_partition(G)
    
    # 使用 ISCD+ 算法生成分区
    partition = run_iscd_plus(G)
    
    # 计算 ARI 和 NMI
    ari = compute_ARI(partition, true_partition)
    nmi = compute_NMI(partition, true_partition)
    
    # 打印结果类似于 Table 4 格式
    print("Table 4: The compared results of different algorithms")
    print("-----------------------------------------------------")
    print("Algorithm | ARI    | NMI")
    print("ISCD+     | {:.4f} | {:.4f}".format(ari, nmi))

# 7. 运行并展示精度
def main():
    # 加载图数据
    G = load_graph('/data/qh_20T_share_file/lct/CDGAFS/football/football.gml')
    
    # 计算并打印检测精度（ARI和NMI）
    compute_detection_accuracy(G)

if __name__ == "__main__":
    main()