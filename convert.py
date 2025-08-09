import pandas as pd
from collections import Counter # 新增：记录哪些基因由多个探针合并而来

# 1. 读注释，构造 probe → gene 映射
annot = pd.read_csv(
    '/data/qh_20T_share_file/lct/CuMiDa/GPL570-55999.txt',
    sep='\t',
    comment='#',
    dtype=str,
    usecols=['ID', 'Gene Symbol']
)
# 去掉空白
annot['ID'] = annot['ID'].str.strip()
annot['Gene Symbol'] = (
    annot['Gene Symbol']
    .fillna('')
    .str.split(r'\s*///\s*').str[0]
    .str.strip()
)
annot = annot[annot['Gene Symbol'] != '']
probe2gene = dict(zip(annot['ID'], annot['Gene Symbol']))

# 2. 读表达矩阵
expr = pd.read_csv('/data/qh_20T_share_file/lct/CuMiDa/Lung_GSE19804.csv', index_col='samples', dtype=str) #记得改输出文件名
# strip 探针列名，防止多余空格
expr.columns = expr.columns.str.strip()

# 3. 把 type 列单独拿出来
sample_type = expr['type']
expr = expr.drop(columns=['type'])

# 4. 看看哪些 probe 名能映射上
probes_in_expr = set(expr.columns)
mapped = probes_in_expr & set(probe2gene.keys())
print(f"总共 {len(probes_in_expr)} 个探针，能映射上的有 {len(mapped)} 个。")

# 5. 重命名并保留映射上的列
expr = expr.loc[:, sorted(mapped)]  # 保留能映射的列
expr = expr.rename(columns=probe2gene)

# 6. 数据类型转换：字符串→浮点
expr = expr.astype(float)

# 7. 同一基因多探针取平均
#    这里直接按列名（基因符号）聚合
expr_by_gene = expr.groupby(expr.columns, axis=1).mean()

# 8. 把 type 加回去
expr_by_gene.insert(0, 'type', sample_type)

# 9. 保存
expr_by_gene.to_csv('/data/qh_20T_share_file/lct/CuMiDa/Lung_GSE19804_byGene.csv', index_label='samples')
print("转换完成，文件行数：", expr_by_gene.shape[0], "列数：", expr_by_gene.shape[1])
