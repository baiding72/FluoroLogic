"""
空间效应指标相关性分析测试
目的：验证各空间指标与还原电位的 Pearson/Spearman 相关系数
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ================= 数据加载 =================
DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/processed/molecules.json')

def load_data():
    """加载并解析 molecules.json"""
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    return data

def extract_features(data):
    """提取空间效应指标用于分析"""
    rows = []
    
    for entry in data:
        if not entry.get('is_bodipy'):
            continue
            
        pot_info = entry.get('potential_info', {})
        states = entry.get('states', {})
        reorg = entry.get('reorganization_metrics', {})
        
        neutral_geom = states.get('neutral', {}).get('geometry', {})
        reduced_geom = states.get('reduced', {}).get('geometry', {})
        
        if not neutral_geom or not reduced_geom:
            continue
            
        # 提取中性态指标
        neutral_heights = neutral_geom.get('steric_heights', {})
        neutral_proximal = neutral_geom.get('proximal_distances', {})
        
        # 提取还原态指标
        reduced_heights = reduced_geom.get('steric_heights', {})
        
        row = {
            'id': entry['id'],
            # 目标变量 (电位)
            'E_red_V': pot_info.get('dft_potential_csv_V'),
            
            # ===== 中性态空间指标 =====
            'neu_core_rmsd': neutral_geom.get('core_rmsd'),
            'neu_max_out_of_plane': neutral_geom.get('max_out_of_plane'),
            'neu_mass_asymmetry': neutral_geom.get('mass_asymmetry'),
            'neu_meso_dihedral': neutral_geom.get('meso_dihedral'),
            'neu_max_height_overall': neutral_heights.get('max_height_overall'),
            'neu_max_height_flanking': neutral_heights.get('max_height_meso_flanking'),
            'neu_max_height_alpha': neutral_heights.get('max_height_alpha'),
            'neu_min_dist_alpha': neutral_proximal.get('min_dist_alpha'),
            'neu_min_dist_flanking': neutral_proximal.get('min_dist_meso_flanking'),
            
            # ===== 还原态空间指标 =====
            'red_core_rmsd': reduced_geom.get('core_rmsd'),
            'red_meso_dihedral': reduced_geom.get('meso_dihedral'),
            'red_max_height_overall': reduced_heights.get('max_height_overall'),
            
            # ===== Delta 指标 (构象变化) =====
            'delta_dihedral': reorg.get('delta_dihedral'),
            'delta_rmsd': reorg.get('delta_rmsd'),
            'delta_max_height': reorg.get('delta_max_height'),
        }
        rows.append(row)
    
    return pd.DataFrame(rows)

def analyze_correlations(df):
    """计算与电位的相关性"""
    # 移除有缺失值的行
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_clean = df[numeric_cols].dropna()
    
    if 'E_red_V' not in df_clean.columns:
        print("Error: E_red_V not found in data")
        return None
        
    # 计算 Pearson 和 Spearman 相关系数
    results = []
    target = 'E_red_V'
    
    for col in numeric_cols:
        if col == target:
            continue
        if df_clean[col].std() == 0:  # 跳过常量列
            continue
            
        pearson_r, pearson_p = stats.pearsonr(df_clean[target], df_clean[col])
        spearman_r, spearman_p = stats.spearmanr(df_clean[target], df_clean[col])
        
        results.append({
            'Metric': col,
            'Pearson_r': round(pearson_r, 4),
            'Pearson_p': round(pearson_p, 4),
            'Spearman_r': round(spearman_r, 4),
            'Spearman_p': round(spearman_p, 4),
            'Significant': 'Yes' if pearson_p < 0.05 else 'No'
        })
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('Pearson_r', ascending=False, key=abs)
    
    return result_df

def plot_correlation_heatmap(df, output_path=None):
    """绘制相关性热力图"""
    # 选择关键指标
    key_cols = [
        'E_red_V',
        'delta_dihedral', 'delta_rmsd', 'delta_max_height',
        'neu_core_rmsd', 'neu_meso_dihedral', 'neu_mass_asymmetry',
        'neu_max_height_overall', 'neu_min_dist_alpha'
    ]
    
    available_cols = [c for c in key_cols if c in df.columns]
    df_subset = df[available_cols].dropna()
    
    corr_matrix = df_subset.corr()
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt='.2f',
        cmap='RdBu_r',
        vmin=-1, vmax=1,
        center=0,
        mask=mask,
        square=True,
        ax=ax
    )
    
    ax.set_title('Steric Metrics vs Redox Potential Correlation Matrix', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved to: {output_path}")
    
    plt.show()
    return fig

def plot_scatter_top_correlations(df, top_n=4, output_path=None):
    """绘制与电位相关性最高的指标的散点图"""
    corr_df = analyze_correlations(df)
    if corr_df is None:
        return
        
    # 选择绝对相关系数最大的前 N 个
    top_metrics = corr_df.head(top_n)['Metric'].tolist()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(top_metrics):
        if i >= len(axes):
            break
        ax = axes[i]
        
        # 清洗数据
        plot_df = df[['E_red_V', metric]].dropna()
        
        ax.scatter(plot_df[metric], plot_df['E_red_V'], alpha=0.6, edgecolor='k', linewidth=0.5)
        
        # 添加回归线
        z = np.polyfit(plot_df[metric], plot_df['E_red_V'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(plot_df[metric].min(), plot_df[metric].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.8, label='Linear Fit')
        
        # 获取相关系数
        r_val = corr_df[corr_df['Metric'] == metric]['Pearson_r'].values[0]
        
        ax.set_xlabel(metric, fontsize=10)
        ax.set_ylabel('E_red (V)', fontsize=10)
        ax.set_title(f'{metric}\n(r = {r_val:.3f})', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Top Correlated Steric Metrics vs Redox Potential', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Scatter plots saved to: {output_path}")
    
    plt.show()
    return fig

def main():
    print("=" * 60)
    print("空间效应指标相关性分析测试")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    data = load_data()
    print(f"    加载分子数: {len(data)}")
    
    # 2. 提取特征
    print("\n[2/4] 提取空间效应指标...")
    df = extract_features(data)
    print(f"    有效分子数: {len(df)}")
    print(f"    指标数量: {len(df.columns) - 1}")  # 减去 id 列
    
    # 3. 相关性分析
    print("\n[3/4] 计算相关性...")
    corr_results = analyze_correlations(df)
    
    print("\n" + "=" * 60)
    print("相关性分析结果 (按 |Pearson r| 排序)")
    print("=" * 60)
    print(corr_results.to_string(index=False))
    
    # 统计显著相关的指标
    sig_count = len(corr_results[corr_results['Significant'] == 'Yes'])
    print(f"\n显著相关指标 (p < 0.05): {sig_count}/{len(corr_results)}")
    
    # 识别强相关指标 (|r| > 0.3)
    strong_corr = corr_results[corr_results['Pearson_r'].abs() > 0.3]
    print(f"强相关指标 (|r| > 0.3): {len(strong_corr)}")
    if len(strong_corr) > 0:
        print("  " + ", ".join(strong_corr['Metric'].tolist()))
    
    # 4. 可视化
    print("\n[4/4] 生成可视化...")
    output_dir = os.path.dirname(__file__)
    
    try:
        plot_correlation_heatmap(df, os.path.join(output_dir, 'steric_correlation_heatmap.png'))
        plot_scatter_top_correlations(df, output_path=os.path.join(output_dir, 'steric_scatter_plots.png'))
    except Exception as e:
        print(f"可视化生成失败: {e}")
        print("(可能是无头环境，跳过图形显示)")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    
    return df, corr_results

if __name__ == "__main__":
    df, results = main()
