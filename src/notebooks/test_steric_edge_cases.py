"""
边界案例验证测试
目的：验证空间指标能否解释违反 Hammett 预测的异常电位案例
"""

import json
import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/processed/molecules.json')

def load_data():
    with open(DATA_PATH, 'r') as f:
        return json.load(f)

def find_flattening_molecules(data):
    """
    测试用例1: 寻找还原后"变平" (Flattening) 的分子
    物理假设: delta_dihedral < -20° 表示显著变平，可能带来额外稳定化
    """
    results = []
    
    for mol in data:
        if not mol.get('is_bodipy'):
            continue
            
        reorg = mol.get('reorganization_metrics', {})
        delta_dihed = reorg.get('delta_dihedral')
        
        if delta_dihed is None:
            continue
            
        if delta_dihed < -20:  # 显著变平
            pot_info = mol.get('potential_info', {})
            states = mol.get('states', {})
            
            results.append({
                'id': mol['id'],
                'E_red_V': pot_info.get('dft_potential_csv_V'),
                'neu_dihedral': states.get('neutral', {}).get('geometry', {}).get('meso_dihedral'),
                'red_dihedral': states.get('reduced', {}).get('geometry', {}).get('meso_dihedral'),
                'delta_dihedral': delta_dihed,
                'delta_rmsd': reorg.get('delta_rmsd'),
                'behavior': 'Flattening'
            })
    
    return pd.DataFrame(results)

def find_rigid_molecules(data):
    """
    测试用例2: 寻找构象刚性的分子 (几乎不变化)
    物理假设: |delta_dihedral| < 5° 表示构象刚性
    """
    results = []
    
    for mol in data:
        if not mol.get('is_bodipy'):
            continue
            
        reorg = mol.get('reorganization_metrics', {})
        delta_dihed = reorg.get('delta_dihedral')
        
        if delta_dihed is None:
            continue
            
        if abs(delta_dihed) < 5:  # 刚性
            pot_info = mol.get('potential_info', {})
            states = mol.get('states', {})
            
            results.append({
                'id': mol['id'],
                'E_red_V': pot_info.get('dft_potential_csv_V'),
                'neu_dihedral': states.get('neutral', {}).get('geometry', {}).get('meso_dihedral'),
                'red_dihedral': states.get('reduced', {}).get('geometry', {}).get('meso_dihedral'),
                'delta_dihedral': delta_dihed,
                'delta_rmsd': reorg.get('delta_rmsd'),
                'behavior': 'Rigid'
            })
    
    return pd.DataFrame(results)

def find_folding_molecules(data):
    """
    测试用例3: 寻找有折叠构象的分子
    物理假设: proximal_distance < 4.0 Å 可能表示 π-stacking
    """
    results = []
    
    for mol in data:
        if not mol.get('is_bodipy'):
            continue
            
        states = mol.get('states', {})
        neu_geom = states.get('neutral', {}).get('geometry', {})
        proximal = neu_geom.get('proximal_distances', {})
        
        min_dist_alpha = proximal.get('min_dist_alpha')
        
        if min_dist_alpha and min_dist_alpha < 4.5:
            pot_info = mol.get('potential_info', {})
            
            results.append({
                'id': mol['id'],
                'E_red_V': pot_info.get('dft_potential_csv_V'),
                'min_dist_alpha': min_dist_alpha,
                'meso_dihedral': neu_geom.get('meso_dihedral'),
                'behavior': 'Potential_Folding'
            })
    
    return pd.DataFrame(results)

def compare_similar_pairs(data):
    """
    测试用例4: 寻找结构相似但电位差异大的分子对
    用于验证空间效应能否解释差异
    """
    # 提取所有数据
    molecules = []
    for mol in data:
        if not mol.get('is_bodipy'):
            continue
            
        pot_info = mol.get('potential_info', {})
        reorg = mol.get('reorganization_metrics', {})
        states = mol.get('states', {})
        
        e_red = pot_info.get('dft_potential_csv_V')
        if e_red is None:
            continue
            
        molecules.append({
            'id': mol['id'],
            'E_red_V': e_red,
            'delta_dihedral': reorg.get('delta_dihedral'),
            'delta_rmsd': reorg.get('delta_rmsd'),
            'neu_dihedral': states.get('neutral', {}).get('geometry', {}).get('meso_dihedral')
        })
    
    df = pd.DataFrame(molecules)
    
    # 找到电位最正和最负的分子
    if len(df) < 2:
        return None
        
    most_positive = df.nlargest(5, 'E_red_V')
    most_negative = df.nsmallest(5, 'E_red_V')
    
    return {
        'most_positive': most_positive,
        'most_negative': most_negative
    }

def agent_reasoning_test_cases():
    """
    生成用于测试 Agent 推理能力的对话示例
    """
    test_cases = [
        {
            "question": "为什么 BE_24NO2 的电位比 BE_23NO2 更负?",
            "expected_tool_calls": ["analyze_structural_reorganization"],
            "expected_reasoning": [
                "BE_24NO2 有更大的 delta_dihedral (-57.4° vs 0.5°)",
                "还原后发生显著构象弛豫 (Flattening)",
                "产生额外稳定化能，导致电位偏正(应该比纯电子效应预测更正)"
            ]
        },
        {
            "question": "M100023564 的二面角为什么是 null?",
            "expected_tool_calls": ["query_bodi_database"],
            "expected_reasoning": [
                "该分子在 Meso 位没有共轭取代基",
                "连接的是 sp3 碳或氢原子",
                "不存在可测量的二面角"
            ]
        },
        {
            "question": "哪些分子在还原时发生显著的构象变化?",
            "expected_tool_calls": ["query_bodi_database"],
            "expected_reasoning": [
                "搜索 delta_dihedral > 20° 或 < -20° 的分子",
                "分析这些分子的共同结构特征"
            ]
        }
    ]
    return test_cases

def main():
    print("=" * 70)
    print("边界案例验证测试")
    print("=" * 70)
    
    data = load_data()
    print(f"加载分子数: {len(data)}")
    
    # 测试1: Flattening 分子
    print("\n" + "=" * 70)
    print("测试1: Flattening 分子 (delta_dihedral < -20°)")
    print("=" * 70)
    
    flatten_df = find_flattening_molecules(data)
    print(f"发现 {len(flatten_df)} 个 Flattening 分子\n")
    
    if len(flatten_df) > 0:
        print(flatten_df.to_string(index=False))
        
        # 统计分析
        mean_potential = flatten_df['E_red_V'].mean()
        print(f"\n平均电位: {mean_potential:.3f} V")
    
    # 测试2: 刚性分子
    print("\n" + "=" * 70)
    print("测试2: 刚性分子 (|delta_dihedral| < 5°)")
    print("=" * 70)
    
    rigid_df = find_rigid_molecules(data)
    print(f"发现 {len(rigid_df)} 个刚性分子\n")
    
    if len(rigid_df) > 0:
        print(rigid_df.head(10).to_string(index=False))
        mean_potential_rigid = rigid_df['E_red_V'].mean()
        print(f"\n平均电位: {mean_potential_rigid:.3f} V")
    
    # 对比: Flattening vs Rigid
    if len(flatten_df) > 0 and len(rigid_df) > 0:
        print("\n" + "-" * 50)
        print("Flattening vs Rigid 电位对比:")
        print(f"  Flattening 平均电位: {flatten_df['E_red_V'].mean():.3f} V (n={len(flatten_df)})")
        print(f"  Rigid 平均电位: {rigid_df['E_red_V'].mean():.3f} V (n={len(rigid_df)})")
        delta = flatten_df['E_red_V'].mean() - rigid_df['E_red_V'].mean()
        print(f"  差异: {delta:.3f} V")
        
        if delta > 0.05:
            print("  → Flattening 分子电位更正 (符合物理预期: 额外稳定化)")
        elif delta < -0.05:
            print("  → Flattening 分子电位更负 (需进一步分析)")
        else:
            print("  → 两组差异不显著")
    
    # 测试3: 折叠构象
    print("\n" + "=" * 70)
    print("测试3: 可能存在折叠的分子 (min_dist_alpha < 4.5 Å)")
    print("=" * 70)
    
    folding_df = find_folding_molecules(data)
    print(f"发现 {len(folding_df)} 个潜在折叠分子\n")
    
    if len(folding_df) > 0:
        print(folding_df.head(10).to_string(index=False))
    
    # 测试4: 极端电位对比
    print("\n" + "=" * 70)
    print("测试4: 电位极端案例对比")
    print("=" * 70)
    
    pairs = compare_similar_pairs(data)
    if pairs:
        print("\n最正电位 Top 5:")
        print(pairs['most_positive'].to_string(index=False))
        
        print("\n最负电位 Top 5:")
        print(pairs['most_negative'].to_string(index=False))
    
    # Agent 测试用例
    print("\n" + "=" * 70)
    print("Agent 推理测试用例 (人工验证)")
    print("=" * 70)
    
    test_cases = agent_reasoning_test_cases()
    for i, case in enumerate(test_cases, 1):
        print(f"\n[Case {i}] {case['question']}")
        print(f"  预期调用工具: {case['expected_tool_calls']}")
        print(f"  预期推理要点:")
        for point in case['expected_reasoning']:
            print(f"    - {point}")
    
    print("\n" + "=" * 70)
    print("边界案例测试完成!")
    print("=" * 70)
    
    return flatten_df, rigid_df, folding_df

if __name__ == "__main__":
    flatten_df, rigid_df, folding_df = main()
