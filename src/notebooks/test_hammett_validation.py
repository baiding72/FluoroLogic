"""
使用案例库中的分子数据验证电子效应计算器
"""

import json
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.tools.electronic import HammettCalculator

DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/processed/molecules.json')

def load_molecules():
    with open(DATA_PATH, 'r') as f:
        return json.load(f)

def validate_hammett():
    """验证 Hammett 计算器对案例库中取代基的覆盖率和准确性"""
    
    print("=" * 70)
    print("Hammett 电子效应计算器验证")
    print("=" * 70)
    
    # 加载分子数据
    molecules = load_molecules()
    print(f"加载分子数: {len(molecules)}")
    
    # 初始化计算器
    calc = HammettCalculator()
    
    # 收集所有取代基
    all_substituents = []
    
    for mol in molecules:
        if not mol.get('is_bodipy'):
            continue
            
        subs = mol.get('substituents', {})
        
        for core_idx, sub_list in subs.items():
            for sub in sub_list:
                smiles = sub.get('smiles')
                sub_type = sub.get('type')
                
                if smiles and smiles not in ['[H]', '*']:
                    all_substituents.append({
                        'mol_id': mol['id'],
                        'type': sub_type,
                        'smiles': smiles
                    })
    
    print(f"收集取代基总数: {len(all_substituents)}")
    
    # 去重统计
    unique_smiles = {}
    for sub in all_substituents:
        smiles = sub['smiles']
        if smiles not in unique_smiles:
            unique_smiles[smiles] = {
                'count': 0,
                'types': set(),
                'examples': []
            }
        unique_smiles[smiles]['count'] += 1
        unique_smiles[smiles]['types'].add(sub['type'])
        if len(unique_smiles[smiles]['examples']) < 2:
            unique_smiles[smiles]['examples'].append(sub['mol_id'])
    
    print(f"唯一取代基数: {len(unique_smiles)}")
    
    # 测试每个唯一取代基
    print("\n" + "=" * 70)
    print("取代基覆盖率分析")
    print("=" * 70)
    
    exact_matches = []
    smarts_matches = []
    gasteiger_fallbacks = []
    errors = []
    
    for smiles, info in unique_smiles.items():
        try:
            result = calc.analyze_substituent(smiles)
            
            if result['method'] == 'exact':
                exact_matches.append({
                    'smiles': smiles,
                    'name': result['name'],
                    'sigma_p': result['sigma_p'],
                    'count': info['count']
                })
            elif result['method'] == 'smarts':
                smarts_matches.append({
                    'smiles': smiles,
                    'name': result['name'],
                    'sigma_p': result['sigma_p'],
                    'count': info['count']
                })
            elif result['method'] == 'gasteiger':
                gasteiger_fallbacks.append({
                    'smiles': smiles[:50],
                    'sigma_p': result['sigma_p'],
                    'count': info['count'],
                    'types': list(info['types']),
                    'examples': info['examples']
                })
            else:
                errors.append(smiles)
        except Exception as e:
            errors.append(f"{smiles}: {e}")
    
    # 统计结果
    total = len(unique_smiles)
    print(f"\n精确匹配 (exact): {len(exact_matches)}/{total} ({100*len(exact_matches)/total:.1f}%)")
    print(f"SMARTS 匹配:     {len(smarts_matches)}/{total} ({100*len(smarts_matches)/total:.1f}%)")
    print(f"Gasteiger 估算:  {len(gasteiger_fallbacks)}/{total} ({100*len(gasteiger_fallbacks)/total:.1f}%)")
    print(f"错误:            {len(errors)}/{total}")
    
    # 显示精确匹配的取代基
    print("\n" + "-" * 50)
    print("精确匹配的取代基 (按出现次数排序)")
    print("-" * 50)
    
    exact_matches.sort(key=lambda x: x['count'], reverse=True)
    for item in exact_matches[:15]:
        print(f"  {item['name']:20s} σ_p={item['sigma_p']:+.2f}  (n={item['count']})")
    
    # 显示需要 Gasteiger 估算的取代基
    print("\n" + "-" * 50)
    print("需要 Gasteiger 估算的取代基 (按出现次数排序)")
    print("-" * 50)
    
    gasteiger_fallbacks.sort(key=lambda x: x['count'], reverse=True)
    for item in gasteiger_fallbacks[:10]:
        smiles_short = item['smiles'][:40] + "..." if len(item['smiles']) > 40 else item['smiles']
        print(f"  {smiles_short}")
        print(f"    σ_p≈{item['sigma_p']:+.2f}, 位点={item['types']}, 出现n={item['count']}")
    
    # 按位点类型分析
    print("\n" + "=" * 70)
    print("按位点类型的电子效应分布")
    print("=" * 70)
    
    site_stats = {}
    for sub in all_substituents:
        site = sub['type']
        if site not in site_stats:
            site_stats[site] = {'sigmas': [], 'count': 0}
        
        try:
            result = calc.get_sigma(sub['smiles'], position='para')
            sigma = result.get('sigma', 0) or 0
            site_stats[site]['sigmas'].append(sigma)
            site_stats[site]['count'] += 1
        except:
            pass
    
    for site, stats in sorted(site_stats.items()):
        if stats['sigmas']:
            import numpy as np
            sigmas = np.array(stats['sigmas'])
            print(f"\n{site:15s} (n={stats['count']})")
            print(f"  σ_p 范围: [{sigmas.min():.2f}, {sigmas.max():.2f}]")
            print(f"  σ_p 均值: {sigmas.mean():.2f}")
    
    print("\n" + "=" * 70)
    print("验证完成!")
    print("=" * 70)
    
    return {
        'exact': len(exact_matches),
        'smarts': len(smarts_matches),
        'gasteiger': len(gasteiger_fallbacks),
        'total': total
    }

if __name__ == "__main__":
    validate_hammett()
