"""
测试新增的空间效应指标: β位二面角 + 共轭链长度
"""

import json
import pandas as pd
import numpy as np
import os
from scipy import stats

DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/processed/molecules.json')

def load_data():
    with open(DATA_PATH, 'r') as f:
        return json.load(f)

def test_new_metrics():
    """测试新指标的功能和分布"""
    print("=" * 70)
    print("新增空间效应指标测试")
    print("=" * 70)
    
    # 使用 file_reader 重新处理分子以获取新指标
    # 这里我们先手动加载一个测试分子
    
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
    
    from src.utils.chem_utils import BodipyScaffoldMatcher, BodipyStericAnalyzer
    from src.utils.file_reader import DataIntegrator
    
    # 加载现有数据检查结构
    data = load_data()
    print(f"加载分子数: {len(data)}")
    
    # 检查是否有已处理的新指标
    sample = data[0]
    states = sample.get('states', {})
    neu_geom = states.get('neutral', {}).get('geometry', {})
    
    print("\n现有指标字段:")
    for key in neu_geom.keys():
        print(f"  - {key}")
    
    # 检查是否需要重新处理
    has_beta = 'beta_dihedrals' in neu_geom
    has_conj = 'conjugation_lengths' in neu_geom
    
    if has_beta and has_conj:
        print("\n✅ 新指标已存在于数据中!")
        analyze_existing_data(data)
    else:
        print("\n⚠️ 新指标尚未计算，需要重新运行 file_reader.py")
        print("运行命令: python src/utils/file_reader.py")
        
        # 演示单分子测试
        print("\n" + "=" * 50)
        print("单分子测试演示")
        print("=" * 50)
        demo_single_molecule()

def analyze_existing_data(data):
    """分析已有数据中的新指标"""
    beta_angles = []
    conj_lengths = []
    potentials = []
    
    for mol in data:
        if not mol.get('is_bodipy'):
            continue
            
        states = mol.get('states', {})
        neu_geom = states.get('neutral', {}).get('geometry', {})
        pot_info = mol.get('potential_info', {})
        
        # 提取 beta 二面角
        beta_info = neu_geom.get('beta_dihedrals', {})
        avg_beta = beta_info.get('avg_beta_dihedral')
        
        # 提取共轭长度
        conj_info = neu_geom.get('conjugation_lengths', {})
        max_conj = conj_info.get('max_conjugation_length', 0)
        
        e_red = pot_info.get('dft_potential_csv_V')
        
        if e_red is not None:
            potentials.append(e_red)
            beta_angles.append(avg_beta if avg_beta else 0)
            conj_lengths.append(max_conj if max_conj else 0)
    
    df = pd.DataFrame({
        'E_red_V': potentials,
        'avg_beta_dihedral': beta_angles,
        'max_conjugation_length': conj_lengths
    })
    
    print("\n新指标统计:")
    print(df.describe())
    
    print("\n与电位的相关性:")
    for col in ['avg_beta_dihedral', 'max_conjugation_length']:
        valid = df[[col, 'E_red_V']].dropna()
        if len(valid) > 5:
            r, p = stats.pearsonr(valid[col], valid['E_red_V'])
            print(f"  {col}: r={r:.4f}, p={p:.4f}")

def demo_single_molecule():
    """演示单分子新指标计算"""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
    
    try:
        from src.utils.file_reader import DataIntegrator
        
        # 加载数据集成器
        integrator = DataIntegrator()
        
        # 选择一个测试分子
        test_mol_id = "BE_Br"  # 一个有 Beta 位 Br 取代的分子
        
        print(f"\n测试分子: {test_mol_id}")
        
        RAW_DFT_DIR = "data/raw_DFT"
        neu_path = os.path.join(RAW_DFT_DIR, f"{test_mol_id}_sp.log")
        
        if os.path.exists(neu_path):
            neu_res = integrator._parse_log_structure(neu_path)
            if neu_res:
                mol = neu_res['mol']
                is_bodipy, scaffold = integrator.matcher.analyze(mol)
                
                if is_bodipy:
                    from src.utils.chem_utils import BodipyStericAnalyzer
                    
                    subs = integrator.matcher.extract_substituents_detailed(mol, scaffold)
                    analyzer = BodipyStericAnalyzer(mol, scaffold)
                    
                    # 计算新指标
                    beta_result = analyzer.calc_beta_dihedrals(subs)
                    conj_result = analyzer.calc_conjugation_lengths(subs)
                    
                    print("\nβ位二面角计算结果:")
                    print(f"  详情: {beta_result.get('beta_dihedrals')}")
                    print(f"  平均: {beta_result.get('avg_beta_dihedral')}")
                    print(f"  最大: {beta_result.get('max_beta_dihedral')}")
                    
                    print("\n共轭链长度计算结果:")
                    print(f"  详情: {conj_result.get('conjugation_details')}")
                    print(f"  最大: {conj_result.get('max_conjugation_length')}")
                    print(f"  总计: {conj_result.get('total_conjugation_length')}")
                else:
                    print("无法识别 BODIPY 骨架")
            else:
                print("解析分子结构失败")
        else:
            print(f"文件不存在: {neu_path}")
            
    except Exception as e:
        print(f"演示过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_new_metrics()
