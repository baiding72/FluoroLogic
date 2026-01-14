"""
电子效应计算器 - 增强版
功能：
1. Hammett 常数查询 (精确匹配 + SMARTS 模式匹配)
2. 电子传递逻辑 (饱和链衰减 + 共轭桥传递)
3. Gasteiger 电荷兜底估算
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
import numpy as np

class HammettCalculator:
    """
    电子效应计算器
    基于 Hansch & Leo (1991) 综述的权威数据
    """
    
    # 电子传递参数
    SATURATED_ATTENUATION = 0.4   # 饱和链每隔一个 CH2 衰减因子
    CONJUGATED_GAMMA = {
        'vinyl': 0.6,      # -CH=CH-
        'ethynyl': 0.6,    # -C≡C-
        'phenyl': 0.35,    # -Ph-
    }
    
    def __init__(self, csv_path="data/Hammett.csv"):
        self.db = self._load_database(csv_path)
        
    def _load_database(self, path):
        """加载 Hammett 数据库并预编译 SMARTS"""
        try:
            # 跳过注释行
            df = pd.read_csv(path, comment='#')
            df = df.where(pd.notnull(df), None)
            
            patterns = []
            for _, row in df.iterrows():
                entry = row.to_dict()
                
                # 清洗数值列中的特殊字符 (如 −0.17 → -0.17)
                for col in ['Sigma_m', 'Sigma_p', 'F', 'R']:
                    val = entry.get(col)
                    if val and isinstance(val, str):
                        val = val.replace('−', '-').strip()
                        try:
                            entry[col] = float(val)
                        except:
                            entry[col] = None
                
                # 编译 SMARTS pattern
                smarts = entry.get('Smarts_Pattern')
                if smarts and isinstance(smarts, str):
                    smarts = smarts.strip()
                    try:
                        entry['mol_pattern'] = Chem.MolFromSmarts(smarts, mergeHs=True)
                        if entry['mol_pattern'] is None:
                            print(f"Warning: Failed to compile SMARTS for {entry.get('Name')}: {smarts}")
                    except Exception as e:
                        print(f"Error compiling SMARTS for {entry.get('Name')}: {e}")
                        entry['mol_pattern'] = None
                else:
                    entry['mol_pattern'] = None
                    
                # 编译 SMILES (用于精确匹配)
                smiles = entry.get('Smiles_Pattern')
                if smiles and isinstance(smiles, str):
                    smiles = smiles.strip()
                    try:
                        clean_smi = smiles.replace('*', '')
                        if clean_smi:
                            entry['mol_exact'] = Chem.MolFromSmiles(clean_smi, sanitize=True)
                        else:
                            entry['mol_exact'] = None
                    except:
                        entry['mol_exact'] = None
                else:
                    entry['mol_exact'] = None
                
                patterns.append(entry)
            
            print(f"Loaded {len(patterns)} entries from Hammett database")
            return patterns
            
        except Exception as e:
            print(f"Error loading Hammett DB: {e}")
            return []

    def get_sigma(self, substituent_smiles, position='para'):
        """
        获取取代基的 Hammett σ 值
        
        参数:
            substituent_smiles: 取代基 SMILES (可带 * 连接点)
            position: 'para' 或 'meta'，决定返回 σ_p 或 σ_m
        
        优先级: 精确SMILES > SMARTS模式 > Gasteiger估算
        """
        if not substituent_smiles or substituent_smiles == "[H]":
            return {"sigma": 0.0, "name": "Hydrogen", "method": "exact"}
        
        # 预处理 - 处理各种连接点格式
        clean_input = substituent_smiles.strip()
        # 移除各种连接点表示
        for pattern in ['[*]', '*', '-*', '[*]-', '-[*]', '*-']:
            clean_input = clean_input.replace(pattern, '')
        clean_input = clean_input.strip('-')
        
        if not clean_input:
            return {"sigma": 0.0, "name": "Unknown", "method": "default"}

        input_mol = Chem.MolFromSmiles(clean_input)
        if not input_mol:
            return {"sigma": None, "name": "Invalid SMILES", "method": "error"}

        sigma_col = 'Sigma_p' if position == 'para' else 'Sigma_m'

        # 1. 精确 SMILES 匹配
        for entry in self.db:
            target_smi = str(entry.get('Smiles_Pattern', '')).replace('*', '').strip()
            if target_smi == clean_input:
                return {
                    "sigma": entry.get(sigma_col),
                    "name": entry.get('Name'),
                    "method": "exact"
                }

        # 2. SMARTS 模式匹配 (优先匹配特定模式，通用模式放最后)
        matched_entry = None
        for entry in self.db:
            pattern = entry.get('mol_pattern')
            if pattern and input_mol.HasSubstructMatch(pattern):
                # 跳过通用模式，先找特定匹配
                if entry.get('Category') != 'generic':
                    matched_entry = entry
                    break
                elif matched_entry is None:
                    # 暂存通用模式作为备选
                    matched_entry = entry
        
        if matched_entry:
            return {
                "sigma": matched_entry.get(sigma_col),
                "name": matched_entry.get('Name'),
                "method": "smarts"
            }

        # 3. Gasteiger 电荷估算 (兜底)
        estimated = self._estimate_sigma_from_charge(input_mol)
        return {
            "sigma": estimated,
            "name": "Estimated",
            "method": "gasteiger"
        }

    def _estimate_sigma_from_charge(self, mol):
        """
        使用 Gasteiger 电荷估算 σ 值
        
        原理：连接点原子的偏电荷反映推拉电子能力
        - 正电荷 → 吸电子 → σ > 0
        - 负电荷 → 供电子 → σ < 0
        """
        try:
            AllChem.ComputeGasteigerCharges(mol)
            # 假设第一个原子是连接点
            charge = mol.GetAtomWithIdx(0).GetDoubleProp('_GasteigerCharge')
            
            # 线性缩放: Gasteiger 范围约 [-0.3, +0.3] → σ 范围约 [-0.5, +0.8]
            # 使用非对称缩放，因为吸电子基通常 σ 值更大
            if charge >= 0:
                sigma = charge * 2.5  # 吸电子方向
            else:
                sigma = charge * 1.5  # 供电子方向
                
            return round(sigma, 2)
        except Exception as e:
            return 0.0

    def calc_effective_sigma(self, mol, attachment_idx, terminal_sigma):
        """
        计算复杂取代基通过链传递后的有效 σ 值
        
        参数:
            mol: RDKit 分子对象
            attachment_idx: 取代基连接到母体的原子索引
            terminal_sigma: 末端官能团的 σ 值
        """
        # 从连接点开始遍历取代基
        chain_type, chain_length = self._analyze_chain(mol, attachment_idx)
        
        if chain_type == 'saturated':
            # 饱和链衰减
            attenuation = self.SATURATED_ATTENUATION ** chain_length
            effective_sigma = terminal_sigma * attenuation
            
            # 如果衰减后效应很小，返回通用烷基值
            if abs(effective_sigma) < 0.05:
                return -0.15, "attenuated_alkyl"
            return effective_sigma, "saturated_chain"
            
        elif chain_type == 'conjugated':
            # 共轭传递
            gamma = self._get_conjugation_gamma(mol, attachment_idx)
            effective_sigma = gamma * terminal_sigma
            return effective_sigma, "conjugated_bridge"
            
        else:
            # 直接连接
            return terminal_sigma, "direct"

    def _analyze_chain(self, mol, start_idx):
        """
        分析取代基的链类型
        返回: (类型, 长度)
        类型: 'saturated', 'conjugated', 'direct'
        """
        atom = mol.GetAtomWithIdx(start_idx)
        
        # 检查是否为饱和碳
        if atom.GetAtomicNum() == 6 and atom.GetHybridization() == Chem.HybridizationType.SP3:
            # 计算饱和碳链长度
            length = self._count_saturated_chain(mol, start_idx, set())
            if length > 0:
                return 'saturated', length
        
        # 检查是否为共轭起点
        if atom.GetAtomicNum() == 6 and atom.GetHybridization() in [Chem.HybridizationType.SP2, Chem.HybridizationType.SP]:
            return 'conjugated', 0
        
        return 'direct', 0

    def _count_saturated_chain(self, mol, start_idx, visited):
        """递归计算饱和碳链长度"""
        if start_idx in visited:
            return 0
        visited.add(start_idx)
        
        atom = mol.GetAtomWithIdx(start_idx)
        if atom.GetAtomicNum() != 6:
            return 0
        if atom.GetHybridization() != Chem.HybridizationType.SP3:
            return 0
            
        max_length = 0
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited:
                length = 1 + self._count_saturated_chain(mol, nbr_idx, visited)
                max_length = max(max_length, length)
        
        return max_length

    def _get_conjugation_gamma(self, mol, start_idx):
        """确定共轭桥的传递系数 γ"""
        atom = mol.GetAtomWithIdx(start_idx)
        
        # 检查是否在芳香环中
        if atom.GetIsAromatic():
            return self.CONJUGATED_GAMMA['phenyl']
        
        # 检查是否为 sp 杂化 (三键)
        if atom.GetHybridization() == Chem.HybridizationType.SP:
            return self.CONJUGATED_GAMMA['ethynyl']
        
        # 默认为双键
        return self.CONJUGATED_GAMMA['vinyl']

    def analyze_substituent(self, smiles):
        """
        全面分析取代基的电子效应
        
        返回包含:
        - sigma_p: para 位 σ 值
        - sigma_m: meta 位 σ 值
        - character: 'EWG' (吸电子) / 'EDG' (供电子) / 'neutral'
        - strength: 'strong' / 'moderate' / 'weak'
        """
        result_p = self.get_sigma(smiles, position='para')
        result_m = self.get_sigma(smiles, position='meta')
        
        sigma_p = result_p.get('sigma', 0) or 0
        sigma_m = result_m.get('sigma', 0) or 0
        
        # 判断推拉电子性质
        if sigma_p > 0.3:
            character = 'EWG'
            strength = 'strong' if sigma_p > 0.6 else 'moderate'
        elif sigma_p < -0.3:
            character = 'EDG'
            strength = 'strong' if sigma_p < -0.6 else 'moderate'
        elif abs(sigma_p) > 0.1:
            character = 'EWG' if sigma_p > 0 else 'EDG'
            strength = 'weak'
        else:
            character = 'neutral'
            strength = 'negligible'
        
        return {
            'smiles': smiles,
            'name': result_p.get('name'),
            'sigma_p': sigma_p,
            'sigma_m': sigma_m,
            'character': character,
            'strength': strength,
            'method': result_p.get('method')
        }


# === LangChain Tool Wrapper ===
def check_hammett(substituent_smiles: str) -> str:
    """
    查询取代基的 Hammett 常数和电子效应分析
    
    Args:
        substituent_smiles: 取代基的 SMILES 表示 (可带 * 连接点)
    
    Returns:
        包含 σ_p, σ_m 值和电子效应分析的文本报告
    """
    calc = HammettCalculator()
    result = calc.analyze_substituent(substituent_smiles)
    
    report = f"""
取代基分析: {result['name']}
SMILES: {result['smiles']}
---
σ_para = {result['sigma_p']:.2f}
σ_meta = {result['sigma_m']:.2f}
---
电子性质: {result['character']} ({result['strength']})
数据来源: {result['method']}
"""
    return report


# 测试代码
if __name__ == "__main__":
    calc = HammettCalculator()
    
    # 测试用例
    tests = [
        "*C",                          # 甲基 → -0.17
        "*[N+](=O)[O-]",              # 硝基 → 0.78
        "*N(C)C",                      # 二甲氨基 → -0.83
        "*CCCCCCCC",                   # 长链烷基 → Generic
        "*C=Cc1ccccc1",               # 苯乙烯基
        "*c1ccc([N+](=O)[O-])cc1",    # 对硝基苯基
        "*C(F)(F)F",                   # 三氟甲基
        "*c1ccccc1",                   # 苯基
    ]
    
    print("=" * 60)
    print("Hammett 电子效应计算器测试")
    print("=" * 60)
    
    for smiles in tests:
        result = calc.analyze_substituent(smiles)
        print(f"\n{smiles}:")
        print(f"  Name: {result['name']}")
        print(f"  σ_p = {result['sigma_p']:.3f}, σ_m = {result['sigma_m']:.3f}")
        print(f"  {result['character']} ({result['strength']}) via {result['method']}")