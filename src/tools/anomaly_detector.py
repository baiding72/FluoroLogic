"""
异常电位识别工具
检测违反化学直觉的还原电位结果

异常类型:
1. Hammett 偏差异常 - 电位与电子效应预测不符
2. 构象重组异常 - 异常的几何变化导致电位偏移
3. 电子-空间冲突 - 电子效应与空间效应矛盾
"""

import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# 尝试导入项目模块
try:
    from src.tools.electronic import HammettCalculator
except ImportError:
    HammettCalculator = None


class AnomalyType(Enum):
    """异常类型枚举"""
    HAMMETT_DEVIATION = "hammett_deviation"  # Hammett 预测偏差
    REORGANIZATION_ANOMALY = "reorganization_anomaly"  # 结构重组异常
    ELECTRONIC_STERIC_CONFLICT = "electronic_steric_conflict"  # 电子-空间冲突
    DIHEDRAL_INVERSION = "dihedral_inversion"  # 二面角反转异常
    UNEXPLAINED_SHIFT = "unexplained_shift"  # 无法解释的电位偏移


@dataclass
class Anomaly:
    """异常检测结果"""
    mol_id: str
    anomaly_type: AnomalyType
    severity: str  # 'low', 'medium', 'high'
    description: str
    evidence: Dict
    suggested_cause: str


class AnomalyDetector:
    """
    异常电位检测器
    
    检测策略:
    1. 统计异常: 电位超出同类分子的合理范围
    2. Hammett 偏差: 电位与取代基电子效应预测不符
    3. 构象异常: 还原后几何变化异常
    4. 效应冲突: 电子效应和空间效应的预测相矛盾
    """
    
    # 阈值参数
    SIGMA_THRESHOLD = 2.0      # 统计异常阈值 (标准差倍数)
    HAMMETT_SLOPE = -0.15      # σ_p 对电位的影响斜率 (V/σ)
    HAMMETT_TOLERANCE = 0.15   # Hammett 预测容差 (V)
    DIHEDRAL_THRESHOLD = 30    # 二面角变化阈值 (度)
    RMSD_THRESHOLD = 0.15      # RMSD 变化阈值 (Å)
    
    def __init__(self, molecules: List[Dict], hammett_calc=None):
        """
        初始化检测器
        
        Args:
            molecules: molecules.json 中的分子列表
            hammett_calc: HammettCalculator 实例 (可选)
        """
        self.molecules = molecules
        self.hammett_calc = hammett_calc
        
        # 预计算统计量
        self._compute_statistics()
    
    def _compute_statistics(self):
        """计算电位分布统计"""
        potentials = []
        for mol in self.molecules:
            if mol.get('is_bodipy'):
                e_red = mol.get('potential_info', {}).get('dft_potential_csv_V')
                if e_red is not None:
                    potentials.append(e_red)
        
        self.potentials = np.array(potentials)
        self.mean_potential = np.mean(self.potentials)
        self.std_potential = np.std(self.potentials)
        
        print(f"电位统计: μ={self.mean_potential:.3f}V, σ={self.std_potential:.3f}V")
    
    def detect_all_anomalies(self) -> List[Anomaly]:
        """检测所有分子的异常"""
        all_anomalies = []
        
        for mol in self.molecules:
            if not mol.get('is_bodipy'):
                continue
            
            anomalies = self.detect_anomalies(mol)
            all_anomalies.extend(anomalies)
        
        return all_anomalies
    
    def detect_anomalies(self, mol: Dict) -> List[Anomaly]:
        """
        检测单个分子的所有异常
        
        返回: 该分子的异常列表
        """
        anomalies = []
        mol_id = mol.get('id', 'unknown')
        
        # 1. 统计异常检测
        stat_anomaly = self._detect_statistical_anomaly(mol)
        if stat_anomaly:
            anomalies.append(stat_anomaly)
        
        # 2. Hammett 偏差检测
        hammett_anomaly = self._detect_hammett_deviation(mol)
        if hammett_anomaly:
            anomalies.append(hammett_anomaly)
        
        # 3. 构象重组异常检测
        reorg_anomaly = self._detect_reorganization_anomaly(mol)
        if reorg_anomaly:
            anomalies.append(reorg_anomaly)
        
        # 4. 电子-空间冲突检测
        conflict_anomaly = self._detect_electronic_steric_conflict(mol)
        if conflict_anomaly:
            anomalies.append(conflict_anomaly)
        
        return anomalies
    
    def _detect_statistical_anomaly(self, mol: Dict) -> Optional[Anomaly]:
        """检测统计异常 (电位超出 ±2σ)"""
        e_red = mol.get('potential_info', {}).get('dft_potential_csv_V')
        if e_red is None:
            return None
        
        z_score = (e_red - self.mean_potential) / self.std_potential
        
        if abs(z_score) > self.SIGMA_THRESHOLD:
            direction = "更正" if z_score > 0 else "更负"
            severity = "high" if abs(z_score) > 3 else "medium"
            
            return Anomaly(
                mol_id=mol['id'],
                anomaly_type=AnomalyType.UNEXPLAINED_SHIFT,
                severity=severity,
                description=f"电位 {e_red:.3f}V 显著{direction}于平均值 ({z_score:+.1f}σ)",
                evidence={
                    'e_red': e_red,
                    'z_score': z_score,
                    'mean': self.mean_potential,
                    'std': self.std_potential
                },
                suggested_cause="可能存在特殊的稳定化/去稳定化机制"
            )
        
        return None
    
    def _detect_hammett_deviation(self, mol: Dict) -> Optional[Anomaly]:
        """检测 Hammett 预测偏差"""
        if self.hammett_calc is None:
            return None
        
        e_red = mol.get('potential_info', {}).get('dft_potential_csv_V')
        if e_red is None:
            return None
        
        # 获取 Meso 位取代基的 sigma 值
        subs = mol.get('substituents', {})
        meso_sigma = 0.0
        meso_name = None
        
        for core_idx, sub_list in subs.items():
            for sub in sub_list:
                if sub.get('type') == 'meso':
                    smiles = sub.get('smiles', '')
                    if smiles and smiles != '[H]':
                        result = self.hammett_calc.get_sigma(smiles, position='para')
                        meso_sigma = result.get('sigma', 0) or 0
                        meso_name = result.get('name', 'Unknown')
                        break
        
        # 基于 Hammett 的电位预测
        # E_pred = E_base + slope * sigma
        e_base = self.mean_potential
        e_predicted = e_base + self.HAMMETT_SLOPE * meso_sigma
        deviation = e_red - e_predicted
        
        if abs(deviation) > self.HAMMETT_TOLERANCE:
            direction = "更正" if deviation > 0 else "更负"
            severity = "high" if abs(deviation) > 0.25 else "medium"
            
            return Anomaly(
                mol_id=mol['id'],
                anomaly_type=AnomalyType.HAMMETT_DEVIATION,
                severity=severity,
                description=f"实际电位比 Hammett 预测{direction} {abs(deviation):.2f}V",
                evidence={
                    'e_red_actual': e_red,
                    'e_red_predicted': e_predicted,
                    'deviation': deviation,
                    'meso_sigma': meso_sigma,
                    'meso_substituent': meso_name
                },
                suggested_cause="可能存在空间效应或共轭调制"
            )
        
        return None
    
    def _detect_reorganization_anomaly(self, mol: Dict) -> Optional[Anomaly]:
        """检测构象重组异常"""
        reorg = mol.get('reorganization_metrics', {})
        
        delta_dihedral = reorg.get('delta_dihedral')
        delta_rmsd = reorg.get('delta_rmsd')
        
        anomalies_found = []
        
        # 检测大幅二面角变化
        if delta_dihedral is not None and abs(delta_dihedral) > self.DIHEDRAL_THRESHOLD:
            direction = "变平" if delta_dihedral < 0 else "扭曲加剧"
            anomalies_found.append(f"二面角变化 {delta_dihedral:+.1f}° ({direction})")
        
        # 检测大幅 RMSD 变化
        if delta_rmsd is not None and delta_rmsd > self.RMSD_THRESHOLD:
            anomalies_found.append(f"骨架变形 ΔRMSD={delta_rmsd:.3f}Å")
        
        if anomalies_found:
            return Anomaly(
                mol_id=mol['id'],
                anomaly_type=AnomalyType.REORGANIZATION_ANOMALY,
                severity="medium",
                description="; ".join(anomalies_found),
                evidence={
                    'delta_dihedral': delta_dihedral,
                    'delta_rmsd': delta_rmsd
                },
                suggested_cause="还原态构象弛豫可能提供额外稳定化能"
            )
        
        return None
    
    def _detect_electronic_steric_conflict(self, mol: Dict) -> Optional[Anomaly]:
        """检测电子效应与空间效应的冲突"""
        if self.hammett_calc is None:
            return None
        
        e_red = mol.get('potential_info', {}).get('dft_potential_csv_V')
        reorg = mol.get('reorganization_metrics', {})
        states = mol.get('states', {})
        
        if e_red is None:
            return None
        
        # 获取电子效应预测
        subs = mol.get('substituents', {})
        total_sigma = 0.0
        
        for core_idx, sub_list in subs.items():
            for sub in sub_list:
                smiles = sub.get('smiles', '')
                if smiles and smiles != '[H]':
                    result = self.hammett_calc.get_sigma(smiles, position='para')
                    sigma = result.get('sigma', 0) or 0
                    total_sigma += sigma
        
        # 获取空间效应指标
        neu_geom = states.get('neutral', {}).get('geometry', {})
        meso_dihedral = neu_geom.get('meso_dihedral')
        
        # 检测冲突：
        # 1. 强吸电子基 (sigma > 0.3) 但电位很负
        # 2. 强供电子基 (sigma < -0.3) 但电位很正
        # 同时考虑二面角的影响
        
        if meso_dihedral is not None:
            # 大二面角会切断共轭，削弱电子效应
            if meso_dihedral > 60 and abs(total_sigma) > 0.3:
                expected_effect = "吸电子" if total_sigma > 0 else "供电子"
                
                return Anomaly(
                    mol_id=mol['id'],
                    anomaly_type=AnomalyType.ELECTRONIC_STERIC_CONFLICT,
                    severity="low",
                    description=f"大二面角 ({meso_dihedral:.0f}°) 可能削弱{expected_effect}效应",
                    evidence={
                        'meso_dihedral': meso_dihedral,
                        'total_sigma': total_sigma,
                        'e_red': e_red
                    },
                    suggested_cause="共轭被二面角切断，电子效应传递受阻"
                )
        
        return None


def analyze_anomalies(mol_id: str = None) -> str:
    """
    LangChain Tool: 分析异常电位
    
    Args:
        mol_id: 可选，指定分子ID。如果不指定，分析所有分子
    
    Returns:
        异常分析报告
    """
    import os
    
    # 加载数据
    data_path = os.path.join(os.path.dirname(__file__), '../data/processed/molecules.json')
    with open(data_path, 'r') as f:
        molecules = json.load(f)
    
    # 初始化检测器
    hammett_calc = HammettCalculator() if HammettCalculator else None
    detector = AnomalyDetector(molecules, hammett_calc)
    
    if mol_id:
        # 分析特定分子
        mol = next((m for m in molecules if m['id'] == mol_id), None)
        if not mol:
            return f"未找到分子: {mol_id}"
        
        anomalies = detector.detect_anomalies(mol)
    else:
        # 分析所有分子
        anomalies = detector.detect_all_anomalies()
    
    # 生成报告
    if not anomalies:
        return "未检测到异常电位"
    
    report = f"检测到 {len(anomalies)} 个异常\n\n"
    
    for a in anomalies[:10]:  # 限制输出数量
        report += f"[{a.severity.upper()}] {a.mol_id}\n"
        report += f"  类型: {a.anomaly_type.value}\n"
        report += f"  描述: {a.description}\n"
        report += f"  可能原因: {a.suggested_cause}\n\n"
    
    return report


# 测试
if __name__ == "__main__":
    import os
    import sys
    
    # 添加项目路径
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    from src.tools.electronic import HammettCalculator
    
    # 加载数据
    data_path = os.path.join(os.path.dirname(__file__), '../../data/processed/molecules.json')
    with open(data_path, 'r') as f:
        molecules = json.load(f)
    
    print("=" * 70)
    print("异常电位检测器测试")
    print("=" * 70)
    
    # 初始化检测器
    hammett_calc = HammettCalculator()
    detector = AnomalyDetector(molecules, hammett_calc)
    
    # 检测所有异常
    all_anomalies = detector.detect_all_anomalies()
    
    print(f"\n检测到 {len(all_anomalies)} 个异常")
    
    # 按类型统计
    by_type = {}
    for a in all_anomalies:
        t = a.anomaly_type.value
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(a)
    
    print("\n按类型统计:")
    for t, items in sorted(by_type.items()):
        print(f"  {t}: {len(items)}")
    
    # 显示高严重度异常
    high_severity = [a for a in all_anomalies if a.severity == 'high']
    print(f"\n高严重度异常 ({len(high_severity)} 个):")
    
    for a in high_severity[:5]:
        print(f"\n  [{a.mol_id}] {a.anomaly_type.value}")
        print(f"    {a.description}")
        print(f"    原因: {a.suggested_cause}")
