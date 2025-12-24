import sys
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

import sys
from enum import Enum
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

# ================= 枚举定义 =================

class HybridizationType(str, Enum):
    """原子杂化方式枚举"""
    SP3 = "sp3"         # 四配位，四面体构型 (如: 甲基 -CH3)
    SP2 = "sp2"         # 三配位，平面三角形 (如: 苯环, 羰基, 烯烃)
    SP = "sp"           # 二配位，直线型 (如: 炔烃)
    S = "s"             # 氢原子
    UNKNOWN = "unknown" # 未知或判定失败

class ConnectionType(str, Enum):
    """Meso 位连接基团类型枚举"""
    HYDROGEN = "hydrogen"         # 连接的是氢原子
    ALKYL = "alkyl"               # 烷基 (如: Methyl, Ethyl, Cyclohexyl)，通常无共轭
    ARYL = "aryl"                 # 芳基 (如: Phenyl)，主要共轭来源
    CONJUGATED = "conjugated"     # 非芳香性的共轭链 (如: Alkenyl, Carbonyl)
    ALKYNYL = "alkynyl"           # 炔基 (直线型)
    HETEROATOM = "heteroatom"     # 杂原子 (如: N, O, S, F, Cl)，涉及孤对电子共轭
    UNKNOWN = "unknown"           # 无法识别

# ===========================================

class BodipyScaffoldMatcher:
    """
    BODIPY 骨架与取代基分析器
    [升级版] 集成 N-B-N 闭环验证，精准识别单体，排除二聚体干扰。
    """
    def __init__(self):
        # 1. 核心模式：Meso碳 连接两个 5元环 (C-C-C-C-N)
        # 这是一个"开放式"的配体骨架，还没限制 B 的连接
        self.smarts_str = "[#6](~[#6]1~[#6]~[#6]~[#6]~[#7]1)(~[#6]1~[#6]~[#6]~[#6]~[#7]1)"
        self.core_smarts = Chem.MolFromSmarts(self.smarts_str)

    def analyze(self, mol):
        """
        分析分子，返回 (是否匹配成功, 骨架信息字典)。
        包含 N-B-N 拓扑验证。
        """
        if not mol or not self.core_smarts:
            return False, None

        # 1. 骨架初筛 (找到 dipyrromethene 配体核心)
        try:
            matches = mol.GetSubstructMatches(self.core_smarts)
        except:
            return False, None
            
        if not matches:
            return False, None
        
        # 2. 遍历所有匹配，寻找符合 N-B-N 闭环的那个
        # (二聚体可能有多个 match，比如桥碳和真 Meso 碳，我们需要过滤)
        valid_match = None
        
        for match in matches:
            # SMARTS 索引映射:
            # 0: Meso
            # 5: N1 (左环)
            # 10: N2 (右环)
            n1_idx = match[5]
            n2_idx = match[10]
            
            n1_atom = mol.GetAtomWithIdx(n1_idx)
            n2_atom = mol.GetAtomWithIdx(n2_idx)
            
            # === 核心验证：N1 和 N2 必须连接到同一个 B ===
            common_boron = None
            
            # 找 N1 的邻居
            n1_nbrs = [x.GetIdx() for x in n1_atom.GetNeighbors()]
            # 找 N2 的邻居
            n2_nbrs = [x.GetIdx() for x in n2_atom.GetNeighbors()]
            
            # 求交集
            common_nbrs = set(n1_nbrs).intersection(set(n2_nbrs))
            
            for idx in common_nbrs:
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetAtomicNum() == 5: # 5 = Boron
                    common_boron = idx
                    break
            
            if common_boron is not None:
                # 找到了！这就是我们要的真 BODIPY 骨架
                # 即使分子有其他硼，只要这个骨架闭环了，就是合法的
                valid_match = match
                break
        
        if valid_match is None:
            # 虽然匹配到了 C-吡咯-吡咯，但 N 没有被 B 锁住
            # 可能是二聚体的桥碳，或者是未配位的配体
            return False, {"error": "no_N_B_N_bridge"}
            
        # 3. 组装结果
        match = valid_match
        scaffold_info = {
            "meso_idx": match[0],
            "alpha_indices": [match[1], match[6]], 
            "beta_indices": [match[2], match[3], match[7], match[8]], 
            "nitrogen_indices": [match[5], match[10]],
            "boron_idx": common_boron, # 顺便存下 B 的位置
            "all_core_indices": set(match)
        }
        
        return True, scaffold_info

    def analyze_meso_structure(self, mol, scaffold_info):
        """
        [逻辑修正版] 深度分析 Meso 位结构
        修复了空锚点判定和显式氢原子的归类问题
        """
        if not scaffold_info: return None
        
        meso_idx = scaffold_info['meso_idx']
        meso_atom = mol.GetAtomWithIdx(meso_idx)
        core_indices = scaffold_info['all_core_indices']
        
        # 1. 寻找锚点原子 (Anchor Atom)
        anchor_atom = None
        for nbr in meso_atom.GetNeighbors():
            if nbr.GetIdx() not in core_indices:
                anchor_atom = nbr
                break
        
        # === 情况 A: 异常 - 没有找到锚点原子 ===
        # 在全显式氢模型下，如果找不到邻居，说明结构缺失或异常，不能默认为 H
        if anchor_atom is None:
            return {
                "exists": False,
                "connection_type": ConnectionType.UNKNOWN.value, # 修正：标记为未知
                "hybridization": HybridizationType.UNKNOWN.value,
                "is_conjugated": False,
                "dihedral_angle": None,
                "anchor_idx": None
            }
            
        anchor_idx = anchor_atom.GetIdx()
        atomic_num = anchor_atom.GetAtomicNum()

        # === 情况 B: 显式氢原子 (Meso-H) ===
        # 必须在判断杂原子之前先排除 H
        if atomic_num == 1:
            return {
                "exists": True,
                "connection_type": ConnectionType.HYDROGEN.value, # 正确归类
                "hybridization": HybridizationType.S.value,
                "is_conjugated": False,
                "dihedral_angle": None,
                "anchor_idx": anchor_idx
            }
        
        # === 情况 C: 真正的杂原子 (N, O, S, F, Cl...) ===
        if atomic_num != 6:
            return {
                "exists": True,
                "connection_type": ConnectionType.HETEROATOM.value,
                "element": anchor_atom.GetSymbol(),
                "hybridization": HybridizationType.UNKNOWN.value, 
                "is_conjugated": True, 
                "dihedral_angle": None, 
                "anchor_idx": anchor_idx
            }

        # === 情况 D: 碳原子 (需要区分 Alkyl vs Aryl/Carbonyl) ===
        degree = anchor_atom.GetDegree()
        
        conn_type = ConnectionType.UNKNOWN
        hyb_type = HybridizationType.UNKNOWN
        calc_dihedral = False
        
        if degree == 4:
            # sp3 -> 烷基
            conn_type = ConnectionType.ALKYL
            hyb_type = HybridizationType.SP3
            calc_dihedral = False
            
        elif degree == 3:
            # sp2 -> 芳基 或 共轭链
            if anchor_atom.IsInRing():
                conn_type = ConnectionType.ARYL
            else:
                conn_type = ConnectionType.CONJUGATED
            hyb_type = HybridizationType.SP2
            calc_dihedral = True
            
        elif degree == 2:
            # sp -> 炔基
            conn_type = ConnectionType.ALKYNYL
            hyb_type = HybridizationType.SP
            calc_dihedral = False
            
        # 计算二面角
        angle_val = None
        if calc_dihedral:
            angle_val = self._compute_dihedral_value(mol, scaffold_info, anchor_idx)
            
        return {
            "exists": True,
            "connection_type": conn_type.value,
            "hybridization": hyb_type.value,
            "is_conjugated": calc_dihedral,
            "dihedral_angle": angle_val,
            "anchor_idx": anchor_idx
        }

    def _compute_dihedral_value(self, mol, scaffold_info, anchor_idx):
        """内部工具：计算几何角度"""
        try:
            # 4个原子: Core_Ref -> Meso -> Anchor -> Anchor_Ref
            idx_meso = scaffold_info['meso_idx']
            idx_core_ref = scaffold_info['alpha_indices'][0]
            
            # 寻找 Anchor 的参考邻居 (重原子优先)
            anchor_atom = mol.GetAtomWithIdx(anchor_idx)
            idx_anchor_ref = None
            for nbr in anchor_atom.GetNeighbors():
                if nbr.GetIdx() != idx_meso and nbr.GetAtomicNum() > 1:
                    idx_anchor_ref = nbr.GetIdx()
                    break
            
            # 如果没有重原子邻居 (比如甲醛基 -CH=O 的 H，或者奇怪的结构)，降级用 H
            if idx_anchor_ref is None:
                for nbr in anchor_atom.GetNeighbors():
                    if nbr.GetIdx() != idx_meso:
                        idx_anchor_ref = nbr.GetIdx()
                        break
            
            if idx_anchor_ref is None: return None
            
            conf = mol.GetConformer()
            angle = rdMolTransforms.GetDihedralDeg(conf, idx_core_ref, idx_meso, anchor_idx, idx_anchor_ref)
            angle = abs(angle)
            while angle > 90:
                angle = abs(180 - angle)
            return round(angle, 1)
        except:
            return None