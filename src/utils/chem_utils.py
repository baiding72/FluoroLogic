import sys
import re
from enum import Enum
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

# ================= 枚举定义 =================

class HybridizationType(str, Enum):
    """原子杂化方式枚举"""
    SP3 = "sp3"         
    SP2 = "sp2"         
    SP = "sp"           
    S = "s"             
    UNKNOWN = "unknown" 

class ConnectionType(str, Enum):
    """Meso 位连接基团类型枚举"""
    HYDROGEN = "hydrogen"       # 氢原子 
    ALKYL = "alkyl"             # 烃基
    ARYL = "aryl"               # 芳香基
    CONJUGATED = "conjugated"   # 共轭基团
    ALKYNYL = "alkynyl"         # 炔基
    HETEROATOM = "heteroatom"   # 杂原子基团
    UNKNOWN = "unknown"         # 未知

class BoronType(str, Enum):
    """Boron 连接基团类型枚举"""
    STANDARD_BF2 = "Standard_BF2"  # 标准BF2结构
    MODIFIED = "Modified"          # 其他
    UNKNOWN = "Unknown"            # 未知


# ===========================================

class BodipyScaffoldMatcher:
    """
    BODIPY 骨架与取代基分析器 (全能版)
    """
    def __init__(self):
        # 核心 SMARTS: Meso(0) 连接两个吡咯环
        # 索引定义 (用于 ReplaceCore 归类):
        # 0: Meso
        # 1, 6: Alpha
        # 2, 3, 7, 8: Beta
        # 5, 10: N
        self.smarts_str = "[#6](~[#6]1~[#6]~[#6]~[#6]~[#7]1)(~[#6]1~[#6]~[#6]~[#6]~[#7]1)"
        self.core_smarts = Chem.MolFromSmarts(self.smarts_str)

    def analyze(self, mol):
        """
        骨架识别与 N-B-N 验证
        """
        if not mol or not self.core_smarts: return False, None
        try:
            matches = mol.GetSubstructMatches(self.core_smarts)
        except:
            return False, None
        if not matches: return False, None

        # N-B-N 闭环验证
        valid_match = None
        for match in matches:
            n1_idx = match[5]
            n2_idx = match[10]
            n1_atom = mol.GetAtomWithIdx(n1_idx)
            n2_atom = mol.GetAtomWithIdx(n2_idx)
            
            common_nbrs = set([x.GetIdx() for x in n1_atom.GetNeighbors()]).intersection(
                          set([x.GetIdx() for x in n2_atom.GetNeighbors()]))
            
            common_boron = None
            for idx in common_nbrs:
                if mol.GetAtomWithIdx(idx).GetAtomicNum() == 5: # Boron
                    common_boron = idx
                    break
            
            if common_boron is not None:
                valid_match = match
                break
        
        if valid_match is None:
            return False, {"error": "no_N_B_N_bridge"}

        match = valid_match
        scaffold_info = {
            "meso_idx": match[0],
            "alpha_indices": [match[1], match[6]], 
            "beta_indices": [match[2], match[3], match[7], match[8]], 
            "nitrogen_indices": [match[5], match[10]],
            "boron_idx": common_boron,
            "all_core_indices": set(match)
        }
        return True, scaffold_info

    def analyze_meso_structure(self, mol, scaffold_info):
        """深度分析 Meso 位物理结构 (杂化、二面角)"""
        if not scaffold_info: return None
        
        meso_idx = scaffold_info['meso_idx']
        meso_atom = mol.GetAtomWithIdx(meso_idx)
        core_indices = scaffold_info['all_core_indices']
        
        anchor_atom = None
        for nbr in meso_atom.GetNeighbors():
            if nbr.GetIdx() not in core_indices:
                anchor_atom = nbr
                break
        
        if anchor_atom is None:
            return {
                "exists": False,
                "connection_type": ConnectionType.UNKNOWN.value,
                "hybridization": HybridizationType.UNKNOWN.value,
                "is_conjugated": False,
                "dihedral_angle": None,
                "anchor_idx": None
            }
            
        anchor_idx = anchor_atom.GetIdx()
        atomic_num = anchor_atom.GetAtomicNum()

        if atomic_num == 1:
            return {
                "exists": True,
                "connection_type": ConnectionType.HYDROGEN.value,
                "hybridization": HybridizationType.S.value,
                "is_conjugated": False,
                "dihedral_angle": None,
                "anchor_idx": anchor_idx
            }
        
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

        degree = anchor_atom.GetDegree()
        conn_type = ConnectionType.UNKNOWN
        hyb_type = HybridizationType.UNKNOWN
        calc_dihedral = False
        
        if degree == 4:
            conn_type = ConnectionType.ALKYL
            hyb_type = HybridizationType.SP3
        elif degree == 3:
            if anchor_atom.IsInRing(): conn_type = ConnectionType.ARYL
            else: conn_type = ConnectionType.CONJUGATED
            hyb_type = HybridizationType.SP2
            calc_dihedral = True
        elif degree == 2:
            conn_type = ConnectionType.ALKYNYL
            hyb_type = HybridizationType.SP

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
        try:
            idx_meso = scaffold_info['meso_idx']
            idx_core_ref = scaffold_info['alpha_indices'][0]
            anchor_atom = mol.GetAtomWithIdx(anchor_idx)
            idx_anchor_ref = None
            for nbr in anchor_atom.GetNeighbors():
                if nbr.GetIdx() != idx_meso and nbr.GetAtomicNum() > 1:
                    idx_anchor_ref = nbr.GetIdx()
                    break
            if idx_anchor_ref is None: # 降级用H
                for nbr in anchor_atom.GetNeighbors():
                    if nbr.GetIdx() != idx_meso:
                        idx_anchor_ref = nbr.GetIdx()
                        break
            if idx_anchor_ref is None: return None
            
            conf = mol.GetConformer()
            angle = rdMolTransforms.GetDihedralDeg(conf, idx_core_ref, idx_meso, anchor_idx, idx_anchor_ref)
            angle = abs(angle)
            while angle > 90: angle = abs(180 - angle)
            return round(angle, 1)
        except:
            return None

    def extract_substituents(self, mol):
        """
        利用 Chem.ReplaceCore 一次性提取所有位点的取代基 SMILES
        """
        if not mol or not self.core_smarts: return None

        # 1. 切除母核
        try:
            side_chains = Chem.ReplaceCore(mol, self.core_smarts, labelByIndex=True)
        except:
            return None
            
        if not side_chains:
            return {"meso": "[H]", "alpha": [], "beta": [], "boron_fragment": None}

        # 2. 分离碎片
        frags = Chem.GetMolFrags(side_chains, asMols=True, sanitizeFrags=False)
        
        results = {
            "meso": "[H]", 
            "meso_flanking": [], # 新增：1,7 位 (靠近 Meso)
            "alpha": [],         # 3,5 位 (靠近 N)
            "beta": [],          # 2,6 位 (中间)
            "boron_fragment": None
        }
        
        for frag in frags:
            # 获取 SMILES，此时连接点是带编号的 dummy atom，如 [1*]
            raw_smiles = Chem.MolToSmiles(frag, canonical=True)
            
            # 标准化 SMILES: 将 [1*], [10*] 等统一替换为 *，方便 Hammett 查表
            clean_smiles = re.sub(r'\[\d+\*\]', '*', raw_smiles)

            # 确定该碎片连接的位置 (根据 Dummy Atom 的 Isotope)
            attachment_point = -1
            for atom in frag.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    attachment_point=atom.GetIsotope()
                    break

            if attachment_point < 0: continue

            
            if attachment_point == 0:
                results["meso"] = clean_smiles
            elif attachment_point in [2, 7]:
                results["meso_flanking"].append(clean_smiles)
            elif attachment_point in [4, 9]:
                results["alpha"].append(clean_smiles)
            elif attachment_point in [3, 8]:
                results["beta"].append(clean_smiles)
            elif attachment_point in [5, 10]:
                # === 针对 Boron 碎片的特殊处理 ===
                    # 标准 BF2 的 SMILES 通常含有 B 和 F
                    # RDKit 生成的 clean_smiles 可能是 "FB(F)" 或 "*B(*)(F)F" 等变体
                    # 我们这里不做过度清洗，原样返回，交由后续分析
                    results["boron_fragment"] = clean_smiles
            elif attachment_point in [1, 6]:
                # 非预期的连接点
                print(f"骨架碳上有异常取代基: {attachment_point}，请检查！")
                return None
        
        # === 新增：归一化 Boron 状态 ===
        # 检查是否为标准 BF2
        b_frag = results["boron_fragment"]
        if b_frag:
            # 简单启发式：如果有 B 且 F 的数量 >= 2，且没有 C, N, O 等重原子
            # (注意：SMILES 里的 * 不算重原子)
            has_b = 'B' in b_frag
            f_count = b_frag.count('F')
            has_other_heavy = any(c in b_frag for c in ['C', 'c', 'N', 'n', 'O', 'o'])
            
            if has_b and f_count == 2 and not has_other_heavy:
                # 标记为标准类型，方便人类阅读，也方便 Agent 快速理解
                results["boron_type"] = BoronType.STANDARD_BF2.value
            else:
                results["boron_type"] = BoronType.MODIFIED.value
        else:
             results["boron_type"] = BoronType.NONE.value
        
        return results