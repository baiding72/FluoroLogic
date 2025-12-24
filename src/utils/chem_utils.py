import sys
from rdkit import Chem

class BodipyScaffoldMatcher:
    """
    BODIPY 骨架语义识别器
    功能：
    1. 基于拓扑 SMARTS 识别 BODIPY 核心骨架 (不依赖 RDKit 的 Sanitization/芳香性)。
    2. 语义化提取关键位点索引 (Meso, Alpha, Nitrogen)。
    3. 辅助识别取代基连接原子。
    """
    def __init__(self):
        # 核心模式 (环状): Meso碳 连接两个 5元环 (C-C-C-C-N)
        # 这种模式比线性的更稳健，能匹配到闭合的吡咯环
        self.smarts_str = "[#6](~[#6]1~[#6]~[#6]~[#6]~[#7]1)(~[#6]1~[#6]~[#6]~[#6]~[#7]1)"
        self.core_smarts = Chem.MolFromSmarts(self.smarts_str)

    def analyze(self, mol):
        """
        分析分子，返回 (是否匹配成功, 骨架信息字典)。
        Returns:
            tuple: (is_match (bool), scaffold_info (dict|None))
        """
        if not mol or not self.core_smarts:
            return False, None

        # 1. 匹配骨架
        try:
            matches = mol.GetSubstructMatches(self.core_smarts)
        except:
            return False, None
            
        if not matches:
            return False, None
        
        # 取第一个匹配结果
        match = matches[0]
        
        # 2. 解析索引 (根据 SMARTS 的定义顺序)
        # SMARTS 有 11 个原子:
        # 0: Meso
        # Branch 1: 1(Alpha), 2(Beta), 3(Beta), 4(Alpha'), 5(N)
        # Branch 2: 6(Alpha), 7(Beta), 8(Beta), 9(Alpha'), 10(N)
        
        scaffold_info = {
            "meso_idx": match[0],
            "alpha_indices": [match[1], match[6]], # 直接连 Meso 的 Alpha 位
            "beta_indices": [match[2], match[3], match[7], match[8]], 
            "nitrogen_indices": [match[5], match[10]],
            "all_core_indices": set(match)
        }
        
        return True, scaffold_info

    def get_meso_substituent_atom(self, mol, scaffold_info):
        """获取 Meso 位连接的取代基原子索引 (Pivot Atom)"""
        if not scaffold_info: return None
        
        meso_idx = scaffold_info['meso_idx']
        meso_atom = mol.GetAtomWithIdx(meso_idx)
        core_indices = scaffold_info['all_core_indices']
        
        for nbr in meso_atom.GetNeighbors():
            if nbr.GetIdx() not in core_indices:
                return nbr.GetIdx()
        return None 

    def get_dihedral_atoms(self, mol, scaffold_info):
        """
        智能获取用于计算 Meso 二面角的 4 个原子索引。
        返回 tuple: (Core_Ref, Meso_C, Subst_C, Subst_Ref)
        """
        if not scaffold_info: return None
        
        # 1. 锁定中间轴
        idx_meso = scaffold_info['meso_idx']
        idx_subst = self.get_meso_substituent_atom(mol, scaffold_info)
        
        if idx_subst is None:
            return None 
            
        # 2. 锁定 Core 参考点 (取任意一个 Alpha 碳)
        idx_core_ref = scaffold_info['alpha_indices'][0]
        
        # 3. 锁定 Subst 参考点 (找一个重原子邻居)
        atom_subst = mol.GetAtomWithIdx(idx_subst)
        idx_subst_ref = None
        
        for nbr in atom_subst.GetNeighbors():
            if nbr.GetIdx() != idx_meso and nbr.GetAtomicNum() > 1:
                idx_subst_ref = nbr.GetIdx()
                break
        
        if idx_subst_ref is None:
            return None 
            
        return (idx_core_ref, idx_meso, idx_subst, idx_subst_ref)