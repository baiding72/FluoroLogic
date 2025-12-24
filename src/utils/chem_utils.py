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
        # 核心模式：匹配连接两个吡咯单元的 Meso 碳
        # Index 0: Meso Carbon
        # 左臂: ~C~C~C~N
        # 右臂: ~C~C~C~N
        # 使用 ~ (任意键) 和 #6 (原子序数) 以最大化兼容性
        self.core_smarts = Chem.MolFromSmarts("[#6](~[#6]~[#6]~[#6]~[#7])(~[#6]~[#6]~[#6]~[#7])")

    def analyze(self, mol):
        """
        分析分子，返回骨架上的关键原子索引字典。
        """
        if not mol or not self.core_smarts:
            return None

        # 1. 匹配骨架
        try:
            matches = mol.GetSubstructMatches(self.core_smarts)
        except:
            return None
            
        if not matches:
            # 识别失败
            smiles = Chem.MolToSmiles(mol)
            print(f"  SMILES: {smiles}")
            try:
                mol = Chem.MolFromSmiles(smiles)
                matches = mol.GetSubstructMatches(self.core_smarts)
            except:
                return None
                
            return None
        
        # 取第一个匹配结果
        match = matches[0]
        
        # 2. 解析索引 (根据 SMARTS 的定义顺序)
        # SMARTS: [Meso](~[A1]~[B1]~[B2]~[N1])(~[A2]~[B3]~[B4]~[N2])
        # Indices: 0     1    2    3    4      5    6    7    8
        
        return {
            "meso_idx": match[0],
            "alpha_indices": [match[1], match[5]], # 连接 Meso 的 alpha 位
            "beta_indices": [match[2], match[3], match[6], match[7]], # 远端 beta 位
            "nitrogen_indices": [match[4], match[8]],
            "all_core_indices": set(match)
        }

    def get_meso_substituent_atom(self, mol, scaffold_info):
        """
        获取 Meso 位连接的取代基原子索引 (Pivot Atom)。
        """
        if not scaffold_info: return None
        
        meso_atom = mol.GetAtomWithIdx(scaffold_info['meso_idx'])
        core_indices = scaffold_info['all_core_indices']
        
        # 遍历 Meso 碳的邻居
        for nbr in meso_atom.GetNeighbors():
            # 如果邻居不在骨架定义中，它就是取代基！
            if nbr.GetIdx() not in core_indices:
                return nbr.GetIdx()
                
        return None # 可能是 Meso-H

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
            return None # Meso-H 无二面角
            
        # 2. 锁定 Core 参考点 (取任意一个 Alpha 碳)
        idx_core_ref = scaffold_info['alpha_indices'][0]
        
        # 3. 锁定 Subst 参考点 (找一个重原子邻居)
        atom_subst = mol.GetAtomWithIdx(idx_subst)
        idx_subst_ref = None
        
        # 优先找碳或杂原子 (AtomicNum > 1)，避开氢
        for nbr in atom_subst.GetNeighbors():
            if nbr.GetIdx() != idx_meso and nbr.GetAtomicNum() > 1:
                idx_subst_ref = nbr.GetIdx()
                break
        
        # 如果取代基只有氢邻居（比如甲基 -CH3），二面角通常无物理意义(旋转极快)
        # 但为了数据完整性，我们可以返回 None，或者如果需要的话也可以选一个 H
        if idx_subst_ref is None:
            return None 
            
        return (idx_core_ref, idx_meso, idx_subst, idx_subst_ref)