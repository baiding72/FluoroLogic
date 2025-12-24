import sys
from rdkit import Chem
from rdkit.Chem import Draw

class BodipyScaffoldMatcher:
    """
    [Debug Version] BODIPY 骨架语义识别器
    更新了 SMARTS 以支持显式的 5 元环定义
    """
    def __init__(self):
        # === 核心修复 ===
        # 旧模式 (线性): "[#6](~[#6]~[#6]~[#6]~[#7])(~[#6]~[#6]~[#6]~[#7])" -> 失败
        # 新模式 (环状): Meso碳 连接两个 5元环 (C-C-C-C-N)
        # SMARTS 解释: 
        # [#6]       : Meso 碳 (Index 0)
        # (~[#6]1...) : 连接到左边的环 (Atom 1..5)
        # (~[#6]1...) : 连接到右边的环 (Atom 6..10)
        # 环定义: ~[#6]1~[#6]~[#6]~[#6]~[#7]1 (C-C-C-C-N 闭环)
        self.smarts_str = "[#6](~[#6]1~[#6]~[#6]~[#6]~[#7]1)(~[#6]1~[#6]~[#6]~[#6]~[#7]1)"
        self.core_smarts = Chem.MolFromSmarts(self.smarts_str)

    def analyze(self, mol):
        if not mol or not self.core_smarts:
            print("  [Error] Mol object is None or SMARTS failed to compile.")
            return None

        # 1. 匹配骨架
        matches = mol.GetSubstructMatches(self.core_smarts)
        
        if not matches:
            print(f"  [Fail] No substructure match found for SMARTS.")
            return None
        
        print(f"  [Success] Found {len(matches)} matches for core skeleton.")
        
        # 取第一个匹配结果
        match = matches[0]
        # print(f"  [Info] Match indices (Raw): {match}")
        
        # === 索引映射更新 ===
        # 新 SMARTS 有 11 个原子:
        # 0: Meso
        # Branch 1: 1(Alpha), 2(Beta), 3(Beta), 4(Alpha'), 5(N)
        # Branch 2: 6(Alpha), 7(Beta), 8(Beta), 9(Alpha'), 10(N)
        
        return {
            "meso_idx": match[0],
            "alpha_indices": [match[1], match[6]], # 直接连 Meso 的 Alpha 位
            "nitrogen_indices": [match[5], match[10]], # 氮原子
            "all_core_indices": set(match)
        }

    def get_meso_substituent_atom(self, mol, scaffold_info):
        """寻找 Meso 位连接的取代基原子"""
        if not scaffold_info: return None
        
        meso_idx = scaffold_info['meso_idx']
        meso_atom = mol.GetAtomWithIdx(meso_idx)
        core_indices = scaffold_info['all_core_indices']
        
        print(f"  [Info] Analyzing neighbors of Meso-Carbon (Idx {meso_idx})...")
        
        for nbr in meso_atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            is_core = nbr_idx in core_indices
            symbol = nbr.GetSymbol()
            print(f"    -> Neighbor {nbr_idx} ({symbol}): In Core? {is_core}")
            
            if not is_core:
                return nbr_idx
        
        print("  [Info] No non-core neighbor found (Likely Meso-H).")
        return None

    def get_dihedral_atoms(self, mol, scaffold_info):
        """获取测角所需的 4 个原子"""
        # 1. 锁定取代基连接点
        idx_subst = self.get_meso_substituent_atom(mol, scaffold_info)
        if idx_subst is None:
            print("  [Fail] Cannot determine dihedral: No Meso substituent.")
            return None
            
        # 2. 锁定 Core 参考点 (取任意一个 Alpha 碳)
        idx_core_ref = scaffold_info['alpha_indices'][0]
        
        # 3. 锁定 Subst 参考点 (找一个重原子邻居)
        atom_subst = mol.GetAtomWithIdx(idx_subst)
        idx_subst_ref = None
        
        print(f"  [Info] Analyzing neighbors of Substituent (Idx {idx_subst})...")
        idx_meso = scaffold_info['meso_idx']
        
        for nbr in atom_subst.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            atomic_num = nbr.GetAtomicNum()
            print(f"    -> Neighbor {nbr_idx} ({nbr.GetSymbol()}): AtomicNum={atomic_num}")
            
            # 找一个非 Meso 的重原子邻居 (排除 H)
            if nbr_idx != idx_meso and atomic_num > 1:
                idx_subst_ref = nbr_idx
                print(f"      -> Selected as reference (Heavy Atom).")
                break
        
        if idx_subst_ref is None:
            print("  [Fail] Cannot determine dihedral: Substituent has no heavy atom neighbors (Maybe Methyl/H).")
            return None
            
        return (idx_core_ref, idx_meso, idx_subst, idx_subst_ref)

# ================= 交互式测试 =================

def debug_smiles(smiles):
    print("\n" + "="*50)
    print(f"🧪 Testing Molecule")
    print("="*50)
    
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        print("❌ Invalid SMILES string. RDKit could not parse it.")
        return

    for atom in mol.GetAtoms():
        atom.SetProp('molAtomMapNumber', str(atom.GetIdx()))
    img = Draw.MolToImage(mol, size=(500, 500))
    img.save("molecule_with_index.png")

    matcher = BodipyScaffoldMatcher()
    
    # 步骤 1: 骨架匹配
    print("\n--- Step 1: Core Matching ---")
    scaffold = matcher.analyze(mol)
    
    if not scaffold:
        print("❌ Core matching failed.")
        return
    else:
        print(f"✅ Core identified. Meso Atom Index: {scaffold['meso_idx']}")

    # 步骤 2: 二面角原子提取
    print("\n--- Step 2: Dihedral Atom Selection ---")
    atoms = matcher.get_dihedral_atoms(mol, scaffold)
    
    if atoms:
        print(f"✅ Dihedral Atoms Found: {atoms}")
        print(f"   Order: Core_Ref -> Meso -> Subst -> Subst_Ref")
    else:
        print("⚠️ Could not define dihedral angle atoms.")

if __name__ == "__main__":
    print("请输入出错分子的 SMILES (按回车确认):")
    # 你的出错分子:
    # [B-]1([N+]2=C(C=C(C2=C(C(=O)Cc2c([N+](=O)[O-])cc(cc2)[N+](=O)[O-])c2n1c(cc2C)/C=C/c1ccc(OCCOCCOCCOC)cc1)C)/C=C/c1ccc(OCCOCCOCCOC)cc1)(F)F
    
    user_input = "[N-]=[N+]=NCCOc1ccc(/C=C/c2ccc3n2[B-](F)(F)[n+]2c4n(c5ccccc52)[B-](F)(F)[N+]2=CC=CC2=C34)cc1"
    if user_input:
        debug_smiles(user_input)
    