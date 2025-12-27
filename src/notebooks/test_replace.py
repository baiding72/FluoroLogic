from rdkit import Chem
from rdkit.Chem import rdRGroupDecomposition
import copy

def replace_core_with_original_indices(mol, core_pattern, replacement_pattern):
    """
    使用ReplaceCore替换核心结构，同时保留原始原子索引信息
    
    Args:
        mol: 原始分子
        core_pattern: 要替换的核心结构模式
        replacement_pattern: 替换模式
    
    Returns:
        替换后的分子和原始原子索引映射
    """
    # 创建原始分子的副本用于索引追踪
    original_mol = copy.deepcopy(mol)
    
    # 找到核心结构的匹配位置
    core_matches = mol.GetSubstructMatches(core_pattern, uniquify=False)
    
    if not core_matches:
        return mol, {}
    
    # 获取核心结构中的原子索引
    core_atom_indices = set()
    for match in core_matches[0]:  # 取第一个匹配
        core_atom_indices.add(match)
    
    # 获取连接点（非核心结构的原子）
    connection_points = []
    for atom_idx in core_atom_indices:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in core_atom_indices:
                connection_points.append((atom_idx, neighbor_idx))
    
    # 执行核心替换
    try:
        # 使用ReplaceCore进行替换
        new_mol = Chem.ReplaceCore(mol, core_pattern, replacement_pattern)
        
        # 创建索引映射
        original_to_new_mapping = {}
        new_to_original_mapping = {}
        
        # 通过SMILES或结构比对建立原子索引映射
        for i in range(new_mol.GetNumAtoms()):
            new_atom = new_mol.GetAtomWithIdx(i)
            if new_atom.GetAtomicNum() == 0:  # Dummy atom
                continue
            
            # 尝试找到对应的原始原子
            for j in range(original_mol.GetNumAtoms()):
                orig_atom = original_mol.GetAtomWithIdx(j)
                if (new_atom.GetAtomicNum() == orig_atom.GetAtomicNum() and
                    new_atom.GetFormalCharge() == orig_atom.GetFormalCharge() and
                    new_atom.GetDegree() == orig_atom.GetDegree()):
                    original_to_new_mapping[j] = i
                    new_to_original_mapping[i] = j
                    break
        
        return new_mol, new_to_original_mapping
    
    except Exception as e:
        print(f"替换过程中出现错误: {e}")
        return mol, {}

def track_rgroup_decomposition_with_indices(mol, core_pattern):
    """
    使用RGroup分解保留原始索引信息
    """
    original_mol = copy.deepcopy(mol)
    
    # 使用RGroup分解
    rgroup_decomposer = rdRGroupDecomposition.RGroupDecomposition([core_pattern])
    rgroup_decomposer.Add([mol])
    
    results = rgroup_decomposer.Process()
    
    if results == 0:
        print("RGroup分解失败")
        return mol, {}
    
    # 获取分解结果
    rgroups = rgroup_decomposer.GetRGroupsAsDictionaries()
    
    # 创建索引映射
    index_mapping = {}
    for rgroup_id, rgroup_data in rgroups.items():
        rgroup_mol = rgroup_data['RGroup']
        core_mol = rgroup_data['Core']
        
        # 这里可以进一步处理索引映射逻辑
        # 通过比对原子属性建立映射关系
        for i in range(rgroup_mol.GetNumAtoms()):
            r_atom = rgroup_mol.GetAtomWithIdx(i)
            if r_atom.GetAtomicNum() != 0:  # 不是dummy atom
                for j in range(original_mol.GetNumAtoms()):
                    o_atom = original_mol.GetAtomWithIdx(j)
                    if (r_atom.GetAtomicNum() == o_atom.GetAtomicNum() and
                        r_atom.GetFormalCharge() == o_atom.GetFormalCharge()):
                        index_mapping[i] = j
                        break
    
    return rgroup_decomposer, index_mapping

# 示例使用
def example_usage():
    # 创建示例分子
    mol = Chem.MolFromSmiles('CCc1ccccc1')  # 乙基苯
    core_pattern = Chem.MolFromSmarts('c1ccccc1')  # 苯环
    replacement_pattern = Chem.MolFromSmarts('[*]')  # 用dummy atom替换
    
    print("原始分子:", Chem.MolToSmiles(mol))
    print("原子数量:", mol.GetNumAtoms())
    
    # 显示原始分子的原子索引
    print("\n原始分子原子信息:")
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        print(f"原子 {i}: 原子序数 {atom.GetAtomicNum()}, 度数 {atom.GetDegree()}")
    
    # 使用自定义函数进行替换
    new_mol, index_mapping = replace_core_with_original_indices(mol, core_pattern, replacement_pattern)
    
    print(f"\n替换后分子: {Chem.MolToSmiles(new_mol)}")
    print(f"原子数量: {new_mol.GetNumAtoms()}")
    print(f"索引映射: {index_mapping}")
    
    # 显示替换后分子的信息
    print("\n替换后分子原子信息:")
    for i in range(new_mol.GetNumAtoms()):
        atom = new_mol.GetAtomWithIdx(i)
        original_idx = index_mapping.get(i, 'N/A')
        print(f"原子 {i}: 原子序数 {atom.GetAtomicNum()}, 度数 {atom.GetDegree()}, 原始索引: {original_idx}")

# 运行示例
if __name__ == "__main__":
    example_usage()



