import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

def calculate_critical_dihedral(mol_3d, core_smarts="[...]"): # 填入BODIPY骨架SMARTS
    """
    输入：带有3D构象的RDKit Mol对象 (从Gaussian output转换而来)
    输出：关键二面角 (度数)、骨架平面性
    """
    # 1. 定义子结构：找到 meso-位连接键
    # 假设 meso-位是 C8，连接一个苯环
    # 这里需要写一段精妙的 SMARTS 来匹配 "BODIPY核心 - meso键 - 苯环"
    
    # 2. 获取关键的 4 个原子索引 (a, b, c, d)
    # a, b 在 BODIPY 环上
    # c, d 在 meso-苯环上
    # matches = mol_3d.GetSubstructMatches(...)
    # a, b, c, d = matches[0]...
    
    # 3. 从 3D 构象中计算角度
    conf = mol_3d.GetConformer()
    angle_rad = rdMolTransforms.GetDihedralRad(conf, a, b, c, d)
    angle_deg = np.degrees(angle_rad)
    
    # 取绝对值并归一化到 0-90 度 (因为对于共轭破坏，-45度和+45度是一样的，90度破坏最强)
    angle_deg = abs(angle_deg)
    if angle_deg > 90:
        angle_deg = 180 - angle_deg
        
    return angle_deg

def analyze_planarity(mol_3d):
    """
    计算 BODIPY 核心骨架 9 个原子的平面偏差 RMSD
    """
    # 提取核心原子坐标，拟合一个最佳平面，计算所有点到平面的距离标准差
    # 如果 RMSD 很大，说明骨架弯曲了 (Buckling)，这也是影响电位的重要因素
    pass