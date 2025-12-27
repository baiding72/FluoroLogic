import sys
import re
from enum import Enum
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
import numpy as np

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

        # 建立 索引 -> 类型 的映射表
        # 根据 SMARTS: 
        # Meso=0, Alpha=1,6 (Core), Meso-Flanking=2,7, Beta=3,8, Alpha(N)=4,9
        
        type_map = {}
        type_map[match[0]] = "meso"
        # type_map[match[1]] = "core_alpha_link"
        # type_map[match[6]] = "core_alpha_link"
        type_map[match[2]] = "meso_flanking"
        type_map[match[7]] = "meso_flanking"
        type_map[match[3]] = "beta"
        type_map[match[8]] = "beta"
        type_map[match[4]] = "alpha"
        type_map[match[9]] = "alpha"
        type_map[match[5]] = "nitrogen"
        type_map[match[10]] = "nitrogen"

        scaffold_info = {
            "meso_idx": match[0],
            "alpha_idx": [match[4], match[9]],
            "meso_flanking_idx": [match[2], match[7]],
            "beta_idx": [match[3], match[8]], 
            "nitrogen_idx": [match[5], match[10]],
            "boron_idx": common_boron,
            "type_map": type_map, # <--- 新增这个 map，传给 extract
            # 顺便把细分索引也存一下，方便 analyzer 使用
            "indices_by_type": {
                "meso": [match[0]],
                "meso_flanking": [match[2], match[7]],
                "beta": [match[3], match[8]],
                "alpha": [match[4], match[9]]
            },
            "all_core_idx": set(match)
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

    def compute_site_dihedral(self, mol, root_idx, core_neighbor_idx):
        """
        计算任意取代基的二面角
        Path: [Core_Ref] - [Core_Root] - [Subst_Atom] - [Subst_Ref]
        """
        # 1. Core Root: core_neighbor_idx (骨架上的连接点，如 Beta 碳)
        # 2. Subst Atom: root_idx (取代基上的第一个原子)
        
        # 寻找 Core Reference (骨架上相邻的另一个碳)
        core_atom = mol.GetAtomWithIdx(core_neighbor_idx)
        core_ref = None
        for nbr in core_atom.GetNeighbors():
            if nbr.IsInRing(): # 骨架上的参考点必须在环上
                core_ref = nbr.GetIdx()
                break
        if core_ref is None: return None
        
        # 寻找 Subst Reference (取代基上的重原子)
        subst_atom = mol.GetAtomWithIdx(root_idx)
        subst_ref = None
        for nbr in subst_atom.GetNeighbors():
            if nbr.GetIdx() != core_neighbor_idx and nbr.GetAtomicNum() > 1:
                subst_ref = nbr.GetIdx()
                break
        
        if subst_ref is None: return None # 可能是甲基或 H，不计算二面角
        
        try:
            conf = mol.GetConformer()
            angle = rdMolTransforms.GetDihedralDeg(conf, core_ref, core_neighbor_idx, root_idx, subst_ref)
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
    # src/utils/chem_utils.py

    # src/utils/chem_utils.py

    def extract_substituents_detailed(self, mol, scaffold_info):
        """
        基于图遍历提取取代基信息 (SMILES + 原始索引)
        """
        if not mol or not scaffold_info: return {}

        core_indices = scaffold_info['all_core_indices']
        type_map = scaffold_info['type_map']
        results = {}

        # 遍历骨架上的每一个原子，看它外面连了什么
        for core_idx in core_indices:
            # 只处理我们在 type_map 里定义的感兴趣的位点
            # (跳过 index 1, 6 这种骨架内部连接点)
            site_type = type_map.get(core_idx)
            if not site_type: continue
            
            core_atom = mol.GetAtomWithIdx(core_idx)
            
            for nbr in core_atom.GetNeighbors():
                nbr_idx = nbr.GetIdx()
                
                # 如果邻居也是骨架原子，忽略
                if nbr_idx in core_indices: continue
                
                # === 发现取代基 (Root: nbr_idx) ===
                
                # 1. 也是最重要的一步：获取取代基的所有原始原子索引
                subst_indices = self._get_substructure_indices(mol, nbr_idx, core_indices)
                
                # 2. 生成 SMILES (方便 Hammett 查表)
                # 使用 MolFragmentToSmiles，它不会改变原始分子的索引，只是提取子图生成字符串
                subst_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=subst_indices, canonical=True)
                
                # 3. 归类
                # 如果是 Boron Center (N)，它连出来的除了 B 还有别的吗？
                # N 连着 B。所以 nbr_idx 应该是 B 的索引。
                # 我们这里要做个特殊判断：如果 type 是 boron_center，我们只关心那个 B 原子及其挂件
                
                final_type = site_type
                if site_type == "boron_center":
                    if mol.GetAtomWithIdx(nbr_idx).GetAtomicNum() == 5: # 也就是 B
                        final_type = "boron_fragment"
                    else:
                        continue # N 上连的其他非骨架非B原子？极其罕见，忽略

                # 存入结果
                if core_idx not in results:
                    results[core_idx] = []
                
                results[core_idx].append({
                    "type": final_type,
                    "smiles": subst_smiles,
                    "atom_indices": subst_indices, # [25, 26, 27...] 全局索引保住了！
                    "root_idx": nbr_idx
                })

        return results

    def _get_substructure_indices(self, mol, root_idx, forbidden_indices):
        """BFS 遍历获取子结构索引"""
        indices = []
        visited = set(forbidden_indices) # 骨架是墙
        stack = [root_idx]
        
        while stack:
            curr = stack.pop()
            if curr in visited: continue
            visited.add(curr)
            indices.append(curr) # 记录原始索引
            
            atom = mol.GetAtomWithIdx(curr)
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() not in visited:
                    stack.append(nbr.GetIdx())
        return indices



class BodipyStericAnalyzer:
    """
    BODIPY 空间效应与几何构象分析器
    对应物理路径：
    1. 共轭破坏 (Planarity, Dihedrals)
    2. 轨道局域化 (Symmetry, Dihedrals)
    3. 稳态差异 (RMSD, Proximal Distance)
    """
    def __init__(self, mol, scaffold_info):
        self.mol = mol
        self.scaffold = scaffold_info
        self.conf = mol.GetConformer()
        
        # 预计算核心平面
        self.core_indices = list(scaffold_info['all_core_indices'])
        self.plane_coeffs, self.core_centroid = self._fit_core_plane()

    def _fit_core_plane(self):
        """
        拟合 BODIPY 母核 (C9N2B) 的最佳平面
        使用 SVD 分解计算最小二乘平面: ax + by + cz + d = 0
        """
        coords = []
        for idx in self.core_indices:
            pos = self.conf.GetAtomPosition(idx)
            coords.append(np.array([pos.x, pos.y, pos.z]))
        
        coords = np.array(coords)
        centroid = coords.mean(axis=0)
        
        # SVD 分解求法向量
        # 使得点到平面的距离平方和最小
        u, s, vh = np.linalg.svd(coords - centroid)
        normal = vh[2, :] # 最小奇异值对应的特征向量即为法向量
        
        # 平面方程: normal . (x - centroid) = 0
        # ax + by + cz - (normal . centroid) = 0
        d = -np.dot(normal, centroid)
        
        return (normal[0], normal[1], normal[2], d), centroid

    def _dist_to_plane(self, atom_idx):
        """计算原子到核心平面的垂直距离"""
        pos = self.conf.GetAtomPosition(atom_idx)
        x, y, z = pos.x, pos.y, pos.z
        a, b, c, d = self.plane_coeffs
        
        # 距离公式: |ax + by + cz + d| / sqrt(a^2 + b^2 + c^2)
        # 因为 normal 是归一化的，分母为 1
        dist = abs(a*x + b*y + c*z + d)
        return dist

    def calc_core_rmsd(self):
        """
        指标 1: 核心平面性 (RMSD)
        描述 BODIPY 骨架的扭曲程度
        """
        sq_dists = []
        max_displacement = 0.0
        
        for idx in self.core_indices:
            d = self._dist_to_plane(idx)
            sq_dists.append(d**2)
            if d > max_displacement:
                max_displacement = d
                
        rmsd = np.sqrt(np.mean(sq_dists))
        return {
            "rmsd": round(rmsd, 4),
            "max_out_of_plane": round(max_displacement, 4)
        }

    # src/utils/chem_utils.py -> BodipyStericAnalyzer 类

    def calc_steric_heights(self, substituents_detailed):
        """
        [修正版] 指标 3: 取代基的最大垂直高度 (Steric Height / Bulkiness)
        
        物理含义：
        - Max Distance 越大 -> 取代基越"翘" -> 越容易切断共轭或阻挡溶剂。
        - 如果是 Meso-Phenyl，Max Dist 大说明是垂直构象。
        """
        metrics = {
            "max_height_overall": 0.0,
            "max_height_meso_flanking": 0.0,
            "max_height_beta": 0.0,
            "max_height_alpha": 0.0
        }
        
        # 平面参数: ax + by + cz + d = 0
        a, b, c, d = self.plane_coeffs
        denom = np.sqrt(a*a + b*b + c*c)
        
        for core_idx, subs_list in substituents_detailed.items():
            for sub in subs_list:
                # 忽略 F, OH 等小原子，只看重原子骨架
                # 同样忽略连接在 N 上的 B 碎片
                if sub.get('type') == 'boron_fragment': continue
                
                indices = sub['atom_indices']
                if not indices: continue
                
                # 过滤掉 H 原子，只算重原子
                heavy_indices = [idx for idx in indices if self.mol.GetAtomWithIdx(idx).GetAtomicNum() > 1]
                if not heavy_indices: continue
                
                # 计算该取代基内所有原子的"高度" (到平面的垂直距离)
                local_max_h = 0.0
                
                for idx in heavy_indices:
                    pos = self.conf.GetAtomPosition(idx)
                    dist = abs(a*pos.x + b*pos.y + c*pos.z + d) / denom
                    
                    if dist > local_max_h:
                        local_max_h = dist
                
                # 更新全局最大值
                if local_max_h > metrics["max_height_overall"]:
                    metrics["max_height_overall"] = round(local_max_h, 3)
                
                # 更新特定位点
                s_type = sub.get('type')
                val = round(local_max_h, 3)
                
                if s_type in 'meso_flanking':
                    # 记录最大的那个 flanking 基团高度
                    metrics["max_height_meso_flanking"] = max(metrics["max_height_meso_flanking"], val)
                elif s_type == 'beta':
                    metrics["max_height_beta"] = max(metrics["max_height_beta"], val)
                elif s_type == 'alpha':
                    metrics["max_height_alpha"] = max(metrics["max_height_alpha"], val)

        return metrics

    def calc_symmetry_index(self):
        """
        指标 4: 几何对称性破坏指数
        比较左半球 (Branch 1) 和 右半球 (Branch 2) 的质量分布差异
        """
        # Scaffold Index 回顾:
        # Branch 1 (Left): Alpha=1, Beta=2,3, Alpha'=4, N=5
        # Branch 2 (Right): Alpha=6, Beta=7,8, Alpha'=9, N=10
        left_indices = [1, 2, 3, 4, 5]
        right_indices = [6, 7, 8, 9, 10]
        
        # 获取连接在左/右半球上的所有原子（递归）
        def get_branch_mass(root_indices):
            total_mass = 0.0
            visited = set(self.core_indices) # 骨架视为墙
            stack = list(root_indices)
            
            # 初始这几个骨架原子本身的质量不计入（或者计入也行，对称的）
            # 我们只关心取代基
            
            subst_mass = 0.0
            
            # 重新遍历，这次从 root 的邻居开始
            search_stack = []
            for r in root_indices:
                atom = self.mol.GetAtomWithIdx(r)
                for nbr in atom.GetNeighbors():
                    if nbr.GetIdx() not in visited:
                        search_stack.append(nbr.GetIdx())
                        visited.add(nbr.GetIdx())
            
            while search_stack:
                curr = search_stack.pop()
                atom = self.mol.GetAtomWithIdx(curr)
                subst_mass += atom.GetMass()
                
                for nbr in atom.GetNeighbors():
                    if nbr.GetIdx() not in visited:
                        visited.add(nbr.GetIdx())
                        search_stack.append(nbr.GetIdx())
            return subst_mass

        left_mass = get_branch_mass(left_indices)
        right_mass = get_branch_mass(right_indices)
        
        # 归一化不对称度: |L - R| / (L + R + 1e-6)
        asymmetry = abs(left_mass - right_mass) / (left_mass + right_mass + 0.1)
        
        return {
            "mass_asymmetry": round(asymmetry, 3),
            "left_mass": round(left_mass, 1),
            "right_mass": round(right_mass, 1)
        }