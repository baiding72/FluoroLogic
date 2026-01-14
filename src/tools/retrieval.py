"""
因果特征检索模块
支持:
1. 多维度因果特征索引 (电子/空间/重组)
2. Activity Cliff Pair 检测 (结构相似但电位差异大)
3. 基于机理的重排策略
"""

import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# 导入其他工具
try:
    from src.tools.electronic import HammettCalculator
    HAMMETT_AVAILABLE = True
except ImportError:
    HAMMETT_AVAILABLE = False
    HammettCalculator = None

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

DB_PATH = "data/processed/molecules.json"


@dataclass
class CausalFeatures:
    """因果特征索引"""
    mol_id: str
    e_red: float
    
    # 电子效应
    meso_sigma: float = 0.0
    total_sigma: float = 0.0
    
    # 空间效应
    meso_dihedral: float = 0.0
    conjugation_length: int = 0
    
    # 重组模式
    delta_dihedral: float = 0.0
    delta_rmsd: float = 0.0
    reorganization_type: str = "unknown"


@dataclass
class ActivityCliffPair:
    """Activity Cliff 分子对"""
    mol_a: Dict
    mol_b: Dict
    similarity: float
    delta_potential: float
    key_difference: str
    mechanism_hint: str


class CausalFeatureRetriever:
    """因果特征检索器"""
    
    # 阈值
    SIMILARITY_THRESHOLD = 0.6   # 结构相似度阈值
    POTENTIAL_DIFF_THRESHOLD = 0.15  # 电位差阈值 (V)
    
    def __init__(self, db_path: str = DB_PATH):
        self.molecules = self._load_database(db_path)
        self.hammett = HammettCalculator() if HAMMETT_AVAILABLE else None
        self.feature_index = self._build_feature_index()
    
    def _load_database(self, path: str) -> List[Dict]:
        if not os.path.exists(path):
            print(f"Warning: Database not found at {path}")
            return []
        with open(path, 'r') as f:
            return json.load(f)
    
    def _build_feature_index(self) -> List[CausalFeatures]:
        """构建因果特征索引"""
        index = []
        
        for mol in self.molecules:
            if not mol.get('is_bodipy'):
                continue
            
            e_red = mol.get('potential_info', {}).get('dft_potential_csv_V')
            if e_red is None:
                continue
            
            # 提取特征
            states = mol.get('states', {})
            reorg = mol.get('reorganization_metrics', {})
            neu_geom = states.get('neutral', {}).get('geometry', {})
            
            features = CausalFeatures(
                mol_id=mol['id'],
                e_red=e_red,
                meso_dihedral=neu_geom.get('meso_dihedral', 0) or 0,
                conjugation_length=neu_geom.get('conjugation_lengths', {}).get('max_conjugation_length', 0) or 0,
                delta_dihedral=reorg.get('delta_dihedral', 0) or 0,
                delta_rmsd=reorg.get('delta_rmsd', 0) or 0,
                reorganization_type=self._classify_reorganization(reorg.get('delta_dihedral'))
            )
            
            # 计算电子效应 (如果 Hammett 可用)
            if self.hammett:
                features.meso_sigma, features.total_sigma = self._calc_electronic_features(mol)
            
            index.append(features)
        
        return index
    
    def _classify_reorganization(self, delta_dihedral: Optional[float]) -> str:
        """分类重组模式"""
        if delta_dihedral is None:
            return "unknown"
        if delta_dihedral < -20:
            return "flattening"
        elif delta_dihedral > 20:
            return "twisting"
        else:
            return "rigid"
    
    def _calc_electronic_features(self, mol: Dict) -> Tuple[float, float]:
        """计算电子效应特征"""
        subs = mol.get('substituents', {})
        meso_sigma = 0.0
        total_sigma = 0.0
        
        for core_idx, sub_list in subs.items():
            for sub in sub_list:
                smiles = sub.get('smiles', '')
                if smiles and smiles != '[H]':
                    result = self.hammett.get_sigma(smiles, position='para')
                    sigma = result.get('sigma', 0) or 0
                    total_sigma += sigma
                    
                    if sub.get('type') == 'meso':
                        meso_sigma = sigma
        
        return meso_sigma, total_sigma
    
    def _calc_similarity(self, mol_a: Dict, mol_b: Dict) -> float:
        """计算分子结构相似度 (Tanimoto)"""
        if not RDKIT_AVAILABLE:
            return 0.0
        
        try:
            smi_a = mol_a.get('smiles', '')
            smi_b = mol_b.get('smiles', '')
            
            if not smi_a or not smi_b:
                return 0.0
            
            mol_a_rdkit = Chem.MolFromSmiles(smi_a)
            mol_b_rdkit = Chem.MolFromSmiles(smi_b)
            
            if mol_a_rdkit is None or mol_b_rdkit is None:
                return 0.0
            
            fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a_rdkit, 2, nBits=1024)
            fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b_rdkit, 2, nBits=1024)
            
            return DataStructs.TanimotoSimilarity(fp_a, fp_b)
        except:
            return 0.0
    
    def find_activity_cliff_pairs(self, query_mol_id: str = None, top_k: int = 5) -> List[ActivityCliffPair]:
        """
        找到 Activity Cliff Pairs
        结构相似度高，但电位差异大的分子对
        """
        pairs = []
        
        if query_mol_id:
            # 针对特定分子找对照
            query_mol = next((m for m in self.molecules if m['id'] == query_mol_id), None)
            if query_mol:
                pairs = self._find_cliffs_for_molecule(query_mol)
        else:
            # 扫描所有分子对
            pairs = self._find_all_cliffs()
        
        # 按电位差排序
        pairs.sort(key=lambda x: x.delta_potential, reverse=True)
        return pairs[:top_k]
    
    def _find_cliffs_for_molecule(self, query_mol: Dict) -> List[ActivityCliffPair]:
        """为特定分子找 Activity Cliff 对照"""
        pairs = []
        query_e = query_mol.get('potential_info', {}).get('dft_potential_csv_V')
        
        if query_e is None:
            return pairs
        
        for mol in self.molecules:
            if mol['id'] == query_mol['id']:
                continue
            
            mol_e = mol.get('potential_info', {}).get('dft_potential_csv_V')
            if mol_e is None:
                continue
            
            similarity = self._calc_similarity(query_mol, mol)
            delta_e = abs(query_e - mol_e)
            
            if similarity >= self.SIMILARITY_THRESHOLD and delta_e >= self.POTENTIAL_DIFF_THRESHOLD:
                diff, hint = self._analyze_difference(query_mol, mol)
                pairs.append(ActivityCliffPair(
                    mol_a=query_mol,
                    mol_b=mol,
                    similarity=similarity,
                    delta_potential=delta_e,
                    key_difference=diff,
                    mechanism_hint=hint
                ))
        
        return pairs
    
    def _find_all_cliffs(self) -> List[ActivityCliffPair]:
        """扫描所有分子找 Activity Cliff Pairs"""
        pairs = []
        n = len(self.molecules)
        
        for i in range(n):
            mol_a = self.molecules[i]
            e_a = mol_a.get('potential_info', {}).get('dft_potential_csv_V')
            if e_a is None:
                continue
            
            for j in range(i + 1, n):
                mol_b = self.molecules[j]
                e_b = mol_b.get('potential_info', {}).get('dft_potential_csv_V')
                if e_b is None:
                    continue
                
                similarity = self._calc_similarity(mol_a, mol_b)
                delta_e = abs(e_a - e_b)
                
                if similarity >= self.SIMILARITY_THRESHOLD and delta_e >= self.POTENTIAL_DIFF_THRESHOLD:
                    diff, hint = self._analyze_difference(mol_a, mol_b)
                    pairs.append(ActivityCliffPair(
                        mol_a=mol_a,
                        mol_b=mol_b,
                        similarity=similarity,
                        delta_potential=delta_e,
                        key_difference=diff,
                        mechanism_hint=hint
                    ))
        
        return pairs
    
    def _analyze_difference(self, mol_a: Dict, mol_b: Dict) -> Tuple[str, str]:
        """分析两个分子的关键差异"""
        differences = []
        hints = []
        
        # 比较电位
        e_a = mol_a.get('potential_info', {}).get('dft_potential_csv_V', 0)
        e_b = mol_b.get('potential_info', {}).get('dft_potential_csv_V', 0)
        
        # 比较 Meso 二面角
        states_a = mol_a.get('states', {}).get('neutral', {}).get('geometry', {})
        states_b = mol_b.get('states', {}).get('neutral', {}).get('geometry', {})
        
        dih_a = states_a.get('meso_dihedral', 0) or 0
        dih_b = states_b.get('meso_dihedral', 0) or 0
        
        if abs(dih_a - dih_b) > 15:
            differences.append(f"Meso二面角: {dih_a:.0f}° vs {dih_b:.0f}°")
            if dih_a > dih_b and e_a < e_b:
                hints.append("更大的二面角→更负电位 (共轭减弱)")
            elif dih_a < dih_b and e_a > e_b:
                hints.append("更小的二面角→更正电位 (共轭增强)")
        
        # 比较取代基
        subs_a = mol_a.get('substituents', {})
        subs_b = mol_b.get('substituents', {})
        
        # 简化比较 - 比较 meso 位
        meso_a = self._get_meso_substituent(subs_a)
        meso_b = self._get_meso_substituent(subs_b)
        
        if meso_a != meso_b:
            differences.append(f"Meso取代基: {meso_a} vs {meso_b}")
            hints.append("取代基电子效应差异")
        
        diff_str = "; ".join(differences) if differences else "结构差异不明显"
        hint_str = "; ".join(hints) if hints else "需要进一步分析"
        
        return diff_str, hint_str
    
    def _get_meso_substituent(self, subs: Dict) -> str:
        """获取 Meso 位取代基"""
        for core_idx, sub_list in subs.items():
            for sub in sub_list:
                if sub.get('type') == 'meso':
                    return sub.get('smiles', 'unknown')[:20]
        return "H"
    
    def query_by_mechanism(self, mechanism_type: str) -> List[Dict]:
        """按机理类型检索"""
        results = []
        
        for feat in self.feature_index:
            match = False
            
            if mechanism_type == "electron_withdrawing":
                match = feat.meso_sigma > 0.3
            elif mechanism_type == "electron_donating":
                match = feat.meso_sigma < -0.3
            elif mechanism_type == "flattening":
                match = feat.reorganization_type == "flattening"
            elif mechanism_type == "rigid":
                match = feat.reorganization_type == "rigid"
            elif mechanism_type == "high_conjugation":
                match = feat.conjugation_length >= 6
            elif mechanism_type == "low_dihedral":
                match = feat.meso_dihedral < 30
            elif mechanism_type == "high_dihedral":
                match = feat.meso_dihedral > 60
            
            if match:
                mol = next((m for m in self.molecules if m['id'] == feat.mol_id), None)
                if mol:
                    results.append({
                        "id": feat.mol_id,
                        "e_red": feat.e_red,
                        "meso_sigma": feat.meso_sigma,
                        "meso_dihedral": feat.meso_dihedral,
                        "reorg_type": feat.reorganization_type
                    })
        
        return results
    
    def get_contrastive_context(self, query_mol_id: str) -> str:
        """
        生成对比学习 Context
        返回适合 Agent 推理的结构化信息
        """
        pairs = self.find_activity_cliff_pairs(query_mol_id, top_k=1)
        
        if not pairs:
            return f"未找到与 {query_mol_id} 相关的 Activity Cliff 对照案例。"
        
        pair = pairs[0]
        mol_a = pair.mol_a
        mol_b = pair.mol_b
        
        e_a = mol_a.get('potential_info', {}).get('dft_potential_csv_V', 0)
        e_b = mol_b.get('potential_info', {}).get('dft_potential_csv_V', 0)
        
        context = f"""
## Activity Cliff 对比分析

我找到了两个参考案例：
- **案例 A**: {mol_a['id']} (E_red = {e_a:.3f}V)
- **案例 B**: {mol_b['id']} (E_red = {e_b:.3f}V)

**结构相似度**: {pair.similarity:.2f}
**电位差**: {abs(e_a - e_b):.3f}V

**关键差异**: {pair.key_difference}
**机理提示**: {pair.mechanism_hint}

---
请基于此对比，推断这些差异对电位的影响规律。
"""
        return context


# ==================== LangChain Tools ====================

class AdvancedQueryInput(BaseModel):
    query_text: str = Field(description="查询关键词或分子ID")
    filter_type: str = Field(
        default="general", 
        description="过滤器类型: 'general', 'mechanism', 'activity_cliff'"
    )

# 全局检索器实例
_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = CausalFeatureRetriever()
    return _retriever


@tool(args_schema=AdvancedQueryInput)
def query_bodi_database(query_text: str, filter_type: str = "general") -> str:
    """
    多模态检索 BODIPY 知识库。
    
    支持三种检索模式:
    1. 'general': 按分子ID或关键词检索，返回完整分子信息
    2. 'mechanism': 按机理类型检索 (electron_withdrawing, flattening, high_dihedral 等)
    3. 'activity_cliff': 找到 Activity Cliff 对照案例，用于对比分析
    """
    retriever = get_retriever()
    q = query_text.strip()
    
    if filter_type == "activity_cliff":
        # 找 Activity Cliff 对照
        context = retriever.get_contrastive_context(q)
        return context
    
    elif filter_type == "mechanism":
        # 按机理检索
        results = retriever.query_by_mechanism(q)
        if not results:
            return f"未找到符合机理类型 '{q}' 的分子。"
        
        output = f"找到 {len(results)} 个符合 '{q}' 机理的分子:\n"
        for r in results[:5]:
            output += f"- {r['id']}: E={r['e_red']:.2f}V, σ_meso={r['meso_sigma']:.2f}, 二面角={r['meso_dihedral']:.0f}°\n"
        return output
    
    else:
        # 通用检索 - 返回完整信息
        for mol in retriever.molecules:
            if q.lower() in mol['id'].lower():
                # 提取所有关键信息
                pot = mol.get('potential_info', {})
                states = mol.get('states', {})
                neu = states.get('neutral', {})
                red = states.get('reduced', {})
                neu_geom = neu.get('geometry', {})
                red_geom = red.get('geometry', {})
                reorg = mol.get('reorganization_metrics', {})
                subs = mol.get('substituents', {})
                
                # 构建详细报告
                output = f"""
## 分子: {mol['id']}

### 基本信息
- **SMILES**: `{mol.get('smiles', 'N/A')[:80]}...`
- **是否BODIPY**: {mol.get('is_bodipy', False)}

### 还原电位
- **E_red (DFT)**: {pot.get('dft_potential_csv_V', 'N/A')} V
- **E_red (实验)**: {pot.get('exp_potential', 'N/A')} V

### 中性态几何
- **Meso 二面角**: {neu_geom.get('meso_dihedral', 'N/A')}°
- **最大共轭长度**: {neu_geom.get('conjugation_lengths', {}).get('max_conjugation_length', 'N/A')}
- **β位二面角**: {neu_geom.get('beta_dihedrals', 'N/A')}

### 还原态几何
- **Meso 二面角**: {red_geom.get('meso_dihedral', 'N/A')}°

### 结构重组
- **Δ二面角**: {reorg.get('delta_dihedral', 'N/A')}°
- **ΔRMSD**: {reorg.get('delta_rmsd', 'N/A')} Å
- **重组类型**: {reorg.get('reorganization_type', 'N/A')}

### 取代基信息
"""
                # 添加取代基详情
                for core_idx, sub_list in subs.items():
                    for sub in sub_list:
                        sub_type = sub.get('type', 'unknown')
                        sub_smiles = sub.get('smiles', 'N/A')[:30]
                        output += f"- **{sub_type}**: `{sub_smiles}`\n"
                
                return output
        
        return f"未找到匹配 '{q}' 的分子。"


@tool
def find_activity_cliff(mol_id: str) -> str:
    """
    为指定分子找 Activity Cliff 对照案例。
    返回结构相似但电位差异大的分子对，用于对比分析推理。
    
    Args:
        mol_id: 目标分子的 ID
    """
    retriever = get_retriever()
    return retriever.get_contrastive_context(mol_id)


@tool
def query_by_mechanism(mechanism_type: str) -> str:
    """
    按机理类型检索分子。
    
    Args:
        mechanism_type: 机理类型，可选值:
            - 'electron_withdrawing': 吸电子基团 (σ > 0.3)
            - 'electron_donating': 供电子基团 (σ < -0.3)
            - 'flattening': 还原时变平
            - 'rigid': 还原时保持刚性
            - 'high_conjugation': 高共轭 (长度 >= 6)
            - 'low_dihedral': 小二面角 (< 30°)
            - 'high_dihedral': 大二面角 (> 60°)
    """
    retriever = get_retriever()
    results = retriever.query_by_mechanism(mechanism_type)
    
    if not results:
        return f"未找到符合 '{mechanism_type}' 机理的分子。"
    
    output = f"找到 {len(results)} 个符合 '{mechanism_type}' 的分子:\n\n"
    for r in results[:8]:
        output += f"- **{r['id']}**: E={r['e_red']:.2f}V, σ={r['meso_sigma']:.2f}, θ={r['meso_dihedral']:.0f}°, {r['reorg_type']}\n"
    
    return output


# 测试
if __name__ == "__main__":
    print("=" * 60)
    print("因果特征检索模块测试")
    print("=" * 60)
    
    retriever = CausalFeatureRetriever()
    print(f"加载 {len(retriever.molecules)} 个分子")
    print(f"构建 {len(retriever.feature_index)} 个特征索引")
    
    # 测试 Activity Cliff 检测
    print("\n--- Activity Cliff Pairs ---")
    pairs = retriever.find_activity_cliff_pairs(top_k=3)
    for p in pairs:
        print(f"{p.mol_a['id']} vs {p.mol_b['id']}: sim={p.similarity:.2f}, ΔE={p.delta_potential:.2f}V")
        print(f"  差异: {p.key_difference}")
    
    # 测试机理检索
    print("\n--- 按机理检索: flattening ---")
    results = retriever.query_by_mechanism("flattening")
    for r in results[:3]:
        print(f"  {r['id']}: E={r['e_red']:.2f}V")
    
    # 测试对比 Context
    print("\n--- 对比学习 Context ---")
    if pairs:
        context = retriever.get_contrastive_context(pairs[0].mol_a['id'])
        print(context[:500])