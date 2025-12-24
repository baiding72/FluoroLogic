import json
import os
import math
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

# 加载你的双态数据
DATA_PATH = "data/processed/molecules.json"

def load_data():
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, 'r') as f:
            return json.load(f)
    return []

MOLECULES_DB = load_data()

class ReorgInput(BaseModel):
    smiles_or_id: str = Field(description="分子的 ID (如 BODIPY-001) 或 SMILES 字符串")

def _calculate_similarity(smiles1, smiles2):
    """辅助函数：计算两个 SMILES 的 Tanimoto 相似度"""
    if not Chem: return 0.0
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except:
        return 0.0

@tool(args_schema=ReorgInput)
def analyze_structural_reorganization(smiles_or_id: str) -> str:
    """
    核心工具：分析分子在氧化还原过程中的结构重组 (Reorganization) 效应。
    
    功能：
    1. 如果输入是已知 ID，返回精确的中性/还原态构象变化数据。
    2. 如果输入是新分子 (SMILES)，在数据库中检索最相似的"替身"，
       并基于替身的构象变化趋势，预测新分子的弛豫模式（变平/变扭）。
    
    这是预测电位是否因"绝热电子转移"而发生偏移的关键依据。
    """
    # 1. 尝试精确匹配 ID
    target_mol = next((m for m in MOLECULES_DB if m['id'].lower() == smiles_or_id.lower()), None)
    
    if target_mol:
        # 命中数据库真值
        metrics = target_mol.get('reorganization_metrics', {})
        states = target_mol.get('states', {})
        return (
            f"✅ Database Hit ({target_mol['id']}):\n"
            f"- Neutral Dihedral: {states.get('neutral', {}).get('dihedral_angle', 'N/A')}°\n"
            f"- Reduced Dihedral: {states.get('reduced', {}).get('dihedral_angle', 'N/A')}°\n"
            f"- Change (Delta): {metrics.get('delta_dihedral', 'N/A')}°\n"
            f"- Reorganization Type: {metrics.get('reorganization_type', 'Unknown')}\n"
            f"Analysis: The molecule undergoes '{metrics.get('reorganization_type')}' upon reduction, "
            f"contributing {metrics.get('delta_G', 0)} kcal/mol to stability."
        )
    
    # 2. 如果是 SMILES，进行相似度检索 (CBR 推理)
    if Chem and "C" in smiles_or_id: # 简单判断是否为 SMILES
        best_match = None
        highest_sim = -1
        
        for mol in MOLECULES_DB:
            sim = _calculate_similarity(smiles_or_id, mol['smiles'])
            if sim > highest_sim:
                highest_sim = sim
                best_match = mol
        
        if best_match and highest_sim > 0.4: # 设置一个相似度阈值
            metrics = best_match.get('reorganization_metrics', {})
            return (
                f"⚠️ New Molecule Prediction (Based on Analogy):\n"
                f"Most similar existing case is {best_match['id']} (Similarity: {highest_sim:.2f}).\n"
                f"Reference Case Behavior:\n"
                f"- It undergoes '{metrics.get('reorganization_type')}' (Delta: {metrics.get('delta_dihedral')}°).\n"
                f"Inference:\n"
                f"Based on structural similarity, your molecule is highly likely to exhibit similar {metrics.get('reorganization_type')} behavior. "
                f"Expect significant stabilization energy similar to {best_match['id']}."
            )
            
    return "Error: Molecule not found in DB and structural analogy failed."