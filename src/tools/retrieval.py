import json
import os
from langchain_core.tools import tool
from pydantic import BaseModel, Field

DB_PATH = "data/processed/molecules.json"

class AdvancedQueryInput(BaseModel):
    query_text: str = Field(description="查询关键词")
    filter_type: str = Field(
        default="general", 
        description="过滤器类型: 'general' (通用), 'reorganization' (重组模式), 'potential' (电位范围)"
    )

@tool(args_schema=AdvancedQueryInput)
def query_bodi_database(query_text: str, filter_type: str = "general") -> str:
    """
    多模态检索 BODIPY 知识库。
    支持按'重组模式'检索，例如查找所有'Flattening'(变平)或'Twisting'(变扭)的案例。
    这对于验证机理假设至关重要。
    """
    if not os.path.exists(DB_PATH):
        return "Error: Database not found."
    
    with open(DB_PATH, 'r') as f:
        data = json.load(f)
    
    results = []
    q = query_text.lower()
    
    for mol in data:
        match = False
        info = ""
        
        # 模式 A: 重组模式检索
        if filter_type == "reorganization":
            reorg_type = mol.get('reorganization_metrics', {}).get('reorganization_type', '').lower()
            if q in reorg_type:
                match = True
                info = f"[ID: {mol['id']}] Type: {reorg_type.title()} | Delta Angle: {mol['reorganization_metrics']['delta_dihedral']}°"

        # 模式 B: 通用检索 (兼容旧功能)
        else:
            if q in mol['id'].lower() or q in mol.get('description', '').lower():
                match = True
                states = mol.get('states', {})
                info = (f"[ID: {mol['id']}] E_red: {mol['reduction_potential']}V | "
                        f"Neu_Angle: {states.get('neutral', {}).get('dihedral_angle')}° -> "
                        f"Red_Angle: {states.get('reduced', {}).get('dihedral_angle')}°")
        
        if match:
            results.append(info)
            
    if not results:
        return f"No records found for query '{query_text}' with filter '{filter_type}'."
    
    return f"Found {len(results)} matching cases:\n" + "\n".join(results[:5])