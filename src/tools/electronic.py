import pandas as pd
import os
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# 加载数据 (全局加载一次，避免每次调用都读文件)
HAMMETT_FILE = "data/hammett.csv"
if os.path.exists(HAMMETT_FILE):
    hammett_df = pd.read_csv(HAMMETT_FILE)
    # 转为字典方便查询: {'NO2': {'sigma_p': 0.78, ...}}
    hammett_lookup = hammett_df.set_index('substituent').to_dict('index')
else:
    hammett_lookup = {}
    print("Warning: hammett.csv not found.")

# 定义参数模型 (LangChain 1.0 推荐)
class ElectronicInput(BaseModel):
    substituent_name: str = Field(description="取代基的化学名称或SMILES片段，如 NO2, F, OMe")
    position: str = Field(description="取代位置，可选值: 'para' (对位/直接共轭), 'meta' (间位/诱导)")

@tool(args_schema=ElectronicInput)
def check_hammett(substituent_name: str, position: str) -> str:
    """
    查询指定取代基的 Hammett 常数 (Sigma值)，用于评估电子效应。
    正值代表吸电子(LUMO降低)，负值代表供电子(LUMO升高)。
    """
    # 简单的标准化处理
    key = substituent_name.strip().replace("-", "") # 去掉可能的横杠
    
    # 模糊匹配逻辑 (实际项目中可以用 RDKit 子结构匹配)
    if key not in hammett_lookup:
        return f"Database miss: Found no Hammett constant for '{substituent_name}'. Please estimate based on similar groups."
    
    data = hammett_lookup[key]
    
    if position.lower() in ["para", "p", "meso", "alpha", "beta"]:
        val = data['sigma_p']
        effect = "Electron Withdrawing (EWG)" if val > 0 else "Electron Donating (EDG)"
        return f"Substituent: {substituent_name}, Position: {position}, Sigma_p: {val}, Effect: {effect}"
    elif position.lower() in ["meta", "m"]:
        val = data['sigma_m']
        return f"Substituent: {substituent_name}, Position: Meta, Sigma_m: {val}"
    else:
        return f"Error: Unknown position '{position}'. Use 'para' or 'meta'."

# 单元测试
if __name__ == "__main__":
    print(check_hammett.invoke({"substituent_name": "NO2", "position": "para"}))