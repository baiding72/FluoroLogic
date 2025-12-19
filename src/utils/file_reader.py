from cclib.io import ccread
import numpy as np
import os
import json
from rdkit import Chem

class DFTDataReader:
    """
    负责读取 Gaussian 输出文件，提取结构和能量信息，
    构建 Agent 的高精度知识库。
    """
    def __init__(self, raw_data_dir):
        self.raw_data_dir = raw_data_dir

    def parse_gaussian_log(self, file_path):
        """解析单个 .log 文件"""
        try:
            data = ccread(file_path)
            
            # 提取关键信息
            properties = {
                "filename": os.path.basename(file_path),
                "homo_energy": float(data.moenergies[0][data.homos[0]]), # eV
                "lumo_energy": float(data.moenergies[0][data.homos[0] + 1]), # eV
                "dipole_moment": float(np.linalg.norm(data.moments[1])), # Debye
                # 还可以提取最终坐标用于计算二面角
                "atomcoords": data.atomcoords[-1].tolist(),
                "atomnos": data.atomnos.tolist()
            }
            return properties
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None

    def build_knowledge_base(self, output_path="data/processed/molecules.json"):
        """批量处理并保存"""
        database = []
        for filename in os.listdir(self.raw_data_dir):
            if filename.endswith(".log") or filename.endswith(".out"):
                file_path = os.path.join(self.raw_data_dir, filename)
                props = self.parse_gaussian_log(file_path)
                
                if props:
                    # TODO: 这里之后需要整合你 Excel 里的 SMILES 和 电位实验值
                    # 假设你有一个 mapping 或者文件名包含 ID
                    props["smiles"] = "..." # 暂时留空，后续填补
                    props["reduction_potential_exp"] = -1.0 # 暂时留空
                    database.append(props)
        
        with open(output_path, 'w') as f:
            json.dump(database, f, indent=4)
        print(f"Knowledge base built with {len(database)} molecules.")

# 测试代码
if __name__ == "__main__":
    # 假设你的数据在 data/raw_DFT
    reader = DFTDataReader("../../data/raw_DFT")
    # reader.build_knowledge_base() 
    print("Reader initialized. Ready to parse.")