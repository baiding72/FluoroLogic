import pandas as pd
from rdkit import Chem

class HammettCalculator:
    """
    电子效应计算器
    功能：根据输入的 SMILES 或 SMARTS，查询 Hammett 常数 (Sigma_p, Sigma_m)
    """
    def __init__(self, csv_path="data/Hammett.csv"):
        self.db = self._load_database(csv_path)
        
    def _load_database(self, path):
        try:
            df = pd.read_csv(path)
            df = df.where(pd.notnull(df), None)
            # 预编译 SMARTS pattern 以加速匹配
            patterns = []
            for _, row in df.iterrows():
                entry = row.to_dict()
                # 1. 编译 SMARTS
                smarts = entry.get('Smarts_Pattern')
                if smarts:
                    try:
                        # mergeHs=True 可以避免某些氢原子警告
                        entry['mol_pattern'] = Chem.MolFromSmarts(smarts, mergeHs=True)
                        if entry['mol_pattern'] is None:
                            print(f"Warning: Failed to compile SMARTS for {entry['Name']}: {smarts}")
                    except Exception as e:
                        print(f"Error compiling SMARTS for {entry['Name']}: {e}")
                        entry['mol_pattern'] = None
                else:
                    entry['mol_pattern'] = None
                    
                # 2. 编译 SMILES (用于精确匹配)
                smiles = entry.get('Smiles_Pattern')
                if smiles:
                    try:
                        clean_smi = smiles.replace('*', '')
                        if clean_smi:
                            # sanitize=False 可以防止 RDKit 对孤立 [H] 发出警告或报错
                            # 既然只是为了做精确字符串匹配或简单的图结构对比，不需要严格清洗
                            entry['mol_exact'] = Chem.MolFromSmiles(clean_smi, sanitize=True)
                        else:
                            # 如果 SMILES 只有 "*" (虽然不应该发生)，则为空
                            entry['mol_exact'] = None
                    except:
                         entry['mol_exact'] = None
                else:
                    entry['mol_exact'] = None
                
                patterns.append(entry)
            return patterns
        except Exception as e:
            print(f"Error loading Hammett DB: {e}")
            return []

    def get_sigma(self, substituent_smiles):
        """
        根据取代基 SMILES 获取 sigma 参数
        优先级：精确 SMILES 匹配 > SMARTS 子结构匹配 > 默认值
        """
        if not substituent_smiles or substituent_smiles == "[H]":
            return {"sigma_p": 0.0, "sigma_m": 0.0, "name": "Hydrogen"}
            
        # 1. 预处理输入
        # 我们的 json 里存的是带有 * 或 [H] 的，需要清洗一下
        # 简单清洗：去掉 *
        clean_input = substituent_smiles.replace('*', '')
        if not clean_input: # 比如输入就是 "*"
             return {"sigma_p": 0.0, "sigma_m": 0.0, "name": "Unknown"}

        input_mol = Chem.MolFromSmiles(clean_input)
        if not input_mol:
            return {"sigma_p": None, "sigma_m": None, "name": "Invalid SMILES"}

        # 2. 尝试精确匹配 (Exact Match)
        # 这一步是为了区分细微差别，比如 Phenyl 和 Pyridyl
        for entry in self.db:
            target_smi = str(entry.get('Smiles_Pattern', '')).replace('*', '')
            if target_smi == clean_input:
                return {
                    "sigma_p": entry['Sigma_p'],
                    "sigma_m": entry['Sigma_m'],
                    "name": entry['Name']
                }

        # 3. 尝试子结构匹配 (SMARTS Match)
        # 这一步是为了通用性，比如捕捉所有长链烷基为 "Generic Alkyl"
        matched_entry = None
        for entry in self.db:
            pattern = entry.get('mol_pattern')
            if pattern and input_mol.HasSubstructMatch(pattern):
                # 找到匹配！但我们希望找到"最特异"的匹配吗？
                # 目前简单逻辑：找到第一个就返回 (CSV 里的顺序很重要，特异的放前面，通用的放后面)
                matched_entry = entry
                break
        
        if matched_entry:
            return {
                "sigma_p": matched_entry['Sigma_p'],
                "sigma_m": matched_entry['Sigma_m'],
                "name": matched_entry['Name']
            }

        # 4. 未知基团
        return {"sigma_p": None, "sigma_m": None, "name": "Unknown"}

# 测试代码
if __name__ == "__main__":
    calc = HammettCalculator()
    # 测试几个例子
    tests = ["*C", "*[N+](=O)[O-]", "*CCCC"] # 甲基，硝基，丁基(应该匹配 Generic Alkyl)
    for t in tests:
        print(f"{t}: {calc.get_sigma(t)}")