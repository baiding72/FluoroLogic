from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from rdkit import Chem
# 引用我们在 utils 里写的核心逻辑
from src.utils.chem_utils import BodipyScaffoldMatcher, BodipyStericAnalyzer

class StericAnalysisInput(BaseModel):
    smiles: str = Field(description="The SMILES string of the BODIPY molecule to analyze.")

class StericAnalysisTool(BaseTool):
    name = "analyze_steric_properties"
    description = """
    Useful for analyzing the 3D spatial conformation and steric hindrance of a BODIPY molecule.
    Returns quantitative metrics like Core RMSD (planarity), Symmetry Index, and Proximal Distances.
    Use this when you need to judge if the molecule is planar/conjugated or distorted.
    """
    args_schema = StericAnalysisInput

    def _run(self, smiles: str):
        # 1. 初始化分子
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return "Invalid SMILES."
        
        # 注意：这里可能需要先生成 3D 构象 (EmbedMolecule) 
        # 因为 Agent 输入的 SMILES 通常是 2D 的，而 file_reader 处理的是 DFT 算好的 3D 结构
        # 这是一个关键的工程差异！
        mol = Chem.AddHs(mol)
        Chem.AllChem.EmbedMolecule(mol, Chem.AllChem.ETKDG()) 

        # 2. 调用核心逻辑 (Engine)
        matcher = BodipyScaffoldMatcher()
        success, scaffold = matcher.analyze(mol)
        
        if not success:
            return "Failed to identify BODIPY scaffold."
            
        analyzer = BodipyStericAnalyzer(mol, scaffold)
        
        # 3. 获取数据
        rmsd_info = analyzer.calc_core_rmsd()
        symmetry = analyzer.calc_symmetry_index()
        # ... 其他计算
        
        # 4. 生成自然语言报告 (给 LLM 看的)
        report = f"""
        **Steric Analysis Report**
        - Core Planarity (RMSD): {rmsd_info['rmsd']} Å (Max displacement: {rmsd_info['max_out_of_plane']} Å)
          -> Interpretation: {"Highly planar" if rmsd_info['rmsd'] < 0.05 else "Distorted/Buckled"}
        - Symmetry Index: {symmetry['mass_asymmetry']} (Left mass: {symmetry['left_mass']}, Right mass: {symmetry['right_mass']})
          -> Interpretation: {"Symmetric" if symmetry['mass_asymmetry'] < 0.1 else "Asymmetric distribution"}
        """
        return report