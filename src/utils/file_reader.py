import os
import json
import pandas as pd
import cclib
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

# === 引入我们新建的通用工具 ===
# 确保在项目根目录运行，或者将 src 加入 PYTHONPATH
try:
    from src.utils.chem_utils import BodipyScaffoldMatcher
except ImportError:
    # 简单的 fallback，方便直接运行脚本测试
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.utils.chem_utils import BodipyScaffoldMatcher

# ================= 配置区 =================
RAW_DFT_DIR = "data/raw_DFT"
CSV_PATH = "data/data.csv"
OUTPUT_JSON = "data/processed/molecules.json"
# =========================================

class DataIntegrator:
    def __init__(self):
        self.metadata = self._load_metadata()
        self.matcher = BodipyScaffoldMatcher() # 初始化一次即可复用
        
    def _load_metadata(self):
        """加载 CSV 并处理空值"""
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"CSV file not found at {CSV_PATH}")
        
        df = pd.read_csv(CSV_PATH)
        df.columns = [c.strip() for c in df.columns]
        df = df.where(pd.notnull(df), None)
        
        meta = {}
        for _, row in df.iterrows():
            mol_id = str(row.get('MOLECULEID', '')).strip()
            if mol_id:
                meta[mol_id] = {
                    "smiles": row.get('smiles_cano', ''),
                    "abs_max_nm": row.get('Absorbance maximum wavelength'), 
                    "dft_potential_csv": row.get('E_AgAgCl_V') 
                }
        print(f"Loaded metadata for {len(meta)} molecules from CSV.")
        return meta

    def _build_mol_from_coords(self, atom_nos, coords):
        """手动构建分子拓扑"""
        mol = Chem.RWMol()
        for atomic_num in atom_nos:
            atom = Chem.Atom(int(atomic_num))
            atom.SetNoImplicit(True)
            mol.AddAtom(atom)
            
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, (x, y, z) in enumerate(coords):
            conf.SetAtomPosition(i, (float(x), float(y), float(z)))
        mol.AddConformer(conf)
        
        num_atoms = mol.GetNumAtoms()
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                pos_i = coords[i]
                pos_j = coords[j]
                dist = np.linalg.norm(pos_i - pos_j)
                if 0.4 < dist < 1.85: 
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
        return mol.GetMol()

    def _parse_log_structure(self, file_path):
        """解析 Log"""
        try:
            data = cclib.io.ccopen(file_path).parse()
            energy_ev = data.scfenergies[-1]
            mol = self._build_mol_from_coords(data.atomnos, data.atomcoords[-1])
            return {"energy": energy_ev, "mol": mol}
        except Exception as e:
            return None

    def _calculate_dihedral(self, mol_3d, mol_id="Unknown"):
        """
        利用 BodipyScaffoldMatcher 计算二面角
        逻辑极其简洁：只需调用通用工具
        """
        if not mol_3d: return None

        # 1. 识别骨架
        scaffold = self.matcher.analyze(mol_3d)
        if not scaffold:
            # print(f"    [Debug] {mol_id}: Not a valid BODIPY scaffold.")
            return None
            
        # 2.以此骨架获取测角所需的4个原子
        target_atoms = self.matcher.get_dihedral_atoms(mol_3d, scaffold)
        
        if target_atoms:
            try:
                conf = mol_3d.GetConformer()
                angle = rdMolTransforms.GetDihedralDeg(conf, *target_atoms)
                
                # 归一化 (0-90)
                angle = abs(angle)
                while angle > 90:
                    angle = abs(180 - angle)
                return round(angle, 1)
            except:
                return None
        
        return None

    def process_all(self):
        processed_data = []
        print(f"Start processing. Meta size: {len(self.metadata)}")
        
        for mol_id, meta_info in self.metadata.items():
            neu_path = os.path.join(RAW_DFT_DIR, f"{mol_id}_sp.log")
            red_path = os.path.join(RAW_DFT_DIR, f"{mol_id}_reduced_sp.log")
            
            if not (os.path.exists(neu_path) and os.path.exists(red_path)):
                continue
            
            neu_res = self._parse_log_structure(neu_path)
            red_res = self._parse_log_structure(red_path)
            
            if not neu_res or not red_res:
                print(f"  -> Parse failed for {mol_id}")
                continue
            
            if mol_id == 'BNP_CHO':
                print(f"  -> Debugging {mol_id}")

            neu_angle = self._calculate_dihedral(neu_res['mol'], f"{mol_id}_neu")
            red_angle = self._calculate_dihedral(red_res['mol'], f"{mol_id}_red")
            
            # 能量与重组能
            neu_E = neu_res['energy']
            red_E = red_res['energy']
            delta_E = red_E - neu_E 
            
            delta_dihedral = None
            reorg_type = "Rigid/Unknown"
            
            if neu_angle is not None and red_angle is not None:
                delta_dihedral = round(red_angle - neu_angle, 1)
                # 判定重组类型
                if abs(red_angle) < abs(neu_angle) - 5: reorg_type = "Flattening"
                elif abs(red_angle) > abs(neu_angle) + 5: reorg_type = "Twisting"
                else: reorg_type = "Rigid"
            
            entry = {
                "id": mol_id,
                "smiles": meta_info['smiles'],
                "potential_info": {
                    "dft_potential_csv_V": meta_info['dft_potential_csv'],
                    "calc_energy_gap_eV": round(delta_E, 3)
                },
                "optical_properties": {
                    "abs_max_nm": meta_info['abs_max_nm']
                },
                "states": {
                    "neutral": {"energy_ev": round(neu_E, 4), "dihedral_angle": neu_angle},
                    "reduced": {"energy_ev": round(red_E, 4), "dihedral_angle": red_angle}
                },
                "reorganization_metrics": {
                    "delta_dihedral": delta_dihedral,
                    "reorganization_type": reorg_type
                }
            }
            processed_data.append(entry)
            
        os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(processed_data, f, indent=4, default=str)
        print(f"\nProcessing Complete! Saved {len(processed_data)} molecules.")

if __name__ == "__main__":
    integrator = DataIntegrator()
    integrator.process_all()