import os
import json
import pandas as pd
import cclib
import numpy as np
from rdkit import Chem

try:
    from src.utils.chem_utils import BodipyScaffoldMatcher
except ImportError:
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
        self.matcher = BodipyScaffoldMatcher()
        
    def _load_metadata(self):
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
        """手动构建分子拓扑，确保显式氢存在以计算配位数"""
        mol = Chem.RWMol()
        for atomic_num in atom_nos:
            atom = Chem.Atom(int(atomic_num))
            atom.SetNoImplicit(True) # 显式处理，不自动加氢
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
        try:
            data = cclib.io.ccopen(file_path).parse()
            energy_ev = data.scfenergies[-1]
            mol = self._build_mol_from_coords(data.atomnos, data.atomcoords[-1])
            
            # 必须调用 FindRings，否则 IsInRing() 判断会失效
            try: Chem.FastFindRings(mol)
            except: pass
            
            return {"energy": energy_ev, "mol": mol}
        except Exception as e:
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
                continue

            # 1. 骨架识别 (在中性态上进行化学结构分析)
            is_bodipy, scaffold = self.matcher.analyze(neu_res['mol'])
            
            # 2. 结构化位点分析 (Substituent Analysis)
            # 默认初始化为空结构
            substituents_info = {
                "meso": None,
                "alpha": None, # 预留
                "beta": None   # 预留
            }
            
            # 提取 Meso 结构信息 (类型、杂化、二面角)
            meso_analysis = None
            if is_bodipy and scaffold:
                meso_analysis = self.matcher.analyze_meso_structure(neu_res['mol'], scaffold)
                substituents_info["meso"] = meso_analysis

            # 3. 还原态构象分析 (仅用于计算重组能)
            # 我们需要重新在还原态上跑一遍 analyze 以获取当时的二面角
            red_dihedral = None
            if is_bodipy and meso_analysis and meso_analysis['is_conjugated']:
                 # 只有当 Meso 是共轭连接时，计算还原态二面角才有意义
                 # 需要重新匹配骨架，因为原子索引在不同 Log 文件中可能一致，但也可能微变(虽然通常Gaussian保持顺序)
                 # 保险起见，我们假设原子顺序一致，直接用 scaffold 的索引
                 # 如果不一致，需要对 red_mol 重新做 matcher.analyze
                 red_angle_val = self.matcher._compute_dihedral_value(
                     red_res['mol'], scaffold, meso_analysis['anchor_idx']
                 )
                 red_dihedral = red_angle_val

            # 4. 计算指标
            neu_E = neu_res['energy']
            red_E = red_res['energy']
            delta_E = red_E - neu_E 
            
            neu_dihedral = meso_analysis['dihedral_angle'] if meso_analysis else None
            
            delta_dihedral = None
            reorg_type = "Rigid/Unknown"
            
            if neu_dihedral is not None and red_dihedral is not None:
                delta_dihedral = round(red_dihedral - neu_dihedral, 1)
                if abs(red_dihedral) < abs(neu_dihedral) - 5: reorg_type = "Flattening"
                elif abs(red_dihedral) > abs(neu_dihedral) + 5: reorg_type = "Twisting"
                else: reorg_type = "Rigid"

            entry = {
                "id": mol_id,
                "is_bodipy": is_bodipy,
                "smiles": meta_info['smiles'],
                
                # === 新增结构化字段 ===
                "substituents": substituents_info,
                # ===================

                "potential_info": {
                    "dft_potential_csv_V": meta_info['dft_potential_csv'],
                    "calc_energy_gap_eV": round(delta_E, 3)
                },
                "optical_properties": {
                    "abs_max_nm": meta_info['abs_max_nm']
                },
                "states": {
                    "neutral": {
                        "energy_ev": round(neu_E, 4), 
                        # 如果没有共轭二面角，这里就是 None，合理
                        "dihedral_angle": neu_dihedral 
                    },
                    "reduced": {
                        "energy_ev": round(red_E, 4), 
                        "dihedral_angle": red_dihedral
                    }
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