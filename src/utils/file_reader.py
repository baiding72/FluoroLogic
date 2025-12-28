import os
import json
import pandas as pd
import cclib
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from src.utils.chem_utils import BodipyScaffoldMatcher, BodipyStericAnalyzer
from src.utils.json_utils import dump_json_pretty

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
        """
        [智能版] 基于共价半径构建分子拓扑
        解决 C-I 键过长但 H...H 非键距离过短的冲突
        """
        mol = Chem.RWMol()
        pt = Chem.GetPeriodicTable() # 获取元素周期表工具
        
        # 1. 添加原子
        for atomic_num in atom_nos:
            atom = Chem.Atom(int(atomic_num))
            atom.SetNoImplicit(True)
            mol.AddAtom(atom)
            
        # 2. 设置坐标
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, (x, y, z) in enumerate(coords):
            conf.SetAtomPosition(i, (float(x), float(y), float(z)))
        mol.AddConformer(conf)
        
        # 3. 智能连键
        num_atoms = mol.GetNumAtoms()
        
        # 预先获取所有原子的共价半径，避免循环内重复调用
        radii = [pt.GetRcovalent(int(n)) for n in atom_nos]
        
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                pos_i = coords[i]
                pos_j = coords[j]
                dist = np.linalg.norm(pos_i - pos_j)
                
                # 获取两原子的共价半径之和
                # RDKit 的 Rcovalent 单位是埃 (Angstrom)
                sum_radii = radii[i] + radii[j]
                
                # 判定标准：
                # 1. 下限 0.4: 防止原子重合导致的错误
                # 2. 上限: 半径之和 + 0.35 Å 的容差 (Tolerance)
                #    对于 C-I (0.76 + 1.33 = 2.09)，加上 0.35 = 2.44，足以覆盖 2.14 的键长
                #    对于 H...H (0.32 + 0.32 = 0.64)，加上 0.35 = 0.99，远小于常见的非键距离(>1.6)
                if 0.4 < dist < (sum_radii + 0.35):
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                    
        return mol.GetMol()

    def _parse_log_structure(self, file_path):
        try:
            data = cclib.io.ccopen(file_path).parse()
            energy_ev = data.scfenergies[-1]
            mol = self._build_mol_from_coords(data.atomnos, data.atomcoords[-1])
            try: Chem.FastFindRings(mol)
            except: pass
            return {"energy": energy_ev, "mol": mol}
        except Exception as e:
            return None

    # === 定义一个辅助函数来跑分析 ===
    def analyze_conformation(self, mol, scaffold, subs_detailed):
        """对给定的构象进行全套几何分析"""
        if not mol or not scaffold: return None
        
        analyzer = BodipyStericAnalyzer(mol, scaffold)
        
        metrics = {
            # 1. 骨架变形
            **analyzer.calc_core_rmsd(), # 包含 core_rmsd, max_out_of_plane
            # 2. 对称性 (虽然 mass 不变，但空间分布可能微调，算一下无妨)
            **analyzer.calc_symmetry_index(),
            # 3. 空间高度 (张开程度)
            "steric_heights": analyzer.calc_steric_heights(subs_detailed),
            # 4. [新增] 折叠程度 (最小距离)
            "proximal_distances": analyzer.calc_proximal_distances(subs_detailed)
        }
        
        # 5. 二面角 (Meso)
        meso_struct = self.matcher.analyze_meso_structure(mol, scaffold)
        metrics["meso_dihedral"] = meso_struct.get("dihedral_angle") if meso_struct else None
        
        return metrics

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
            if not neu_res or not red_res: continue

            # 1. 骨架识别
            is_bodipy, scaffold = self.matcher.analyze(neu_res['mol'])
            
            # 2. 取代基提取 (化学语义)
            substituents_detailed = {}
            steric_metrics = None
            
            is_dimer = False
            if not is_bodipy and scaffold and scaffold.get("error") == "no_N_B_N_bridge":
                # N-B-N 检查失败，可能是二聚体
                # 这里简单标记，不再深入提取
                is_dimer = True
            
            # === 执行双态分析 ===
            neutral_metrics = None
            reduced_metrics = None

            # 仅当是标准 BODIPY 时进行详细提取
            if is_bodipy:
                # A. 提取详细取代基 (Dict: core_idx -> list of info)
                substituents_detailed = self.matcher.extract_substituents_detailed(neu_res['mol'], scaffold)
                
                # 分别分析两种构象
                neutral_metrics = self.analyze_conformation(neu_res['mol'], scaffold, substituents_detailed)
                # 注意：还原态直接复用 Neutral 的 scaffold 和 substituents 索引信息
                # 前提：原子编号在 neu.log 和 red.log 中必须一致！
                reduced_metrics = self.analyze_conformation(red_res['mol'], scaffold, substituents_detailed)
            else:
                # 非 BODIPY，保持默认空值
                red_dihedral = None

            # 能量与重组
            neu_E = neu_res['energy']
            red_E = red_res['energy']
            delta_E = red_E - neu_E 
            delta_dihedral = None
            reorg_type = "Rigid/Unknown"
        

            entry = {
                "id": mol_id,
                "is_bodipy": is_bodipy,
                "is_dimer": is_dimer,
                "smiles": meta_info['smiles'],
                
                # === 核心数据结构 ===
                "substituents": substituents_detailed,
                "steric_properties": steric_metrics,
                # ===================

                "potential_info": {
                    "dft_potential_csv_V": meta_info['dft_potential_csv'],
                    "calc_energy_gap_eV": round(delta_E, 3)
                },
                "optical_properties": {
                    "abs_max_nm": meta_info['abs_max_nm']
                },
                # === 核心修改：双态几何数据 ===
                "states": {
                    "neutral": {
                        "energy_ev": round(neu_E, 4),
                        "geometry": neutral_metrics # <--- 包含 RMSD, Height, Dist, Dihedral
                    },
                    "reduced": {
                        "energy_ev": round(red_E, 4),
                        "geometry": reduced_metrics # <--- 同上，对比由此产生
                    }
                },
                # 可以在这里预计算一些 Delta 值方便 Agent 直接看
                "reorganization_metrics": {
                    "delta_energy_ev": round(delta_E, 3),
                    "delta_dihedral": round(reduced_metrics['meso_dihedral'] - neutral_metrics['meso_dihedral'], 1) if (neutral_metrics and reduced_metrics and neutral_metrics['meso_dihedral']) else None,
                    "delta_rmsd": round(reduced_metrics['core_rmsd'] - neutral_metrics['core_rmsd'], 4) if (neutral_metrics and reduced_metrics) else None,
                    # 新增：判断是否发生"呼吸" (张开/闭合)
                    "delta_max_height": round(reduced_metrics['steric_heights']['max_height_overall'] - neutral_metrics['steric_heights']['max_height_overall'], 3) if (neutral_metrics and reduced_metrics) else None
                },
            }
            processed_data.append(entry)
            
        dump_json_pretty(processed_data, OUTPUT_JSON)

        print(f"\nProcessing Complete! Saved {len(processed_data)} molecules.")

if __name__ == "__main__":
    integrator = DataIntegrator()
    integrator.process_all()