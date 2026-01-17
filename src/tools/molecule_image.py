"""
分子图像处理工具
功能：
1. SMILES → 分子结构图 (RDKit)
2. 分子结构图 → SMILES (MolScribe/在线API)
3. 分子修改可视化
"""

import os
import io
import base64
import tempfile
from typing import Optional, Tuple
from PIL import Image

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available, molecule drawing disabled")


def smiles_to_image(smiles: str, size: Tuple[int, int] = (400, 300), 
                    highlight_atoms: list = None) -> Optional[Image.Image]:
    """
    将 SMILES 转换为分子结构图
    
    Args:
        smiles: 分子 SMILES 字符串
        size: 图像大小 (width, height)
        highlight_atoms: 要高亮的原子索引列表
    
    Returns:
        PIL Image 对象
    """
    if not RDKIT_AVAILABLE:
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # 生成 2D 坐标
        AllChem.Compute2DCoords(mol)
        
        # 创建绘图对象
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        
        # 设置绘图选项
        opts = drawer.drawOptions()
        opts.addAtomIndices = False
        opts.addStereoAnnotation = True
        
        # 绘制分子
        if highlight_atoms:
            drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)
        else:
            drawer.DrawMolecule(mol)
        
        drawer.FinishDrawing()
        
        # 转换为 PIL Image
        png_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(png_data))
        
        return img
        
    except Exception as e:
        print(f"Error generating molecule image: {e}")
        return None


def smiles_to_base64(smiles: str, size: Tuple[int, int] = (400, 300)) -> Optional[str]:
    """
    将 SMILES 转换为 Base64 编码的图像
    用于在 Markdown 中嵌入图像
    """
    img = smiles_to_image(smiles, size)
    if img is None:
        return None
    
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    return base64.b64encode(buffer.read()).decode('utf-8')


def compare_molecules(smiles1: str, smiles2: str, 
                      labels: Tuple[str, str] = ("Before", "After"),
                      size: Tuple[int, int] = (350, 250)) -> Optional[Image.Image]:
    """
    并排比较两个分子结构
    
    Args:
        smiles1: 第一个分子 SMILES
        smiles2: 第二个分子 SMILES  
        labels: 标签 (label1, label2)
        size: 单个分子图像大小
    
    Returns:
        并排对比的 PIL Image
    """
    if not RDKIT_AVAILABLE:
        return None
    
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return None
        
        AllChem.Compute2DCoords(mol1)
        AllChem.Compute2DCoords(mol2)
        
        # 使用 MolsToGridImage 创建并排图像
        img = Draw.MolsToGridImage(
            [mol1, mol2],
            molsPerRow=2,
            subImgSize=size,
            legends=list(labels)
        )
        
        return img
        
    except Exception as e:
        print(f"Error comparing molecules: {e}")
        return None


def save_molecule_image(smiles: str, filepath: str, size: Tuple[int, int] = (400, 300)) -> bool:
    """
    保存分子结构图到文件
    """
    img = smiles_to_image(smiles, size)
    if img is None:
        return False
    
    try:
        img.save(filepath)
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False


# ==================== 图像识别 (OCSR) ====================

def image_to_smiles_molscribe(image_path: str) -> Optional[str]:
    """
    使用 MolScribe 将分子图像转换为 SMILES
    
    注意: 需要安装 molscribe: pip install molscribe
    """
    try:
        from molscribe import MolScribe
        
        model = MolScribe()
        smiles = model.predict_image(image_path)
        
        return smiles
    except ImportError:
        print("MolScribe not installed. Install with: pip install molscribe")
        return None
    except Exception as e:
        print(f"MolScribe error: {e}")
        return None


def image_to_smiles_decimer(image_path: str) -> Optional[str]:
    """
    使用 DECIMER 将分子图像转换为 SMILES
    
    注意: 需要安装 DECIMER: pip install DECIMER
    """
    try:
        from DECIMER import predict_SMILES
        
        smiles = predict_SMILES(image_path)
        return smiles
    except ImportError:
        print("DECIMER not installed. Install with: pip install DECIMER")
        return None
    except Exception as e:
        print(f"DECIMER error: {e}")
        return None


def image_to_smiles(image_path: str) -> Tuple[Optional[str], str]:
    """
    尝试多种方法将分子图像转换为 SMILES
    
    Returns:
        (smiles, method) 或 (None, error_message)
    """
    # 尝试 MolScribe
    try:
        from molscribe import MolScribe
        model = MolScribe()
        smiles = model.predict_image(image_path)
        if smiles:
            # 验证 SMILES
            if RDKIT_AVAILABLE:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    return smiles, "MolScribe"
    except:
        pass
    
    # 尝试 DECIMER
    try:
        from DECIMER import predict_SMILES
        smiles = predict_SMILES(image_path)
        if smiles and RDKIT_AVAILABLE:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return smiles, "DECIMER"
    except:
        pass
    
    return None, "无法识别分子结构。请安装 molscribe 或 DECIMER。"


# ==================== LangChain Tools ====================

from langchain_core.tools import tool


@tool
def draw_molecule(smiles: str) -> str:
    """
    绘制分子结构图。
    
    Args:
        smiles: 分子的 SMILES 表示
    
    Returns:
        分子信息和图片保存路径
    """
    if not RDKIT_AVAILABLE:
        return "RDKit 未安装，无法绘制分子结构。"
    
    # 验证 SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"无效的 SMILES: {smiles}"
    
    # 保存图像到临时文件
    import uuid
    filename = f"/tmp/mol_{uuid.uuid4().hex[:8]}.png"
    
    success = save_molecule_image(smiles, filename)
    if not success:
        return "生成分子图像失败。"
    
    # 计算分子信息
    formula = rdMolDescriptors.CalcMolFormula(mol)
    mw = Descriptors.MolWt(mol)
    
    return f"""
**分子结构图已生成**

- **文件路径**: {filename}
- **SMILES**: `{smiles}`
- **分子式**: {formula}
- **分子量**: {mw:.2f}

(图片已保存，可在应用中查看)
"""


@tool
def compare_molecule_structures(smiles_before: str, smiles_after: str, 
                                 label_before: str = "修改前", 
                                 label_after: str = "修改后") -> str:
    """
    并排比较两个分子结构，用于展示分子修改建议。
    
    Args:
        smiles_before: 修改前的分子 SMILES
        smiles_after: 修改后的分子 SMILES
        label_before: 修改前标签
        label_after: 修改后标签
    
    Returns:
        包含对比图像的 Markdown 文本
    """
    if not RDKIT_AVAILABLE:
        return "RDKit 未安装，无法绘制分子结构。"
    
    mol1 = Chem.MolFromSmiles(smiles_before)
    mol2 = Chem.MolFromSmiles(smiles_after)
    
    if mol1 is None:
        return f"修改前 SMILES 无效: {smiles_before}"
    if mol2 is None:
        return f"修改后 SMILES 无效: {smiles_after}"
    
    # 生成对比图
    img = compare_molecules(smiles_before, smiles_after, (label_before, label_after))
    if img is None:
        return "生成对比图像失败。"
    
    # 转换为 Base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    b64_img = base64.b64encode(buffer.read()).decode('utf-8')
    
    # 计算差异
    mw1 = Descriptors.MolWt(mol1)
    mw2 = Descriptors.MolWt(mol2)
    
    return f"""
**分子结构对比**

![comparison](data:image/png;base64,{b64_img})

| 属性 | {label_before} | {label_after} |
|------|---------|---------|
| SMILES | `{smiles_before[:40]}...` | `{smiles_after[:40]}...` |
| 分子量 | {mw1:.2f} | {mw2:.2f} |
| Δ分子量 | - | {mw2 - mw1:+.2f} |
"""


# 测试
if __name__ == "__main__":
    print("=" * 50)
    print("分子图像处理工具测试")
    print("=" * 50)
    
    if not RDKIT_AVAILABLE:
        print("RDKit 未安装，跳过测试")
        exit()
    
    # 测试 SMILES 转图像
    test_smiles = "c1ccc2c(c1)[nH]c1ccc(Br)cc12"  # 溴代咔唑
    
    print(f"\n测试 SMILES: {test_smiles}")
    
    img = smiles_to_image(test_smiles)
    if img:
        print(f"生成图像成功: {img.size}")
        img.save("/tmp/test_molecule.png")
        print("保存到 /tmp/test_molecule.png")
    
    # 测试对比
    smiles1 = "c1ccc2c(c1)[nH]c1ccccc12"  # 咔唑
    smiles2 = "c1ccc2c(c1)[nH]c1ccc(Br)cc12"  # 溴代咔唑
    
    print(f"\n对比测试:")
    print(f"  分子1: {smiles1}")
    print(f"  分子2: {smiles2}")
    
    comp_img = compare_molecules(smiles1, smiles2, ("咔唑", "溴代咔唑"))
    if comp_img:
        comp_img.save("/tmp/test_comparison.png")
        print("对比图保存到 /tmp/test_comparison.png")
