import os
import sys
from pathlib import Path

def check_env_loading():
    """检查环境变量是否正确加载"""
    print("=== 环境变量加载检查 ===")
    
    # 检查当前工作目录
    current_dir = Path.cwd()
    print(f"当前工作目录: {current_dir}")
    
    # 检查 .env 文件是否存在
    env_file = current_dir / ".env"
    if env_file.exists():
        print(f"✓ .env 文件存在: {env_file}")
        
        # 读取 .env 文件内容
        with open(env_file, 'r') as f:
            content = f.read()
            print(f".env 文件内容:\n{content}")
    else:
        print("✗ .env 文件不存在")
        return False

    # 检查 DEEPSEEK_API_KEY 是否存在于环境变量中
    api_key_from_env = os.getenv('DEEPSEEK_API_KEY')
    if api_key_from_env:
        print(f"✓ DEEPSEEK_API_KEY 已从环境变量加载 (长度: {len(api_key_from_env)})")
    else:
        print("✗ DEEPSEEK_API_KEY 未从环境变量加载")
    
    return True

def fix_model_factory():
    """生成修复后的 model_factory.py 示例"""
    fixed_code = '''
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 现在您可以安全地使用环境变量
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY 环境变量未设置")

print(f"API Key loaded: {'Yes' if openai_api_key else 'No'}")

# 示例：使用API密钥初始化客户端
from openai import OpenAI

client = OpenAI(api_key=openai_api_key)

# 或者如果您使用的是 langchain
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)
'''
    return fixed_code

def install_python_dotenv():
    """生成安装命令"""
    return """
# 安装 python-dotenv 包
pip install python-dotenv

# 或者如果您使用 conda
conda install python-dotenv -c conda-forge
"""

def main():
    success = check_env_loading()
    
    if success:
        print("\n=== 建议修复方案 ===")
        print(install_python_dotenv())
        print("\n=== 修改 model_factory.py 文件 ===")
        print("请在 model_factory.py 文件开头添加以下代码:")
        print(fix_model_factory())
    
    print("\n=== 额外检查项 ===")
    print("1. 确认 .env 文件路径正确 (应该在项目根目录)")
    print("2. 确认 .env 文件权限允许读取")
    print("3. 确认 .env 文件格式正确 (没有多余的空格或特殊字符)")
    print("4. 确认 python-dotenv 包已安装")

if __name__ == "__main__":
    main()



