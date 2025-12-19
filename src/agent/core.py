import os
from typing import Literal
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent  # LangChain 1.0 新接口
from langgraph.checkpoint.memory import MemorySaver

# 1. 导入您的工具 (稍后实现)
# from src.tools.electronic import analyze_electronic_effect
# from src.tools.steric import analyze_steric_effect
# from src.tools.retrieval import search_knowledge_base

# 暂时定义一个 Mock 工具用于测试架构
@tool
def mock_hammett_calculator(smiles: str):
    """计算分子的 Hammett 常数总和"""
    return "Meso-position sum: 0.78 (Electron Withdrawing)"

# 2. 初始化 LLM
llm = ChatOpenAI(
    model="gpt-4o", 
    temperature=0.2,
    api_key=os.getenv("OPENAI_API_KEY")
)

# 3. 定义工具列表
tools = [mock_hammett_calculator]

# 4. 构建 Agent (LangChain 1.0 风格)
# 1.0 版本中 create_agent 自动处理了 ToolNode 和 ModelNode 的连接
agent_executor = create_agent(
    llm,
    tools,
    checkpointer=MemorySaver(), # 原生支持记忆
    system_prompt="""
    你是由 LangChain 1.0 驱动的 BodiMechanist。
    
    你的核心原则：
    1. 机理驱动：必须通过 Hammett 效应和空间位阻来解释电位。
    2. 数据验证：所有设计建议必须经过数据库检索验证。
    """
)

# 5. 测试运行函数
def run_agent(user_input: str, thread_id: str = "1"):
    config = {"configurable": {"thread_id": thread_id}}
    
    # LangGraph 的流式输出
    for chunk in agent_executor.stream(
        {"messages": [("user", user_input)]}, 
        config
    ):
        print(chunk)

if __name__ == "__main__":
    run_agent("BODIPY 分子在 meso 位引入硝基会有什么影响？")