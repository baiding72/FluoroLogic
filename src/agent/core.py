"""
Agent 核心模块
支持动态模型切换
"""

import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

# 导入工具
from src.tools.electronic import check_hammett
from src.tools.retrieval import query_bodi_database, find_activity_cliff, query_by_mechanism
from src.agent.model_factory import ModelFactory
from src.tools.structure import analyze_structural_reorganization
from src.tools.molecule_image import draw_molecule, compare_molecule_structures


# 定义状态
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# 工具列表 (固定)
TOOLS = [
    check_hammett, 
    analyze_structural_reorganization, 
    query_bodi_database,
    find_activity_cliff,
    query_by_mechanism,
    draw_molecule,
    compare_molecule_structures
]


class AgentBuilder:
    """
    Agent 构建器 - 支持动态切换模型
    """
    
    _instance = None
    _current_model = None
    _app = None
    
    @classmethod
    def get_app(cls, model_name: str = None):
        """
        获取 Agent 应用实例
        如果模型变化则重新构建
        """
        if model_name is None:
            model_name = os.getenv("CURRENT_MODEL", "qwen_dev")
        
        # 如果模型相同且已有实例，直接返回
        if cls._app is not None and cls._current_model == model_name:
            return cls._app
        
        # 重新构建 Agent
        print(f"[AgentBuilder] 构建 Agent，模型: {model_name}")
        
        try:
            llm = ModelFactory.get_model(model_name, temperature=0)
            llm_with_tools = llm.bind_tools(TOOLS)
        except Exception as e:
            print(f"[AgentBuilder] 模型加载失败: {e}")
            raise
        
        # 定义节点
        def reasoner_node(state: AgentState):
            return {"messages": [llm_with_tools.invoke(state["messages"])]}
        
        # 构建图
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", reasoner_node)
        workflow.add_node("tools", ToolNode(TOOLS))
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "agent")
        
        # 编译
        cls._app = workflow.compile()
        cls._app.recursion_limit = 50
        cls._current_model = model_name
        
        return cls._app
    
    @classmethod
    def get_current_model(cls):
        return cls._current_model


# 兼容旧代码 - 默认 app 实例
def get_default_app():
    return AgentBuilder.get_app()


# 为了兼容现有导入
app = None  # 延迟初始化


def init_app():
    global app
    app = AgentBuilder.get_app()
    return app


# 模块加载时初始化
init_app()


# 运行测试函数
def run_interactive():
    print("🧪 BodiMechanist Initialized. Type 'quit' to exit.")
    print(f"🤖 Brain: {AgentBuilder.get_current_model()} | 🛠️ Tools: 7")
    
    agent = AgentBuilder.get_app()
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit"]:
            break
            
        inputs = {"messages": [HumanMessage(content=user_input)]}
        for event in agent.stream(inputs, stream_mode="values"):
            last_message = event["messages"][-1]
            last_content = last_message.content
            
            if last_message.type == "ai":
                if not last_content and last_message.tool_calls:
                    print(f"🤖 (Calling Tool): {last_message.tool_calls[0]['name']}")
                else:
                    print(f"🤖 BodiMechanist: {last_content}")
            elif last_message.type == "tool":
                print(f"🔧 Tool Output: {last_content[:100]}...")


if __name__ == "__main__":
    run_interactive()