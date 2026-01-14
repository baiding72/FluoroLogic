import os
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

# 导入你的工具
from src.tools.electronic import check_hammett
from src.tools.retrieval import query_bodi_database, find_activity_cliff, query_by_mechanism
from src.agent.model_factory import ModelFactory
from src.tools.structure import analyze_structural_reorganization

# 1. 定义状态 (State)
# LangGraph 需要定义一个状态对象，这里我们只存储消息列表
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 2. 准备 LLM 和 工具
# 这里记得去 .env 检查一下你的 CURRENT_MODEL 是什么
# 建议先用 API (qwen_pro) 测试，因为 Ollama 可能第一次调用 tool 会失败
llm = ModelFactory.get_model(os.getenv("CURRENT_MODEL", "qwen_dev"), temperature=0)

# 包含新的检索工具
tools = [
    check_hammett, 
    analyze_structural_reorganization, 
    query_bodi_database,
    find_activity_cliff,      # 新增: Activity Cliff 检测
    query_by_mechanism        # 新增: 按机理检索
]
llm_with_tools = llm.bind_tools(tools) # 这一步是关键，把工具绑定到模型

# 3. 定义节点 (Nodes)
def reasoner_node(state: AgentState):
    """思考节点：LLM 决定是说话还是调用工具"""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 4. 构建图 (Graph)
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", reasoner_node)
workflow.add_node("tools", ToolNode(tools)) # LangGraph 内置的工具执行节点

# 添加边 (Edges)
workflow.add_edge(START, "agent")
# 关键逻辑：agent 节点执行完后，检查是否需要调用工具
# 如果 LLM 返回 tool_calls，则跳转到 "tools" 节点，否则跳转到 END
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent") # 工具执行完，结果返回给 agent 继续思考

# 编译图
app = workflow.compile()

# 5. 运行测试函数
def run_interactive():
    print("🧪 BodiMechanist Initialized. Type 'quit' to exit.")
    print(f"🤖 Brain: {os.getenv('CURRENT_MODEL')} | 🛠️ Tools: Electronic, Retrieval")
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit"]:
            break
            
        # 流式输出
        inputs = {"messages": [HumanMessage(content=user_input)]}
        for event in app.stream(inputs, stream_mode="values"):
            # 打印最后一条消息
            last_message = event["messages"][-1]
            last_content = last_message.content
            
            # 简单的打印美化
            if last_message.type == "ai":
                if not last_content and last_message.tool_calls:
                    print(f"🤖 (Calling Tool): {last_message.tool_calls[0]['name']}")
                else:
                    print(f"🤖 BodiMechanist: {last_content}")
            elif last_message.type == "tool":
                print(f"🔧 Tool Output: {last_content[:100]}...") # 只打印前100字

if __name__ == "__main__":
    run_interactive()