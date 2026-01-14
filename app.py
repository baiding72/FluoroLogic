"""
FluoroLogic Agent 对话界面
直接调用 src/agent/core.py 中的 Agent
"""

import gradio as gr
import os
import sys

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# 导入 Agent (来自 src/agent/core.py)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.agent.core import app as agent_app

# 系统提示
SYSTEM_PROMPT = """你是 BodiMechanist，一个专注于 BODIPY 分子电化学性质分析的 AI 专家。

你可以使用以下工具:
1. **check_hammett**: 查询取代基的 Hammett σ 值，分析电子效应
2. **analyze_structural_reorganization**: 分析分子还原时的结构重组
3. **query_bodi_database**: 从数据库检索分子信息
4. **find_activity_cliff**: 找到结构相似但电位差异大的对照案例
5. **query_by_mechanism**: 按机理类型检索分子

分析电位时，请考虑:
- 电子效应 (Hammett σ): 吸电子基团使电位更正，供电子基团使电位更负
- 空间效应: Meso 二面角影响共轭程度
- 结构重组: 还原时的构象弛豫可能额外稳定阴离子

请用中文回答，并给出科学推理过程。
"""


def chat_stream(message: str, history: list):
    """与 Agent 对话 - 流式输出"""
    # 构建消息历史
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    
    for h in history:
        if isinstance(h, dict):
            if h.get("role") == "user":
                messages.append(HumanMessage(content=h.get("content", "")))
            elif h.get("role") == "assistant":
                messages.append(AIMessage(content=h.get("content", "")))
    
    messages.append(HumanMessage(content=message))
    
    # 流式输出
    response_parts = []
    
    try:
        for event in agent_app.stream({"messages": messages}, stream_mode="values"):
            last_message = event["messages"][-1]
            
            if last_message.type == "ai":
                # 工具调用信息
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    for tc in last_message.tool_calls:
                        tool_info = f"🔧 **调用工具**: `{tc['name']}`"
                        if tc.get('args'):
                            args_preview = str(tc['args'])[:100]
                            tool_info += f"\n   参数: `{args_preview}`"
                        response_parts.append(tool_info)
                        yield "\n".join(response_parts) + "\n\n⏳ 执行中..."
                
                # AI 回复
                if last_message.content:
                    response_parts.append(f"\n---\n\n{last_message.content}")
                    yield "\n".join(response_parts)
                    
            elif last_message.type == "tool":
                # 工具返回结果
                tool_result = last_message.content[:200] + "..." if len(last_message.content) > 200 else last_message.content
                response_parts.append(f"📋 **工具返回**: {tool_result}")
                yield "\n".join(response_parts) + "\n\n⏳ 分析中..."
                
    except Exception as e:
        yield f"❌ 错误: {str(e)}"


def respond(message, chat_history):
    """处理用户输入"""
    if not message.strip():
        return "", chat_history
    
    # 添加用户消息
    chat_history.append({"role": "user", "content": message})
    
    # 流式获取回复
    bot_response = ""
    for partial in chat_stream(message, chat_history[:-1]):
        bot_response = partial
    
    chat_history.append({"role": "assistant", "content": bot_response})
    return "", chat_history


def create_ui():
    """创建 Gradio 界面"""
    
    model_name = os.getenv("CURRENT_MODEL", "qwen_dev")
    
    with gr.Blocks(title="FluoroLogic - BODIPY Agent") as demo:
        
        gr.Markdown(f"""
        # 🔬 FluoroLogic
        ## BODIPY 分子还原电位预测 Agent
        
        **模型**: `{model_name}` | **工具**: 电子效应 · 结构重组 · 数据库检索 · Activity Cliff · 机理检索
        
        ---
        
        直接与 Agent 对话，Agent 会自动调用合适的工具进行分析。工具调用过程实时显示。
        """)
        
        chatbot = gr.Chatbot(height=500)
        
        msg = gr.Textbox(
            label="消息",
            placeholder="输入你的问题...",
            lines=2
        )
        
        with gr.Row():
            submit_btn = gr.Button("发送", variant="primary")
            clear_btn = gr.Button("清空对话")
        
        # 丰富的示例问题
        gr.Markdown("### 💡 示例问题")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("**基础查询**")
                gr.Examples(
                    examples=[
                        ["BE_24NO2 的还原电位是多少？"],
                        ["BE_OMe 的电位有什么特点？"],
                    ],
                    inputs=msg,
                    label=""
                )
            
            with gr.Column():
                gr.Markdown("**电子效应分析**")
                gr.Examples(
                    examples=[
                        ["分析三氟甲基 (-CF3) 的电子效应"],
                        ["硝基和甲氧基哪个更吸电子？"],
                    ],
                    inputs=msg,
                    label=""
                )
            
            with gr.Column():
                gr.Markdown("**对比分析**")
                gr.Examples(
                    examples=[
                        ["为什么 BE_OMe 电位比 BE_CN 更负？"],
                        ["找一个和 BE_Br 结构相似但电位不同的分子"],
                    ],
                    inputs=msg,
                    label=""
                )
            
            with gr.Column():
                gr.Markdown("**机理检索**")
                gr.Examples(
                    examples=[
                        ["找出所有 flattening 类型的分子"],
                        ["有哪些分子的 Meso 二面角很大？"],
                    ],
                    inputs=msg,
                    label=""
                )
        
        # 事件绑定
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
        clear_btn.click(lambda: (None, []), None, [msg, chatbot])
        
        gr.Markdown("""
        ---
        **FluoroLogic** | 基于 LangGraph Agent | 工具调用实时显示
        """)
    
    return demo


if __name__ == "__main__":
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["no_proxy"] = "localhost,127.0.0.1"
    
    print("=" * 50)
    print("FluoroLogic - BODIPY Agent")
    print("=" * 50)
    print(f"模型: {os.getenv('CURRENT_MODEL', 'qwen_dev')}")
    print("工具: check_hammett, analyze_structural_reorganization,")
    print("      query_bodi_database, find_activity_cliff, query_by_mechanism")
    
    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=None,  # 自动找可用端口
        share=False
    )
