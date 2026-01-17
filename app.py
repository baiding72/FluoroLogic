"""
FluoroLogic Agent 对话界面
ChatGPT 风格布局：全屏聊天 + 右侧工具栏
"""

import gradio as gr
import os
import sys

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# 导入 Agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.agent.core import AgentBuilder

# 导入分子图像工具
try:
    from src.tools.molecule_image import image_to_smiles
    IMAGE_TOOLS_AVAILABLE = True
except ImportError:
    IMAGE_TOOLS_AVAILABLE = False

# 导入系统提示
from src.agent.prompts import BODIMECHANIST_SYSTEM_PROMPT as SYSTEM_PROMPT


def process_image(image_path: str) -> str:
    if not image_path or not IMAGE_TOOLS_AVAILABLE:
        return ""
    smiles, method = image_to_smiles(image_path)
    if smiles:
        return f"[识别到分子: {smiles} (使用 {method})]"
    return ""


def chat_stream(message: str, history: list, image=None):
    """流式对话"""
    image_info = ""
    if image:
        image_info = process_image(image)
        if image_info:
            message = f"{image_info}\n\n用户问题: {message}"
    
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for h in history:
        if isinstance(h, dict):
            if h.get("role") == "user":
                messages.append(HumanMessage(content=h.get("content", "")))
            elif h.get("role") == "assistant":
                messages.append(AIMessage(content=h.get("content", "")))
    messages.append(HumanMessage(content=message))
    
    response_parts = []
    
    try:
        # 获取当前模型的 Agent
        agent_app = AgentBuilder.get_app()
        
        for event in agent_app.stream({"messages": messages}, stream_mode="values"):
            last_message = event["messages"][-1]
            
            if last_message.type == "ai":
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    for tc in last_message.tool_calls:
                        tool_info = f"🔧 `{tc['name']}`"
                        response_parts.append(tool_info)
                        yield " → ".join(response_parts) + " ⏳"
                
                if last_message.content:
                    response_parts.append(f"\n\n{last_message.content}")
                    yield "".join(response_parts)
                    
            elif last_message.type == "tool":
                pass  # 工具返回不显示，减少噪音
                
    except Exception as e:
        yield f"❌ 错误: {str(e)}"


# 加载 CSS 样式
CSS_FILE = os.path.join(os.path.dirname(__file__), "static", "style.css")
if os.path.exists(CSS_FILE):
    with open(CSS_FILE, "r", encoding="utf-8") as f:
        CUSTOM_CSS = f.read()
else:
    CUSTOM_CSS = ""


def create_ui():
    """创建 ChatGPT 风格界面"""
    
    model_name = os.getenv("CURRENT_MODEL", "qwen_dev")
    
    # 使用 Base 主题并指定标准字体
    custom_theme = gr.themes.Base(
        font=["Arial", "Helvetica", "sans-serif"],
        font_mono=["Consolas", "Monaco", "monospace"]
    )
    
    with gr.Blocks(title="FluoroLogic", theme=custom_theme) as demo:
        
        gr.Markdown("""
        # 🔬 FluoroLogic
        **BODIPY 分子电位预测 Agent**
        """)
        
        with gr.Row():
            # ========== 左侧：聊天区域 ==========
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=800,
                    show_label=False,
                    container=True
                )
            
            # ========== 右侧：工具栏 ==========
            with gr.Column(scale=1):
                gr.Markdown("### 💬 发送消息")
                
                msg = gr.Textbox(
                    placeholder="输入你的问题...",
                    lines=3,
                    show_label=False
                )
                
                with gr.Row():
                    submit_btn = gr.Button("发送", variant="primary", size="lg")
                    clear_btn = gr.Button("清空", size="lg")
                
                gr.Markdown("---")
                gr.Markdown("### 📷 上传分子图片")
                
                image_input = gr.Image(
                    type="filepath",
                    height=120,
                    show_label=False
                )
                
                gr.Markdown("---")
                gr.Markdown("### 💡 快捷问题")
                
                # 示例按钮 - 点击自动填入
                examples = [
                    "BE_24NO2 的电位是多少？",
                    "分析 -CF3 的电子效应",
                    "对比 BE_OMe 和 BE_CN",
                    "找 flattening 类型分子",
                    "绘制咔唑 c1ccc2c(c1)[nH]c1ccccc12"
                ]
                
                for ex in examples:
                    gr.Button(ex, size="sm").click(
                        lambda x=ex: x, None, msg
                    )
                
                gr.Markdown("---")
                gr.Markdown("### ⚙️ 系统设置")
                
                # 模型选择下拉框
                available_models = [
                    ("Qwen Plus (开发)", "qwen_dev"),
                    ("Qwen Max (高级)", "qwen_pro"),
                    ("DeepSeek", "deepseek"),
                    ("Gemini 3 Pro", "gemini"),
                    ("GPT-4o", "gpt4")
                ]
                
                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    value=model_name,
                    label="选择模型",
                    interactive=True
                )
                
                gr.Markdown(f"""
                - **工具数**: 7 个
                - **图像识别**: {'✓' if IMAGE_TOOLS_AVAILABLE else '✗'}
                """)
        
        # ========== 事件处理 ==========
        def user_input(message, history, image):
            if not message.strip() and image is None:
                return "", history, None
            
            user_msg = message
            if image:
                user_msg = f"[📷 上传图片]\n{message}"
            
            history = history + [{"role": "user", "content": user_msg}]
            return "", history, None
        
        def bot_response(history, image):
            if not history:
                return history
            
            last_user_msg = ""
            for msg in reversed(history):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    last_user_msg = msg.get("content", "")
                    break
            
            if not last_user_msg:
                return history
            
            clean_msg = str(last_user_msg).replace("[📷 上传图片]\n", "")
            
            history = history + [{"role": "assistant", "content": ""}]
            
            for partial in chat_stream(clean_msg, history[:-1], image):
                history[-1] = {"role": "assistant", "content": partial}
                yield history
        
        # 绑定事件
        msg.submit(user_input, [msg, chatbot, image_input], [msg, chatbot, image_input]).then(
            bot_response, [chatbot, image_input], chatbot
        )
        submit_btn.click(user_input, [msg, chatbot, image_input], [msg, chatbot, image_input]).then(
            bot_response, [chatbot, image_input], chatbot
        )
        clear_btn.click(lambda: (None, [], None), None, [msg, chatbot, image_input])
        
        # 模型切换处理 - 立即生效
        def on_model_change(new_model, history):
            os.environ["CURRENT_MODEL"] = new_model
            try:
                AgentBuilder.get_app(new_model)  # 重新构建 Agent
                # 添加系统消息
                history = history + [{"role": "assistant", "content": f"✅ 已切换到模型: **{new_model}**"}]
                return history
            except Exception as e:
                history = history + [{"role": "assistant", "content": f"❌ 切换失败: {str(e)}"}]
                return history
        
        model_dropdown.change(
            on_model_change, 
            [model_dropdown, chatbot], 
            [chatbot]
        )
    
    return demo


if __name__ == "__main__":
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["no_proxy"] = "localhost,127.0.0.1"
    
    print("=" * 50)
    print("FluoroLogic - ChatGPT 风格界面")
    print("=" * 50)
    print(f"模型: {os.getenv('CURRENT_MODEL', 'qwen_dev')}")
    
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=None, share=False)
