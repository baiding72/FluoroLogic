import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Literal

# 定义支持的模型类型
ModelType = Literal["deepseek", "qwen_dev", "qwen_pro",  "gemini", "gpt4"]


# 加载 .env 文件
load_dotenv()

class ModelFactory:
    """
    模型工厂：统一管理不同厂商的大模型实例
    """
    
    @staticmethod
    def get_model(model_type: ModelType = "deepseek", temperature: float = 0.1):
        """
        根据名称返回 LangChain ChatModel 对象
        """
        model_type = model_type.lower()
        
        if model_type == "deepseek":
            return ChatOpenAI(
                model="deepseek-chat",
                openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
                openai_api_base=os.getenv("DEEPSEEK_BASE_URL"),
                temperature=temperature,
                max_tokens=4096
            )
            
        elif model_type == "qwen_dev":
            return ChatOpenAI(
                # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://www.alibabacloud.com/help/zh/model-studio/getting-started/models
                model="qwen-plus", 
                openai_api_key=os.getenv("QWEN_API_KEY"),
                openai_api_base=os.getenv("QWEN_BASE_URL"),
                temperature=temperature
            )
        
        elif model_type == "qwen_pro":
            return ChatOpenAI(
                model="qwen3-max", 
                openai_api_key=os.getenv("QWEN_API_KEY"),
                openai_api_base=os.getenv("QWEN_BASE_URL"),
                temperature=temperature
            )
            
        elif model_type == "gemini":
            return ChatGoogleGenerativeAI(
                model="gemini-3-pro-preview", 
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=temperature,
                convert_system_message_to_human=True # 兼容性设置
            )
            
        elif model_type == "gpt4":
            return ChatOpenAI(
                model="gpt-4o",
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                temperature=temperature
            )
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

# 测试代码
if __name__ == "__main__":
    try:
        # 测试 DeepSeek
        # model = ModelFactory.get_model("deepseek")
        # print("Testing DeepSeek:", model.invoke("你好，你是谁？").content)
        
        # 测试 Qwen
        model = ModelFactory.get_model("qwen")
        print("Testing Qwen:", model.invoke("介绍一下BODIPY").content)
        
    except Exception as e:
        print(f"Error: {e}")