项目架构

```
FluoroLogic/
├── data/                        # 数据层
│   ├── raw_dft/                 # 126个分子的原始 Gaussian 文件 (.log/.gjf)
│   ├── processed/               # 清洗后的结构化数据 (molecules.json)
│   ├── knowledge_base/          # 向量数据库存储 (ChromaDB/Milvus 本地文件)
│   └── hammett_table.csv        # Hammett 常数查询表
├── src/                         # 源代码层
│   ├── agent/                   # Agent 核心逻辑
│   │   ├── __init__.py
│   │   ├── core.py              # 初始化 LLM 和 AgentExecutor
│   │   └── prompts.py           # System Prompt 和思维链模板
│   ├── tools/                   # 工具箱 (Toolkits)
│   │   ├── __init__.py
│   │   ├── steric.py            # Tool A: 3D 空间效应分析 (二面角)
│   │   ├── electronic.py        # Tool B: 电子效应分析 (Hammett)
│   │   ├── retrieval.py         # Tool C: 相似性检索 (RAG)
│   │   ├── optics.py            # Tool D: 光学性质 (sTDA-xTB)
│   │   └── visualization.py     # Tool E: 绘图工具 (RDKit/MolScribe)
│   ├── utils/                   # 通用辅助函数
│   │   ├── rdkit_utils.py       # RDKit 封装
│   │   └── file_reader.py       # Gaussian 文件解析
│   └── config.py                # 环境变量配置 (API Key, 路径)
├── notebooks/                   # 实验和测试用的 Jupyter Notebooks
├── app.py                       # 前端入口 (Streamlit/Gradio)
├── requirements.txt             # 依赖包
└── .env                         # 存放 API Key
```