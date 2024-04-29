# LLM_RAG

python3 -m venv .venv  
source .venv/bin/activate  
which python  
python3 -m pip install --upgrade pip  

- model_path = https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF
- Embedding Model: https://huggingface.co/BAAI/bge-base-en-v1.5
- Hugging Face(ChatInterface) -> https://www.gradio.app/docs/chatinterface

步骤：  
1.配置好环境变量如模型位置，数据库存储位置、apikey等；  
2.在inference.py里面可以选择用本地ollama模型或者通过apikey使用chatgpt或是togetherai,有部分代码有保留，没有的话用langchain简单配置一下即可；    
3.先在note_fastapi.iqynb中建立RAG数据库再使用;  
4.运行 python main.py


