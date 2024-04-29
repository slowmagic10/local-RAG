import uvicorn
import os
import gradio as gr
from utils.inference import predict
from api import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI
from enum import Enum

# 加载环境变量
load_dotenv()

#初始化FastAPI应用程序
app = FastAPI()

#定义请求和响应模型
class Request(BaseModel):
    prompt : str

class Response(BaseModel):
    response : str

#定义API端点
@app.post("/",response_model=Response)
async def predict_api(prompt:Request):
    response = predict(Request.prompt)
    return response

#创建Gradio界面
demo = gr.ChatInterface(
    fn=predict,
    textbox=gr.Textbox(
        placeholder="Ask a question", container=False,lines=1,scale=8
    ),
    title="LLM App",
    undo_btn="Delete Previous",
    clear_btn="Clear",
)

#定义打招呼API端点
@app.get("/hello/{name}")
async def hello(name:str):
    return f"Hello {name} "


models = {
    'LLMs' : ['OpenAI', 'Mistral'],
    'NLP' : ['Bert', 'RoBerta'],
    'ML' : ['Xgboost', 'Catboost']
}

#定义一个枚举类
class AvailableModel(str, Enum):
    LLMs = "LLMs"
    NLP = "NLP"
    ML = "ML"

#创建另一个API端点，接受枚举类作为参数    
@app.get("/get_models/{usecase}")
async def get_items(usecase: AvailableModel):
    return models.get(usecase)

#定义一个数据模型
class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

#创建一个API端点，接受数据模型作为参数
@app.post("/items/")
async def create_item(item: Item):
    return item

#将Gradio界面嵌入到FastAPI应用程序中
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    # mounting at the root path
    uvicorn.run(
        app="main:app",
        host=os.getenv("UVICORN_HOST"),  
        port=int(os.getenv("UVICORN_PORT"))
    )


