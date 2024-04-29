from utils.llm import LLM
from utils.build_rag import RAG
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import os
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate

def predict(qns:str,history=None)->str:
    together=LLM().get_llm_together()
    model = ChatOpenAI()
    llama2 = Ollama(model="llama2:13b", temperature=0)
    llama3 = Ollama(model="llama3")
    retriever = RAG().get_retriever()
    output_parser = StrOutputParser()
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """

    template3="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    prompt = ChatPromptTemplate.from_template(template)
    chain =  llama2 | StrOutputParser()
    retrieval_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llama3
        | StrOutputParser()
        )
    
    result = retrieval_chain.invoke(qns)
    
    
    return result


