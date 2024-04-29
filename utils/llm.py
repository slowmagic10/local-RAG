
import os,time
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.llms import CTransformers
from langchain_together import Together


load_dotenv()

class LLM:
    def __init__(self) -> None:
        self.local_model_path = os.getenv('MODEL_PATH')

    def get_llm(self) -> CTransformers:
        
        start = time.time()

        llm = CTransformers(model=self.local_model_path,
                            config={'max_new_tokens': 4096,
                                'temperature': 0.00,
                                'context_length': 4096})
        end = time.time()
        print('Time to load the model:',end-start)
        return llm

    def get_llm_together(self)-> Together:

        llm = Together(model='mistralai/Mistral-7B-Instruct-v0.2',
                        temperature=0.7,
                        max_tokens=128,
                        top_k=1,
                        together_api_key="b29c9a56f3a884d4296c48a75b24a737a6bd4c1d9c59c3f768e866e340e430d1"
                        )
        return llm


