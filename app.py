from flask import Flask, render_template, jsonify, request
from src.helper import embed_model, index
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from llama_index.core import Settings
from src.prompt import *
import os
import torch
from llama_index.llms.huggingface import HuggingFaceLLM

app = Flask(__name__)

load_dotenv()

# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')


embeddings = embed_model

# Customize embed model setting
Settings.embed_model = embed_model
Settings.chunk_size = 512
Settings.chunk_overlap = 30


query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

LLM_MODEL_NAME = "meta-llama/Llama-3.2-1B"

# To import models from HuggingFace directly
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=512,
    generate_kwargs={"temperature": 0.7,"do_sample":False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=LLM_MODEL_NAME,
    model_name=LLM_MODEL_NAME,
    device_map="auto",
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}
)

Settings.llm = llm

query_engine = index.as_query_engine(llm=llm, similarity_top_k=3)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=query_engine.query(input)
    print("Response : ", result.response)
    return str(result.response)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)