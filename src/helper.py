from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from huggingface_hub import login
import os
from llama_index.core import VectorStoreIndex


# Data Loading
documents = SimpleDirectoryReader(input_dir="./data", required_exts=".pdf").load_data()

# Llama model loading
from dotenv import load_dotenv
load_dotenv()

Llama_1B_ACCESS_TOKEN = os.environ.get('Llama_ACCESS_TOKEN')
os.environ["HF_KEY"] = Llama_1B_ACCESS_TOKEN
login(token=os.environ.get('HF_KEY'),add_to_git_credential=True)

# Loading Embedding Model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME, embed_batch_size=8)

index = VectorStoreIndex.from_documents(documents, embed_model = embed_model)
    

