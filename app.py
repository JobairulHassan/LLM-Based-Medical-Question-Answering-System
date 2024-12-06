import faiss
import torch
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
main_data = pd.read_csv('Dataset.csv')
main_data = main_data[:25000] 
main_data = main_data.dropna()

def load_model():
    model = MT5ForConditionalGeneration.from_pretrained('Medical_QA_mT5_Model_v2.pt')
    tokenizer = MT5Tokenizer.from_pretrained('Medical_QA_mT5_Tokenizer_v2.json')
    return model, tokenizer

def retrieve_documents(query, k=5):
    index = faiss.read_index("medical_qa_index.faiss")
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    
    # Get contexts based on indices retrieved
    retrieved_contexts = [main_data['input'][i] for i in indices[0]]
    return retrieved_contexts

# Function to generate a response from the mT5 model
def generate_answer(model, tokenizer, user_input):
    retrieved_contexts = retrieve_documents(user_input)
    
    # Combine retrieved contexts into one string for input to the model
    context_input = " ".join(retrieved_contexts)
    
    input_text = f"Context: {context_input}\nQuestion: {user_input}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(**inputs)
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Streamlit app code
def main():
    st.title("Medical-QA APP")

    # Load the model and tokenizer
    model, tokenizer = load_model()

    if "history" not in st.session_state:
        st.session_state.history = []

    # Input area for the user to ask a question
    user_input = st.text_area("Ask a question", key="input_area")

    if st.button("Submit"):
        if user_input:
            # Get model response
            model_output = generate_answer(model, tokenizer, user_input)

            # Save user input and model response to history
            st.session_state.history.append({"user": user_input, "model": model_output})

    # Display chat history
    for chat in st.session_state.history:
        # User question aligned to the right
        st.markdown(f"<p style='text-align: right; color: black; font-size: 16px;'><b>User:</b> {chat['user']}</p>", unsafe_allow_html=True)

        # Model response aligned to the left
        st.markdown(f"<p style='text-align: left; color: black; font-size: 16px;'><b>Response:</b> {chat['model']}</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()