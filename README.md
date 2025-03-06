# LLM Based Medical Question Answering System with RAG
This project implements a Medical Question Answering (QA) system that leverages the **Llama3.2-1B model** fine-tuned on a **Medical Encyclopedia Book**. The system integrates **Retrieval-Augmented Generation (RAG)** with a **GUI using Flask** to provide real-time answers to medical queries. The project uses the **PDF** dataset for fine-tuning and uses a combination of **Vector Embedding** for efficient search and **llama model** for generating answers.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Setup and Installation](#setup-and-installation)
4. [Dataset](#dataset)
5. [Training Process](#training-process)
6. [Running the Application](#running-the-application)
7. [Dependencies](#dependencies)
8. [Sample Output](#sample-output)

## Overview

This system is designed to allow users to ask medical-related questions and receive relevant answers using `Llama3.2-1B` model. The model's answers are enhanced by a retrieval system based on **RAG** architecture, which first retrieves top 3 relevant context using the cosine similarity index of medical data before generating an answer. The entire system is accessible through a user-friendly interface.

## Features

- **Llama3.2-1B model**: One of the robust and light weight model to generate context-aware, relevant answers.
- **Retrieval-Augmented Generation (RAG)**: Improves the quality of generated answers by fetching relevant context.
- **GUI using Flask**: An interactive web-based interface for users to input medical queries and get real-time answers.
- **Pinecone**: Store vector database to cloud platform.

## Setup and Installation

- **Clone Repository:** `Project Repo: https://github.com/JobairulHassan/LLM-Based-Medical-Question-Answering-System.git`
- **Create a conda environment after opening the repo:** `conda create -n medicalchatbot python=3.11.11 -y`
`conda activate medicalchatbot`
- **Install Dependencies:** `pip install -r requirements.txt`
- **Create a .env file in the root directory and add your Pinecone credentials and Llama3.2-1B access token as follows:** 
    ```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
Llama_ACCESS_TOKEN = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```
- **Download Model:** Download `Llama3.2-1B` model 

## Dataset

Dataset link: [Click Here](https://huggingface.co/datasets/lavita/medical-qa-datasets)


## Training Process

The `mT5` model is fine-tuned using the **Medical-QA dataset**. The training process involves the following steps:

- **Load and preprocess the dataset**: The dataset is split into training, testing and validation sets.
- **Fine-tuning mT5**: The model is fine-tuned on the dataset using custom optimizer and data collator.
- **Saving the model**: After fine-tuning and training the model is saved for inference use.

## Running the Application

Once the model is fine-tuned, you can start the Streamlit-based web application by running the following command: `streamlit run app.py`

## Dependencies

The following libraries are required for the project:
```
streamlit - For the web interface.
transformers - For using the mT5 model.
datasets - For accessing the Hugging Face dataset.
faiss - For efficient nearest neighbor search.
torch - For deep learning tasks with PyTorch.
sentence-transformers - For generating embeddings for RAG-based retrieval.
```
## Sample Output
![Screenshot 2024-12-06 152918](https://github.com/user-attachments/assets/04ca30e2-fa6e-4f7c-bf6f-573ea1211f8a)
