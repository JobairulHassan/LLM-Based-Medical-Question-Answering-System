# LLM Based Medical Question Answering System with RAG
This project implements a Medical Question Answering (QA) system that leverages the **Llama3.2-1B model** on a **Medical Encyclopedia Book**. The system integrates **Retrieval-Augmented Generation (RAG)** , **LlamaIndex**, **LangChain** with a **GUI using Flask** to provide real-time answers to medical queries. The project uses the **PDF** dataset for fine-tuning and uses a combination of **Vector Embedding** for efficient search and **llama model** for generating answers.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Setup and Installation](#setup-and-installation)
4. [Dataset](#dataset)
5. [Dependencies](#dependencies)
6. [License](#license)

## Overview

This system is designed to allow users to ask medical-related questions and receive relevant answers using `Llama3.2-1B` model. The model's answers are enhanced by a retrieval system based on **RAG** architecture, which first retrieves top 3 relevant context using the cosine similarity index of medical data before generating an answer. The entire system is accessible through a user-friendly interface.

## Features

- **Llama3.2-1B model**: One of the robust and light weight model to generate context-aware, relevant answers.
- **Retrieval-Augmented Generation (RAG)**: Improves the quality of generated answers by fetching relevant context.
- **GUI using Flask**: An interactive web-based interface for users to input medical queries and get real-time answers.
- **LlamaIndex**: Help to excel at indexing large datasets and retrieving relevant information quickly and accurately.
- **Pinecone**: Store vector database to cloud platform.

## Setup and Installation

- **Clone Repository:** Project repo
    ```
      https://github.com/JobairulHassan/LLM-Based-Medical-Question-Answering-System.git
    ```
- **Create a conda environment after opening the repo:** `
    ```
        conda create -n medicalchatbot python=3.11.11 -y
        conda activate medicalchatbot
    ```
- **Install Dependencies:**
    ```
      pip install -r requirements.txt
    ```
- **Create a .env file in the root directory and add your Pinecone credentials and Llama3.2-1B access token as follows:** 
    ```
        PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        Llama_ACCESS_TOKEN = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    ```
- **Download Model:** Download `Llama3.2-1B` model using `Hugging Face` provided commands.
- **Run app from command Line:** command to run the app:
    ```
      python app.py
    ```

## Dataset

[PDF File](https://github.com/JobairulHassan/LLM-Based-Medical-Question-Answering-System/blob/main/data/Encyclopedia%20of%20Medicine.pdf)

## Dependencies

- Python
- LlamaIndex
- LangChain
- Flask
- Meta Llama3.2
- Pinecone

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
