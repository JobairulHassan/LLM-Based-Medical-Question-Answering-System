# LLM Based Medical Question Answering System with RAG
This project implements a Medical Question Answering (QA) system that leverages the **mT5 model** fine-tuned on a **Medical QA dataset**. The system integrates **Retrieval-Augmented Generation (RAG)** with a **Streamlit-based GUI** to provide real-time answers to medical queries. The project uses the **Hugging Face** `lavita/medical-qa-datasets` dataset for fine-tuning and uses a combination of **FAISS** for efficient search and **mT5** for generating answers.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Setup and Installation](#setup-and-installation)
4. [Dataset](#dataset)
5. [Training Process](#training-process)
6. [Running the Application](#running-the-application)
7. [Dependencies](#dependencies)

## Overview

This system is designed to allow users to ask medical-related questions and receive relevant answers from a fine-tuned mT5 model. The model's answers are enhanced by a retrieval system based on **RAG** architecture, which first retrieves relevant context from the **FAISS** index of medical data before generating an answer. The entire system is accessible through a user-friendly **Streamlit** interface.

## Features

- **Fine-tuned mT5**: Trained on **Hugging Face** `lavita/medical-qa-datasets` dataset(top 25K data) to generate context-aware, relevant answers.
- **Retrieval-Augmented Generation (RAG)**: Improves the quality of generated answers by fetching relevant context from a FAISS index.
- **Streamlit GUI**: An interactive web-based interface for users to input medical queries and get real-time answers.
- **FAISS Indexing**: Efficient nearest neighbor search for context retrieval.

## Setup and Installation

- **Install Dependencies:** `pip install -r requirements.txt`
- **Download Files:** clone/Download files and run cell by cell.

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
