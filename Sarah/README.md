# Pipeline 2: Multimodel RAG (Part taking image from user and retriving description or story )

This folder contains my work (**Sarah**) in **Pipeline 2**.

## Notebook 1

### `Ankh_RAG_Clean.ipynb`

This notebook implements an end-to-end # Ankh RAG — Egyptian Heritage Tour Guide

Ankh RAG is an advanced Multi-modal Retrieval-Augmented Generation (RAG) system designed to provide interactive historical insights and personalized storytelling for Egyptian monuments and artifacts. 

The system bridges visual input with semantic knowledge, allowing users to upload an image of a site and receive either a professional historical explanation or a creative, persona-based narrative.

## System Architecture

The pipeline follows these core stages:

1.  **Visual Embedding:** Images are processed using **DINOv2** (`facebook/dinov2-base`) to generate 768-dimensional visual vectors.
2.  **Vector Retrieval:** We use **Qdrant** as an in-memory vector database to perform high-speed Cosine similarity searches between user-uploaded images and our pre-computed artifact collection.
3.  **Knowledge Synthesis:** The retrieved text chunks are processed by **Qwen1.5-0.5B-Chat**, a lightweight LLM optimized to generate accurate, context-aware responses.

## Key Features

*   **Dual Mode Interaction:**
    *   `Description Mode`: Provides academic, structured historical summaries as an expert Egyptologist tour guide.
    *   `Story Mode`: Generates personalized narratives tailored to specific personas (e.g., child, adult) to enhance engagement.
*   **Interactive Interface:** Built with **Gradio** for real-time artifact analysis and debugging of retrieved contexts.
*   **Optimized Performance:** Uses `accelerate` and `torch` to handle heavy model inference on T4 GPU environments efficiently.

## Input & Output

*   **Input:** 
    *   An image file (PIL compatible: JPG, PNG).
    *   User selections: `Mode` (Description vs. Story) and `Persona` (for Story mode).
*   **Output:** 
    *   **Generated Response:** The final historical or storytelling output.
    *   **Retrieved Context:** The raw historical facts retrieved from the Qdrant database, ensuring transparency and accuracy.
    *   **Embedding Info:** Technical details (DINOv2 vector dimensions and database scale).

## Retrieval Testing
The system includes a sanity test to verify the integrity of the embedding pipeline. By mapping an image to its corresponding vector in our Qdrant collection (`egmm_corpus_multimodal`), the system confirms the successful retrieval of historical chunks from the EgMM-Corpus.

## Technical Dependencies
- `qdrant-client`: Vector database management.
- `transformers` & `accelerate`: Model loading and GPU optimization.
- `gradio`: Frontend interaction.
- `Pillow`: Image preprocessing.
.

______________________________________________________________________________________________________________________________________________________________________________

## Notebook 2 
### `Ankh_RAG_SS.ipynb`


This notebook implements a Multi-modal Retrieval-Augmented Generation (RAG) system designed to link text descriptions with images of Egyptian monuments. The system leverages advanced deep learning architectures for vector representation and efficient retrieval.

## Models and Technologies
The system utilizes the following stack for feature extraction and data management:

*   **Visual Embeddings:** Uses **DINOv2** (`facebook/dinov2-base`) to generate precise visual representations of monuments.
*   **Text Embeddings:** Uses the **BGE** model (`BAAI/bge-base-en-v1.5`) via `SentenceTransformer` to map text into semantic vector space.
*   **Vector Database:** **Qdrant** is employed to store and manage the multi-modal collection.

## Data and Infrastructure
*   **Dataset:** The system manages a collection of 4,654 multi-modal points related to Egyptian monuments.
*   **Storage:** Each point within the Qdrant collection includes image vectors, text vectors, and rich metadata (monument name, image source, and descriptive text).
*   **Environment:** The project is developed and tested within a Google Colab environment, utilizing Kaggle data integration.

## Evaluation
The system performs scientific evaluation of image-to-image retrieval tasks using **Recall@1** and **Recall@5** metrics to ensure high-performance accuracy across the monument dataset.

## Features
*   **Persona-Based Storytelling:** Includes a `get_personalized_story` function to generate content tailored to specific personas, such as "man_adult" or "girl_child".
*   **Preprocessing:** Includes custom functions for cleaning and preparing Arabic text data (e.g., handling hashtags and emojis).
