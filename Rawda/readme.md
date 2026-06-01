

## &#x20;Ankh RAG — Egyptian Heritage Podcast Generator



This notebook implements a Retrieval-Augmented Generation (RAG) pipeline to create personalized podcast scripts about ancient Egyptian artifacts and sites. It uses a multimodal embedding model (DINOv2) for image-based retrieval, Qdrant as a vector database, and the Groq API for large language model (LLM) generation.



Features

\- \*\*Multimodal RAG\*\*: Search for information using images or text queries.

\- \*\*Persona-Driven Generation\*\*: Customize podcast scripts based on age group, gender, language, and knowledge level.

\- \*\*Structured Podcast Scripting\*\*: Generates multi-section podcast episodes with specific word count targets and narrative styles.

\- \*\*Gradio UI\*\*: Interactive interface for easy experimentation.



How it Works

1\. \*\*Setup\*\*: Installs necessary libraries and mounts Google Drive to load pre-computed embeddings.

2\. \*\*Embeddings \& Qdrant\*\*: Loads image and text embeddings into an in-memory Qdrant vector database.

3\. \*\*DINOv2 \& Groq\*\*: Initializes the DINOv2 model for image embedding and the Groq API client for LLM calls.

4\. \*\*Persona Engine\*\*: Dynamically builds a `PersonaConfig` based on user preferences.

5\. \*\*Prompt Templates\*\*: Constructs tailored prompts for each podcast section, incorporating retrieved context and persona details.

6\. \*\*Podcast Generation\*\*: Orchestrates the retrieval and generation process, calling the Groq LLM sequentially for each section.

7\. \*\*Gradio Interface\*\*: Provides a user-friendly web interface to interact with the RAG pipeline.

