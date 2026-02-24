This project implements a multimodal Retrieval-Augmented Generation (RAG) system capable of answering user queries over text, PDFs, images, audio, and video by retrieving relevant context from a vector database and generating grounded responses using a Large Language Model (LLM). The system is designed to support both offline (locally hosted) and API-based LLMs.

-> In the offline mode, the model runs locally , ensuring data privacty and zero API cost.

-> IN API-Based mode, the system leverages cloud LLMs for better reasoning adn scalability. SInce RAG decouples retrieval from generation, swtiching b/w the two requires minimal changes.