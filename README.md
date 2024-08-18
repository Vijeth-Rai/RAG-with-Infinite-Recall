

# RAG with Infinite Recall

**RAG with Infinite Recall** is a cutting-edge conversational AI system that integrates Retrieval-Augmented Generation (RAG) with an extensive memory mechanism, powered by PostgreSQL and ChromaDB. This system allows the AI to recall and utilize previous interactions, PDFs, and MongoDB data to generate more accurate and context-aware responses. The AI operates locally, ensuring privacy and control over your data.

## Features

- **Infinite Memory**: The AI can recall and store an unlimited number of conversations, providing context-aware responses based on past interactions.
- **PostgreSQL Integration**: Utilizes PostgreSQL to store and retrieve past conversations efficiently.
- **ChromaDB Vector Database**: Stores conversation embeddings for quick and accurate retrieval during interactions.
- **RAG System**: Integrates external data sources like PDFs and MongoDB to enhance response generation.
- **Local Operation**: All processing is done locally, ensuring your data remains private and secure.

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Vijeth-Rai/RAG-with-Infinite-Recall.git
   cd RAG-with-Infinite-Recall
   ```

2. **Install Dependencies**:
   Make sure you have Python installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up PostgreSQL**:
   Create a PostgreSQL database and update the `DB_PARAMS` in the script with your database credentials.

4. **Run the AI**:
   Start the AI by running the main script:
   ```bash
   python main.py
   ```

## Usage

- **Recall Past Conversations**: Type `/recall` followed by your prompt to recall and use past interactions.
- **Forget Last Conversation**: Type `/forget` to remove the last conversation from memory.
- **Memorize Specific Prompts**: Type `/memorize` followed by the prompt and response to store it permanently.

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request. We welcome any contributions that can help improve the AI system.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

- Thanks to the developers of PostgreSQL, ChromaDB, and the other open-source tools that made this project possible.
- Inspired by advancements in conversational AI and the need for intelligent, context-aware assistants.

---

Feel free to customize this README further based on your specific needs and details of your project!
