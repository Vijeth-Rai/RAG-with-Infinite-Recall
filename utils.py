from groq import Groq
import ollama
import chromadb

llm_client = Groq(api_key="")
client = chromadb.Client()

def get_ai_response(messages):
    completion = llm_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    response = ""
    for chunk in completion:
        response += chunk.choices[0].delta.content or ""
    return response.strip()

def stream_response(prompt):
    convo = []
    convo.append({'role': 'user', 'content': prompt})
    response = ""
    stream = llm_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=convo,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    print("\nASSISTANT:")

    for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        response += content
        print(content, end='', flush=True)

    print('\n')
    convo.append({'role': 'assistant', 'content': response})


def create_vector_db(conversations):
    vector_db_name = 'conversations'

    try:
        client.delete_collection(name=vector_db_name)
    except ValueError:
        pass

    vector_db = client.create_collection(name=vector_db_name)

    for c in conversations:
        serialized_convo = f'prompt: {c["prompt"]} response: {c["response"]}'
        response = ollama.embeddings(model='nomic-embed-text', prompt=serialized_convo)
        embedding = response['embedding']

        vector_db.add(
            ids=[str(c['id'])],
            embeddings=[embedding],
            documents=[serialized_convo]
        )

def retrieve_embeddings(prompt):
    # Ensure the collection exists before querying it
    collection_name = 'conversations'

    # Check if the collection exists
    try:
        vector_db = client.get_collection(name=collection_name)
    except ValueError:
        print(f"Collection '{collection_name}' does not exist. Creating the collection.")
        # If the collection doesn't exist, create it and return no context
        create_vector_db([])  # Create an empty collection
        return "No previous context available."

    # Proceed with embedding retrieval
    response = ollama.embeddings(model='nomic-embed-text', prompt=prompt)
    prompt_embedding = response['embedding']

    results = vector_db.query(query_embeddings=[prompt_embedding], n_results=1)

    # Check if any documents are returned
    if results['documents'] and len(results['documents'][0]) > 0:
        best_embedding = results['documents'][0][0]
        return best_embedding
    else:
        return "No relevant context found."


