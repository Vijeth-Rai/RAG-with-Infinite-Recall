from utils import *
import chromadb

convo = []

client = chromadb.Client()

# Main loop to interact with user and handle dynamic message history
while True:
    prompt = input("USER: \n")
    context = retrieve_embeddings(prompt)
    prompt = f'USER PROMPT: {prompt} \nCONTEXT FROM EMBEDDINGS: {context}'
    
    # Stream the response from the AI
    stream_response(prompt)

    # Add the prompt and response to the conversation dynamically
    convo.append({'role': 'user', 'content': prompt})
