import ollama
import chromadb
import psycopg
import ast
from colorama import Fore
from tqdm import tqdm
from psycopg.rows import dict_row

class MemoryAI:
    def __init__(self, db_config):
        self.db_config = db_config
        self.client = chromadb.Client()
        self.system_prompt = (
            'You are an AI assistant with the capability to remember every conversation you have ever had with this user. '
            'On every new prompt, you check for any related messages from past interactions. '
            'If relevant memories are found, use them to inform your response. '
            'If they are not relevant, ignore them and respond normally. '
            'Do not mention the use of past conversations; just seamlessly integrate any useful information.'
        )
        self.conversation_history = [{'role': 'system', 'content': self.system_prompt}]

    def connect_to_db(self):
        return psycopg.connect(**self.db_config)

    def retrieve_past_interactions(self):
        conn = self.connect_to_db()
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute('SELECT * FROM interactions')
            interactions = cursor.fetchall()
        conn.close()
        return interactions

    def save_interaction(self, user_input, ai_response):
        conn = self.connect_to_db()
        with conn.cursor() as cursor:
            cursor.execute(
                'INSERT INTO interactions (timestamp, user_input, ai_response) VALUES (CURRENT_TIMESTAMP, %s, %s)',
                (user_input, ai_response)
            )
            conn.commit()
        conn.close()

    def delete_last_interaction(self):
        conn = self.connect_to_db()
        with conn.cursor() as cursor:
            cursor.execute('DELETE FROM interactions WHERE id = (SELECT MAX(id) FROM interactions)')
            conn.commit()
        conn.close()

    def generate_response(self, user_input):
        response = ''
        stream = ollama.chat(model='llama3', messages=self.conversation_history, stream=True)
        print(Fore.LIGHTGREEN_EX + '\nAI RESPONSE:')

        for chunk in stream:
            content = chunk['message']['content']
            response += content
            print(content, end='', flush=True)

        print('\n')
        self.save_interaction(user_input=user_input, ai_response=response)
        self.conversation_history.append({'role': 'assistant', 'content': response})

    def build_vector_database(self, interactions):
        db_name = 'interactions_memory'

        try:
            self.client.delete_collection(name=db_name)
        except ValueError:
            pass

        memory_db = self.client.create_collection(name=db_name)

        for interaction in interactions:
            serialized_data = f"user_input: {interaction['user_input']} ai_response: {interaction['ai_response']}"
            embedding_response = ollama.embeddings(model='nomic-embed-text', prompt=serialized_data)
            embedding = embedding_response['embedding']

            memory_db.add(
                ids=[str(interaction['id'])],
                embeddings=[embedding],
                documents=[serialized_data]
            )

    def query_embeddings(self, queries, results_per_query=2):
        relevant_embeddings = set()

        for query in tqdm(queries, desc='Querying vector database'):
            embedding_response = ollama.embeddings(model='nomic-embed-text', prompt=query)
            query_embedding = embedding_response['embedding']

            memory_db = self.client.get_collection(name='interactions_memory')
            results = memory_db.query(query_embeddings=[query_embedding], n_results=results_per_query)
            best_matches = results['documents'][0]

            for match in best_matches:
                if match not in relevant_embeddings:
                    if 'yes' in self.evaluate_embedding_relevance(query=query, context=match):
                        relevant_embeddings.add(match)

        return relevant_embeddings

    def generate_search_queries(self, user_input):
        query_instructions = {
            'You are a search query generation AI agent. '
            'Your task is to create a Python list of search queries based on first principles, '
            'which will be used to search an embedding database containing all conversations with the user. '
            'The list should contain queries necessary to retrieve any relevant information '
            'needed to accurately respond to the user input. '
            'Respond with a valid Python list only, no explanations.'
        }
        query_conversation = [
            {'role': 'system', 'content': query_instructions},
            {'role': 'user', 'content': "Draft a compelling email to my car insurance provider requesting a lower monthly premium."},
            {'role': 'assistant', 'content': '["What is the user\'s name?", "What is the user\'s current car insurance company?", "What is the user\'s current monthly premium?"]'},
            {'role': 'user', 'content': "How can I convert the text-to-speech function in my Python AI assistant to use pyttsx3 instead of the current TTS API?"},
            {'role': 'assistant', 'content': '["Text-to-Speech conversion", "Python AI assistant", "pyttsx3", "current TTS API"]'},
            {'role': 'user', 'content': user_input}
        ]

        response = ollama.chat(model='llama3', messages=query_conversation)
        print(Fore.YELLOW + f'\nGenerated Search Queries: {response["message"]["content"]} \n')

        try:
            return ast.literal_eval(response['message']['content'])
        except:
            return [user_input]

    def evaluate_embedding_relevance(self, query, context):
        relevance_check_prompt = (
            'You are an AI agent specialized in classifying embeddings. '
            'You will receive a search query and an embedded text chunk. '
            'You only respond with "yes" or "no" based on whether the context is directly relevant to the query. '
            'If the context precisely matches what the query needs, respond "yes"; otherwise, respond "no".'
        )
        relevance_conversation = [
            {'role': 'system', 'content': relevance_check_prompt},
            {'role': 'user', 'content': "SEARCH QUERY: What is the user's name?\nEMBEDDED CONTEXT: You are AI Austin. How can I assist you today, Austin?"},
            {'role': 'assistant', 'content': 'yes'},
            {'role': 'user', 'content': "SEARCH QUERY: Python AI Assistant\nEMBEDDED CONTEXT: Siri is a voice assistant available on Apple iOS and macOS. It helps users complete simple tasks via voice prompts."},
            {'role': 'assistant', 'content': 'no'},
            {'role': 'user', 'content': f"SEARCH QUERY: {query}\nEMBEDDED CONTEXT: {context}"}
        ]

        response = ollama.chat(model='llama3', messages=relevance_conversation)
        return response['message']['content'].strip().lower()

    def recall_memory(self, user_input):
        search_queries = self.generate_search_queries(user_input=user_input)
        relevant_embeddings = self.query_embeddings(queries=search_queries)
        self.conversation_history.append({'role': 'user', 'content': f'MEMORIES: {relevant_embeddings} \n\n USER INPUT: {user_input}'})
        print(f'\n{len(relevant_embeddings)} relevant embeddings added as context.')

    def run(self):
        past_interactions = self.retrieve_past_interactions()
        self.build_vector_database(interactions=past_interactions)

        while True:
            user_input = input(Fore.WHITE + 'USER: \n')

            if user_input[7:].lower() == '/recall':
                user_input = user_input[8:]
                self.recall_memory(user_input=user_input)
                self.generate_response(user_input=user_input)
            elif user_input[7:].lower() == '/forget':
                self.delete_last_interaction()
                self.conversation_history = self.conversation_history[:-2]
                print('\n')
            elif user_input[9:].lower() == '/memorize':
                user_input = user_input[10:]
                self.save_interaction(user_input=user_input, ai_response='Memory saved.')
                print('\n')
            else:
                self.conversation_history.append({'role': 'user', 'content': user_input})
                self.generate_response(user_input=user_input)


if __name__ == "__main__":
    db_settings = {
        'dbname': 'memory_agent_db',
        'user': 'memory_user',
        'password': 'your_password',
        'host': 'localhost',
        'port': '5432'
    }

    memory_ai = MemoryAI(db_config=db_settings)
    memory_ai.run()
