from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
try:
    import ollama
    print('ollama Python library imported successfully.')
except ImportError:
    print('ollama not available in Python. Either install ollama or select transformers in RAGmaker')
import string



class RAGmaker:
    def __init__(self, **kwargs):
        self.embedding_model_name = kwargs.get('embedding_model_name','all-mpnet-base-v2')
        self.generative_library = kwargs.get('generative_library','ollama')
        self.generative_transformer_model = kwargs.get('generative_transformer_model','allenai/OLMo-2-0425-1B-Instruct')
        self.generative_ollama_model = kwargs.get('generative_ollama_model','llama3.2')

        self.text_data_list = None
        self.metadata = None
        self.embedding_model = None
        self.embeddings = None
        self.faiss_index = None
        self.query = None
        self.dists_q_matched = None
        self.idxs_q_matched = None
        self.search_result_text = None
        self.generative_model = None
        self.generative_pipeline = None
        self.generative_response = None


    def encode_text_database(self, text_data_list, metadata=None):
        self.text_data_list = text_data_list
        self.metadata = metadata
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.embeddings = self.embedding_model.encode(text_data_list, normalize_embeddings=True)

    def index_database(self):
        if (self.embeddings is None):
            print('ERROR: please encode the data first, using ecode_text_database().')
            return
        
        d_emb = len(self.embeddings[0])
        self.faiss_index = faiss.IndexFlatIP(d_emb)
        self.faiss_index.add(self.embeddings) 

    def choose_generative_model(self):
        if self.generative_library == 'transformers':
            self.generative_model = self.generative_transformer_model
            print(f"You're using {self.generative_model}")
            self.generative_pipeline = pipeline(task="text-generation", model=self.generative_model)
        elif self.generative_library == 'ollama':
            available_models = [m["model"] for m in ollama.list()["models"]]
            candidates = [self.generative_ollama_model, f'{self.generative_ollama_model}:latest']
            gen_ollama_model = self.generative_ollama_model 
            for c in candidates:
                if c in available_models:
                    gen_ollama_model = c
            if gen_ollama_model not in available_models:
                print(f"ERROR: ollama model '{self.generative_ollama_model}' is not available. Try `ollama pull {self.generative_ollama_model }`")
                print('available models:')
                print(available_models)
                return
            self.generative_model = gen_ollama_model            
        else:
            print("ERROR: Please set generative_library to either 'transformers' or 'ollama'.")

    def init_text_rag_model(self, text_data_list, metadata=None):
        self.encode_text_database(text_data_list, metadata=metadata)
        self.index_database()
        self.choose_generative_model()

    def search_database(self, query_text, k=1):
        if (self.faiss_index is None):
            print('ERROR: please index the data first, using index_database().')
            return
        
        self.query = query_text
        qemb = self.embedding_model.encode(query_text, normalize_embeddings=True)
        qemb = qemb.reshape((1, qemb.shape[0]))
        self.dists_q_matched, self.idxs_q_matched = self.faiss_index.search(qemb, k)

    def format_text_matches(self, text_character_limit=None):
        if (self.idxs_q_matched is None):
            print('ERROR: please search the data first, using search_database().')
            return
        
        self.search_result_text = ""
        for n,i in enumerate(self.idxs_q_matched[0]):
            # self.search_result_text += f'Match {n+1}:\n'
            if (self.metadata is not None):
                self.search_result_text += 'Metadata:\n'
                self.search_result_text += self.metadata[i] + '\n'
            self.search_result_text += 'Text:\n'
            result = self.text_data_list[i]
            if (text_character_limit is not None):
                result = result[:text_character_limit]
            self.search_result_text += result + '\n\n'

    def search_text_database_and_print_matches(self, query_text, k=1, text_character_limit=None):
        self.search_database(query_text, k=k)
        self.format_text_matches(text_character_limit=text_character_limit)
        print(f"Match(es) to your query \'{query_text}\':\n")
        print(self.search_result_text)

    def generate_response(self, prompt, max_new_tokens=200):
        if self.generative_model is None:
            print("ERROR: generative_model not set.  Please run choose_generative_model first.")
            return
        
        if self.generative_library == 'transformers':
            response = self.generative_pipeline(prompt, max_new_tokens=max_new_tokens)
            response = response[0]['generated_text']
        elif self.generative_library == 'ollama':
            response = ollama.generate(model=self.generative_model, prompt=prompt)
            response = response.response
        else:
            print("ERROR: Please set generative_library to either 'transformers' or 'ollama'.")
            return

        self.generative_response = response


    def query_text_rag_model(self, query_text, system_prompt, k=1, text_character_limit=None, max_new_tokens=200):
        format_keys = [tup[1] for tup in string.Formatter().parse(system_prompt) if tup[1] is not None]
        if ("user_input" not in format_keys or "search_output" not in format_keys):
            print("ERROR: please include 'user_input' and 'search_output' format keys in your system prompt.")
            return
        self.search_database(query_text, k=k)
        self.format_text_matches(text_character_limit=text_character_limit)
        generative_prompt = system_prompt.format(user_input=query_text, search_output=self.search_result_text)
        self.generate_response(generative_prompt, max_new_tokens=max_new_tokens)
        print(self.generative_response)








