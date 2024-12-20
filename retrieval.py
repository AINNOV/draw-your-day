## This code is designed to be a module in evaluate_rag.py. ##

import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from omegaconf import OmegaConf
import os

class DiaryRetriever:
    def __init__(self, index_path = None, model_name = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        ## load faiss index and contract - 'response' pairs ##
        # if index_path:
        #     self.index = faiss.read_index(index_path)
        #     self.prompts, self.responses = self.load_prompts_and_responses(index_path)

        # else:
        #     self.index = None
        #     self.prompts = []
        #     self.responses = []

        if index_path and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self.prompts, self.responses = self.load_prompts_and_responses(index_path)
        else:
            # If index file doesn't exist, create a new one
            self.index = None
            self.prompts = []
            self.responses = []
            if index_path:
                # Generate a new index and save it
                json_data = self.load_json("../data/raw/DYD_newtrain.json")
                self.index, self.prompts = self.create_faiss_index(json_data)
                self.save_faiss_index(self.index, index_path)  # Save it to the provided path

            self.index = faiss.read_index(index_path)
            self.prompts, self.responses = self.load_prompts_and_responses(index_path)

    def load_json(self, file_path):
        with open(file_path, "r", encoding="UTF-8") as f:
            return json.load(f)

    ## create faiss index from your json only if needed ##
    def create_faiss_index(self, json_data):
        prompts = [item["prompt"] for item in json_data]
        embeddings = self.model.encode(prompts, convert_to_numpy = True)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension) # L2 distance
        index.add(embeddings)
        return index, prompts

    ## save only if needed ##
    def save_faiss_index(self, index, file_path="./DYD_faiss.bin"):
        faiss.write_index(index, file_path)

    ## load "prompt"-"response" json and save to instance variables ##
    def load_prompts_and_responses(self, index_path):
        json_file_path = "../data/raw/DYD_newtrain.json"
        json_data = self.load_json(json_file_path)
        prompts = [item["prompt"] for item in json_data]
        responses = [item["response"] for item in json_data]
        return prompts, responses

    ## retrieve top k similar docs(=contracts) and corresponding responses(=analyses) ##
    def search_similar_documents(self, query, top_k=  3):
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        results = [{"prompt": self.prompts[idx], "response": self.responses[idx], "distance": distances[0][i]} for i, idx in enumerate(indices[0])]
        return results
    
print("Retrival load done.")