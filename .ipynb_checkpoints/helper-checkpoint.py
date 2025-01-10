# imports

import os
import re
import math
import json
from typing import List, Dict
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import chromadb
from items import Item
from testing import Tester
from agents.agent import Agent


class IncidentsAgent(Agent):

    name = "Incidents Agent"
    color = Agent.BLUE

    MODEL = "Qwen/Qwen2-7B-Instruct"
    
    def __init__(self, collection):
        """
        Set up this instance by connecting to QWEN Model, to the Chroma Datastore,
        And setting up the vector encoding model
        """
        self.log("Initializing Incidents Agent")
        self.openai = OpenAI()
        self.collection = collection
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.log("Incidents Agent is ready")

    def make_context(self, similars: List[str], potential_accident_Levels: List[int]) -> str:
        """
        Create context that can be inserted into the prompt
        :param similars: similar incidents to the one being queried for
        :param potential_accident_level: potential accident levels of similar incidents
        :return: text to insert in the prompt that provides context
        """
        message = "To provide some context, here are some other incidents that might be similar to the incident you need to answer for.\n\n"
        for similar, potential_accident_level in zip(similars, potential_accident_Levels):
            message += f"Potentially related incidents:\n{similar}\nPotential Accident Level is ${potential_accident_level:.2f}\n\n"
        return message

    def messages_for(self, description: str, similars: List[str], potential_accident_Levels: List[int]) -> List[Dict[str, str]]:
        """
        Create the message list to be included in a call to Chat Model
        With the system and user prompt
        :param description: a description of the incident
        :param similars: similar incidents to this one
        :param  potential_accident_level: potential accident levels of similar incidents
        :return: the list of messages in the format expected by Chat Model
        """
        system_message = "You estimate potential accident level of incidents. Reply only with the potential accident level, no explanation"
        user_prompt = self.make_context(similars, prices)
        user_prompt += "And now the question for you:\n\n"
        user_prompt += "What is the Potential accident level for this incident?\n\n" + description
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "Potential accident level is $"}
        ]

    def find_similars(self, description: str):
        """
        Return a list of incidents similar to the given one by looking in the Chroma datastore
        """
        self.log("Incidents Agent is performing a RAG search of the Chroma datastore to find 5 similar incidents")
        vector = self.model.encode([description])
        results = self.collection.query(query_embeddings=vector.astype(float).tolist(), n_results=5)
        documents = results['documents'][0][:]
        prices = [m['Potential_Accident_Level'] for m in results['metadatas'][0][:]]
        self.log("Frontier Agent has found similar incidents")
        return documents, prices

    def get_potential_accident_level(self, s) -> float:
        """
        A utility that plucks a integer number out of a string
        """
        s = s.replace('$','').replace(',','')
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return int(match.group()) if match else 0.0

    def potential_accident_level(self, description: str) -> float:
        """
        Make a call to Chat Model to estimate the price of the described product,
        by looking up 5 similar incidents and including them in the prompt to give context
        :param description: a description of the product
        :return: an estimate of the potential accident level
        """
        documents, prices = self.find_similars(description)
        self.log("Frontier Agent is about to call Chat Model with context including 5 similar incidents")
        response = self.openai.chat.completions.create(
            model=self.MODEL, 
            messages=self.messages_for(description, documents, prices),
            seed=42,
            max_tokens=5
        )
        reply = response.choices[0].message.content
        result = self.get_price(reply)
        self.log(f"Frontier Agent completed - predicting ${result:.2f}")
        return result
        