import json
import spacy
import numpy as np
from typing import List
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EntityExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        self.entity_labels = []
        self.phrases = {}


    def load_patterns(self, patters_file: str = None):
        with open(patters_file) as f:
            patters_json = json.load(f)
            
        self.patterns = [
            {"label": entity_label, "pattern": [{"LOWER": {"IN": entities}}]} for entity_label, entities in patters_json.items()    
        ]
        self.entity_labels = list(patters_json.keys())

        self.ruler.add_patterns(self.patterns)

    def load_phrases(self, phrases_file: str = None):
        with open(phrases_file) as f:
            self.phrases = json.load(f)


    def extract(self, text):
        if text == "":
            return {}
            
        doc = self.nlp(text)
        
        entities = defaultdict(list)
        
        # Standard named entities
        for ent in doc.ents:
            if ent.label_ in ['LAUGUAGE']:
                entities[ent.label_].append(ent.text)
        
        # Custom entities
        for ent in doc.ents:
            if ent.label_ in self.entity_labels:
                entities[ent.label_].append(ent.text)
        
        # Noun chunks
        doc_phrases = {}
        for phrase_label, phrase_list in self.phrases.items():
            doc_phrases[phrase_label] = [chunk.text for chunk in doc.noun_chunks 
                            if any(tech in chunk.text.lower() 
                                for tech in phrase_list)]

        return dict(entities) | doc_phrases
        

class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, text: str) -> np.ndarray:
        return self.model.encode(text)
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts)
    

class ResumeScorer:
    def __init__(self, entity_extractor: EntityExtractor, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.entity_extractor = entity_extractor

    def score(self, job_text: str, resume_text: str) -> float:
        job_entities = self.entity_extractor.extract(job_text)
        resume_entities = self.entity_extractor.extract(resume_text)

        joint_entity_labels = list(set(job_entities.keys()) & set(resume_entities.keys()))

        scores = {}
        for entity_label in joint_entity_labels:
            if job_entities[entity_label] and resume_entities[entity_label]:
                job_embedds = self.embedding_model.encode_batch(job_entities[entity_label])
                resume_embedds = self.embedding_model.encode_batch(resume_entities[entity_label])
                scores[entity_label] = cosine_similarity(job_embedds, resume_embedds).mean()
            else:
                scores[entity_label] = 0.0

        return scores

    def score_batch(self, job_text: str, resume_texts: List[str]) -> np.ndarray:
        pass
        