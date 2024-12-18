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

    def score(self, job_text: str, resume_text: str, similarity_method: str = "ratio",  thr: float = 0.5):        
        similarity_func = self._get_similarity_func(similarity_method, thr)

        job_entities = self.entity_extractor.extract(job_text)
        resume_entities = self.entity_extractor.extract(resume_text)
        joint_entity_labels = list(set(job_entities.keys()) & set(resume_entities.keys()))

        scores = {}
        for entity_label in joint_entity_labels:
            if job_entities[entity_label] and resume_entities[entity_label]:
                job_embedds = self.embedding_model.encode_batch(job_entities[entity_label])
                resume_embedds = self.embedding_model.encode_batch(resume_entities[entity_label])

                scores[entity_label] = similarity_func(job_embedds, resume_embedds)
            else:
                scores[entity_label] = 0.0

        return scores

    def score_batch(self, job_text: str, resume_texts: List[str], similarity_method: str = "ratio",  thr: float = 0.5):
        similarity_func = self._get_similarity_func(similarity_method, thr)

        job_entities = self.entity_extractor.extract(job_text)
        resumes_entities = [self.entity_extractor.extract(resume_text) for resume_text in resume_texts]

        # prepare job embeddings
        job_embedds = {}
        for entity_label, entity_list in job_entities.items():
            if entity_list:
                job_embedds[entity_label] = self.embedding_model.encode_batch(entity_list)

        # calculate scores for all resumes
        scores = []
        for resume_entities in resumes_entities:
            resume_scores = {}
            for entity_label, resume_entity_list in resume_entities.items():
                if entity_label in job_entities.keys() and resume_entity_list:
                    resume_embedds = self.embedding_model.encode_batch(resume_entity_list)
                    resume_scores[entity_label] = similarity_func(job_embedds[entity_label], resume_embedds)
                else:
                    resume_scores[entity_label] = 0.0
            scores.append(resume_scores)

        return scores




    def _get_similarity_func(self, similarity_method: str, thr: float):
        if similarity_method == "ratio":
            if thr is None:
                raise ValueError("Threshold must be provided for ratio similarity")
            return lambda x, y: self._ratio_similarity(x, y, thr)
        elif similarity_method == "mean":
            return self._mean_similarity
        elif similarity_method == "sum":
            return self._sum_similarity
        else:
            raise ValueError(f"Unknown similarity method: {similarity_method}")
        


    def _ratio_similarity(self, job_embedds: np.ndarray, resume_embedds: np.ndarray, thr: float = 0.5) -> float:
        _scores = cosine_similarity(job_embedds, resume_embedds)
        max_scores = _scores.max(axis=1)
        ratio = (max_scores > thr).sum() / job_embedds.shape[0]
        return ratio
    
    def _mean_similarity(self, job_embedds: np.ndarray, resume_embedds: np.ndarray) -> float:
        _scores = cosine_similarity(job_embedds, resume_embedds)
        max_scores = _scores.max(axis=1)
        return max_scores.mean()
    
    def _sum_similarity(self, job_embedds: np.ndarray, resume_embedds: np.ndarray) -> float:
        _scores = cosine_similarity(job_embedds, resume_embedds)
        max_scores = _scores.max(axis=1)
        return max_scores.sum()
    