import re
import json
import spacy
import unicodedata
import pandas as pd
from collections import defaultdict

import PyPDF2
import pdfplumber


class PDFReader:
    def __init__(self, backend: str = 'pypdf2'):
        if backend == 'pypdf2':
            self.backend = self._read_pypdf2
        elif backend == 'pdfplumber':
            self.backend = self._read_pdfplumber
        else:
            raise ValueError("Backend not supported")
        
    def read(self, file_path: str) -> str:
        return self.backend(file_path)
    
    def _read_pypdf2(self, file_name):
        text = ""
        with open(file_name, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        
        return text

    def _read_pdfplumber(self, file_name):
        text = ""
        with pdfplumber.open(file_name) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
                
        return text  

class TextPreprocessor:
    def __init__(self):
       pass


    def normalize(self, text: str) -> str:
        """
        Normalize text by performing the following operations:
        - Convert to lowercase
        - Remove extra whitespace
        - Handle unicode characters
        - Remove multiple spaces
        - Remove email addresses (optional)
        - Remove URLs (optional)
        - Remove phone numbers (optional)
        
        Args:
            text (str): Input text to normalize
            
        Returns:
            str: Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Handle unicode characters
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text


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
        if pd.isna(text):
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
        