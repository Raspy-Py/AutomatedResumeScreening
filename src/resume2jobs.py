import sys
import pandas as pd
from text_procesing import PDFReader, TextPreprocessor
from scoring import EntityExtractor, EmbeddingModel, ResumeScorer

resumes_df_path = "../data/UpdatedResumeDataSet.csv"
jobs_df_path = "../data/data job posts.csv"

def main():
    # extract several job post from dataframe
    jobs_df = pd.read_csv(jobs_df_path)
    sample_jobs_df = jobs_df.sample(10)
    

    # Read the PDF file
    pdf_reader = PDFReader(backend='pypdf2')
    text = pdf_reader.read('../data/Resume.pdf')
    
    # Normalize the text
    text_preprocessor = TextPreprocessor()
    text = text_preprocessor.normalize(text)
    
    # Extract entities
    entity_extractor = EntityExtractor()
    entity_extractor.load_patterns('../data/patterns.json')
    entity_extractor.load_phrases('../data/phrases.json')
    entities = entity_extractor.extract(text)
    
    # Print the entities
    print(entities)

if __name__ == "__main__":
    main()
