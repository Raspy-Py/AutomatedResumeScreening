import os
import sys

import gradio as gr
import pandas as pd

from text_procesing import PDFReader, TextPreprocessor
from scoring import ResumeScorer, EmbeddingModel, EntityExtractor


source_path = sys.path[0]
data_dir_path = os.path.join(source_path, "../../data")

# Initialize the components
pdf_reader = PDFReader()
text_preprocessor = TextPreprocessor()
embedding_model = EmbeddingModel()
entity_extractor = EntityExtractor()
entity_extractor.load_patterns(os.path.join(data_dir_path, "patterns.json"))
entity_extractor.load_phrases(os.path.join(data_dir_path, "phrases.json"))
resume_scorer = ResumeScorer(entity_extractor, embedding_model)

def load_pdf(path):
    if isinstance(path, list):
        return [pdf_reader.read(file) for file in path]
    else:
        return pdf_reader.read(path)

# Create the Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Resume Screening Application")
    with gr.Row():
        job_text = gr.Textbox(
            label="Job Description",
            placeholder="Paste job description here...",
            lines=10
        )
        job_pdf = gr.File(
            label="Job Description PDF",
            file_types=[".pdf"]
        )
    with gr.Row():
        resume_text = gr.Textbox(
            label="Resumes (one per line)",
            placeholder="Paste multiple resumes here, separated by new lines...",
            lines=10
        )
        resume_pdfs = gr.File(
            label="Resume PDFs",
            file_types=[".pdf"],
            file_count="multiple"
        )
    submit = gr.Button("Screen Resumes")
    
    output_df = gr.Dataframe(
        headers=["DEGREE", "SKILL", "SKILL-PHRASES"],
        interactive=False
    )
    
    
    def process_inputs(job, job_pdf, resumes, resume_pdfs):
        if job_pdf:
            job += load_pdf(job_pdf)

        if resumes:
            resumes = [resumes.split("\n")]
        else:
            resumes = []
    
        if resume_pdfs:
            resumes += load_pdf(resume_pdfs)

        job = text_preprocessor.normalize(job)
        resumes = [text_preprocessor.normalize(resume) for resume in resumes]

        scores = resume_scorer.score_batch(job, resumes, "mean", 0.5)      
        scores_df = pd.DataFrame(scores) 

        return scores_df

    
    submit.click(
        process_inputs,
        inputs=[job_text, job_pdf, resume_text, resume_pdfs],
        outputs=output_df
    )

if __name__ == "__main__":
    app.launch(share=True)