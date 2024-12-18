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

pdf_reader_method = "pypdf2"
scoring_method = "mean"
similarity_threshold = 0.5

def config_update(pdf_reader_dropdown, scoring_dropdown, similarity_threshold_slider):
    global pdf_reader_method, scoring_method, similarity_threshold, pdf_reader

    if pdf_reader_method != pdf_reader_dropdown:
        pdf_reader_method = pdf_reader_dropdown
        pdf_reader = PDFReader(pdf_reader_method)
    scoring_method = scoring_dropdown
    similarity_threshold = similarity_threshold_slider

def load_pdf(path):
    if isinstance(path, list):
        return [pdf_reader.read(file) for file in path]
    else:
        return pdf_reader.read(path)
    

def process_inputs(job: str, job_pdf: list, resumes: pd.DataFrame, resume_pdfs: list):
    if job_pdf:
        job += load_pdf(job_pdf)
    
    resumes_previews = []
    resumes = resumes["Resume Texts"].tolist()
    resumes = [resume for resume in resumes if resume] 
    if resumes:
        resumes_previews = [resume[:25] for resume in resumes]
    else:
        resumes = []

    if resume_pdfs:
        resumes_previews += [os.path.basename(resume_pdf)[:25] for resume_pdf in resume_pdfs]
        resumes += load_pdf(resume_pdfs)

    job = text_preprocessor.normalize(job)
    resumes = [text_preprocessor.normalize(resume) for resume in resumes]

    scores = resume_scorer.score_batch(job, resumes, scoring_method, similarity_threshold)      
    scores_df = pd.DataFrame(scores) 

    scores_df.fillna(0, inplace=True)
    scores_df["RESUME"] = resumes_previews
    scores_df = scores_df[["RESUME", "DEGREE", "SKILL", "EXPERIENCE"]]

    return scores_df

# Create the Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Automated Resume Screening")
    with gr.Tab(label="Inputs"):
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
            resume_texts = gr.Dataframe(
                headers=["Resume Texts"],
                datatype=["str"],
                row_count=5, 
                col_count=1,
                interactive=True,
                wrap=True,
            )
            resume_pdfs = gr.File(
                label="Resume PDFs",
                file_types=[".pdf"],
                file_count="multiple"
            )
    with gr.Tab(label="Configuration"):
        gr.Markdown("Which library to use for PDF recognition.<br>\
                    **pypdf2** - faster<br>\
                    **pdfplumber** - more accurate")
        pdf_reader_dropdown = gr.Dropdown(
            choices=["pypdf2", "pdfplumber"],
            value=pdf_reader_method,
            label="PDF Reader Backend"
        )

        gr.Markdown("How to calculate the similarity score between job and resume.<br>\
                    **mean** - average similarity<br>\
                    **ratio** - ratio of entities above threshold<br>\
                    **sum** - sum of similarities")
        scoring_dropdown = gr.Dropdown(
            choices=["mean", "ratio", "sum"],
            value=scoring_method,
            label="Scoring Method"
        )
            
        gr.Markdown("Threshold for similarity score.<br>\
                    Used in **ratio** similarity method")
        threshold_slider = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            step=0.05,
            value=similarity_threshold,
            label="Similarity Threshold"
        )

        pdf_reader_dropdown.change(fn=config_update, inputs=[pdf_reader_dropdown, scoring_dropdown, threshold_slider])
        scoring_dropdown.change(fn=config_update, inputs=[pdf_reader_dropdown, scoring_dropdown, threshold_slider])
        threshold_slider.change(fn=config_update, inputs=[pdf_reader_dropdown, scoring_dropdown, threshold_slider])

    submit = gr.Button("Screen Resumes")
    
    output_df = gr.Dataframe(
        headers=["RESUME", "DEGREE", "SKILL", "EXPERIENCE"],
        interactive=False
    )

    
    submit.click(
        process_inputs,
        inputs=[job_text, job_pdf, resume_texts, resume_pdfs],
        outputs=output_df
    )

if __name__ == "__main__":
    app.launch()
