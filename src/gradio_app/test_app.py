import gradio as gr
import pandas as pd

def process_resumes(df):
    # This is a placeholder function that you can modify based on your screening logic
    if df is None or df.empty:
        return "No resumes submitted"
    
    # Convert the dataframe to a list of resumes
    resumes = df['Resume Text'].tolist()
    return f"Received {len(resumes)} resumes for processing"

# Create the interface
with gr.Blocks() as demo:
    gr.Markdown("## Resume Screening Application")
    gr.Markdown("Enter multiple resumes below. Each row represents one resume.")
    
    # Create a dataframe input with initial empty structure
    df = gr.Dataframe(
        headers=["Resume Text"],
        datatype=["str"],
        row_count=3,  # Start with 3 rows
        col_count=1,
        interactive=True,
        wrap=True,  # Enable text wrapping for better visibility
    )
    
    # Add a button to process the resumes
    submit_btn = gr.Button("Process Resumes")
    output = gr.Textbox(label="Results")
    
    # Connect the components
    submit_btn.click(
        fn=process_resumes,
        inputs=[df],
        outputs=[output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()