
from dotenv import load_dotenv
load_dotenv() 
import os
openai_key = os.getenv("OPENAI_API_KEY")
import gradio as gr
from utils.document_loader import load_and_split_document
from utils.vector_store import create_vector_store
from utils.qa_chain import setup_qa_chain
import os

def process_document(file):
    """Process uploaded document and return QA chain."""
    docs = load_and_split_document(file.name)
    vector_store = create_vector_store(docs, storage_type="faiss")
    qa_chain = setup_qa_chain(vector_store, model_type="openai")
    return qa_chain

def ask_question(qa_chain, question):
    """Get answer from QA chain."""
    if not qa_chain:
        return "Please upload a document first!"
    result = qa_chain({"query": question})
    return result["result"]

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 📄 Document Q&A System")
    with gr.Row():
        file_input = gr.File(label="Upload PDF/TXT")
        question = gr.Textbox(label="Ask a question")
    answer = gr.Textbox(label="Answer")
    process_btn = gr.Button("Process Document")
    ask_btn = gr.Button("Ask Question")
    
    qa_chain = gr.State()
    
    process_btn.click(process_document, inputs=file_input, outputs=qa_chain)
    ask_btn.click(ask_question, inputs=[qa_chain, question], outputs=answer)

demo.launch(server_name="0.0.0.0", server_port=7860) 