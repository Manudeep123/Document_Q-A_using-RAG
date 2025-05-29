from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

def setup_qa_chain(vector_store, model_type="openai"):
    """Set up the question-answering chain."""
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
       
    )