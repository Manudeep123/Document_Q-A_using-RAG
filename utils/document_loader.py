from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# document loading
def load_and_split_document(file_path):
    if file_path.endswith ('.pdf'):
        loader=PyPDFLoader(file_path)
    elif file_path.endswith('.txt'):
        loader=TextLoader(file_path)
    else:
        raise ValueError("unsupported file format")
    docs=loader.load()
    #text split
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )    
    return text_splitter.split_documents(docs)

