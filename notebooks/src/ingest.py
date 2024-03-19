from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

DATA_PATH = "/Users/airees-adi/Python_Codes/llm_quantization/data/input" # where your pdf is
DB_FAISS_PATH = "/Users/airees-adi/Python_Codes/llm_quantization/vectorstores/db_faiss" # folder to save the vector db

# CREATE VECTOR DATABASE
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader) # to load PDF files
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name= 'sentence-transformers/all-MiniLM-L6-V2', \
                                       model_kwargs = {'device':'cpu'})
    
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    
if __name__ == "__main__":
    create_vector_db()