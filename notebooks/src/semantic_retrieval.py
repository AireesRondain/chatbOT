import os

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


class QandAPrompter:
    def __init__(
        self,
        project_root=r"D:\Projects\llm\llm_quantization",
        sentence_transformer_model_name="sentence-transformers/all-MiniLM-L6-v2",
        sentence_transformer_model_kwargs={"device": "cpu"},
        llm_model_name="llama-2-7b-chat.ggmlv3.q8_0.bin",
        llm_model_type="llama",
        llm_config={"max_new_tokens": 256, "temperature": 0.01},
    ):
        self.data_path = os.path.join(project_root, "data")
        self.models_path = os.path.join(project_root, "models")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=sentence_transformer_model_name, model_kwargs=sentence_transformer_model_kwargs
        )
        self.qa_template = """To answer the user's question, use the following information.
                            If you don't know the answer, simply state that you don't know.
                            Do not make up an answer. Magagalit si Lord.
                            Context: {context}
                            Question: {question}
                            Provide only the helpful answer below and nothing else.
                            Helpful answer:
                            """
        llm_model_name = os.path.join(self.models_path, llm_model_name)
        self.llm = CTransformers(model=llm_model_name, model_type=llm_model_type, config=llm_config)
        self.vector_storage_path = os.path.join(self.data_path, "vectorstore", "db_faiss")

    def build_vector_store(self, chunk_size=500, chunk_overlap=50):
        documents = DirectoryLoader(os.path.join(self.data_path, "input"), glob="*.pdf", loader_cls=PyPDFLoader).load()
        texts = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_documents(
            documents
        )
        FAISS.from_documents(texts, self.embeddings).save_local(self.vector_storage_path)
        print("Successfully built the vector store")

    def set_qa_prompt(self):
        prompt = PromptTemplate(template=self.qa_template, input_variables=["context", "question"])
        return prompt

    def build_retrieval_qa(self, prompt, vectordb):
        dbqa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
        return dbqa

    def setup_dbqa(self):
        vectordb = FAISS.load_local(self.vector_storage_path, self.embeddings)
        qa_prompt = self.set_qa_prompt()
        dbqa = self.build_retrieval_qa(qa_prompt, vectordb)

        return dbqa
