from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import os
import pickle


from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)


def load_url(url):
    loader = PyPDFLoader(url)
    documents = loader.load()
    return documents

def text_splitting(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    documents = text_splitter.split_documents(document)
    return documents

def create_index_and_store_them(documents):
    db = FAISS.from_documents(documents, OpenAIEmbeddings())
    db.save_local('faiss_index')
    return db


def retrieve(vectorstore,prompt, query):
    system_prompt = prompt
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    retriever = vectorstore.as_retriever()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    results = rag_chain.invoke({"input": query})

    return results['answer']


st.title("Scheme Research Application")
st.sidebar.header("Input Options")
url_input = st.sidebar.text_area("Enter URL")
# uploaded_file = st.sidebar.file_uploader("Upload a file containing URLs")
process_button = st.sidebar.button("Process URLs")


if process_button:
    if url_input:
        document = load_url(url_input)
        text_splitted_document = text_splitting(document)
        vector_store = create_index_and_store_them(text_splitted_document)
        st.success("Data processed and indexed!")
        summary_query = 'Provide Summary'
        st.text('Summary is :')
        st.write(retrieve(vectorstore =vector_store, prompt = prompt, query=summary_query))
        

query = st.text_input("Ask a question about the schemes:")
if query:
    print(query)
    vector_store = FAISS.load_local('faiss_index', OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    st.write(retrieve(vectorstore=vector_store,prompt = prompt, query = query))