import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate         
from dotenv import load_dotenv

load_dotenv()             #Load the environment variables

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_text_from_pdf(pdf_docs):     
    text = ""
    for pdf in pdf_docs:                                   
        pdf_reader = PdfReader(pdf)             #Read the pdf
        for page in pdf_reader.pages:           #Go thru each page
            text += page.extract_text()         #Extract text

    return text


def get_text_chunks(text):                      #Divide the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

#Converting these chunks into vectors
def get_vector_store(text_chunks):    
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')    #embeddings means converting text into vectors
    vector_store = FAISS.from_texts(text_chunks,embedding=embeddings)          #Take all these text chunks and embedd according to the embeddings initialized
    
    vector_store.save_local('faiss_index')    #Save the vector store locally; Folder will be created which contains all the vector created

    

def get_conversational_chain():
    #Template is defined to guide the ai model in generating the answer
    prompt_template = """                  
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:                              

    """ 
    #Model Intialization
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)              #temperature controls the randomness of model's responses[Lower= Focused and deterministic, Higher= More creative and variabilty] 
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])             #PromtTemplate is used to guide the model in generating the answer by managing ad storing prompts
    chain = load_qa_chain(model, prompt=prompt, chain_type="stuff")              #stuff coz we need to do internal text summarization

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')    #embeddings: Method of converting text into vectors such that various classification and similarity search can be performed on the text data

    new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)             # Load the vectors stored in faiss_index folder which contains the pdf information; Set allow_dangerous_deserialization to True to allow loading of the pickle file
    
    docs = new_db.similarity_search(user_question)       #Doing similarity search for user question to find the most relevant/similar vectors to user's question

    chain = get_conversational_chain()                   

    response = chain(
        {"input_documents": docs, "question": user_question}          #Passing the input documents and user question to the chain for processing in order to generate context-aware answer; PromptTemplate and conversational chain work together to process the input data and generating the answer
        )                         #Return only the final output text without additional metadata

    print(response)
    st.write("Reply: ", response["output_text"])          #Display the answer in web app using streamlit



def main():
    st.set_page_config(page_title="Chat with Multiple PDF", page_icon="🤖", layout="wide")    #Setting the page configuration
    st.header("Chat with Multiple PDFs using Gemini")    #Header of the web app

    user_question = st.text_input("Ask a Question from PDF files")    #User input to ask the question

    if user_question:
        user_input(user_question)    #If user has asked the question, then process the input

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files and Click on Submit & Process Button", accept_multiple_files=True)    #User can upload multiple pdf files

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_text_from_pdf(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processed Successfully")


if __name__ == "__main__":
    main()