import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain_community.callbacks.manager import get_openai_callback

# Load environment variables
load_dotenv()

# Check if OpenAI API key is set
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")


def process_text(text):
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings(
        api_key=api_key, model="text-embedding-ada-002")
    knowledgeBase = FAISS.from_texts(chunks, embeddings)

    return knowledgeBase


def chat_with_pdf(pdf, query):
    pdf_reader = PdfReader(pdf)
    # Text variable will store the pdf text
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Create the knowledge base object
    knowledgeBase = process_text(text)

    if query:
        docs = knowledgeBase.similarity_search(query)
        llm = OpenAI(api_key=api_key)
        chain = load_qa_chain(llm, chain_type='stuff')

        with get_openai_callback() as cost:
            response = chain.run(input_documents=docs, question=query)
            print(cost)

        return response
    return "Please ask a question."


def main():
    st.title("Chat with your PDF 💬")

    pdf = st.file_uploader('Upload your PDF Document', type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        # Text variable will store the pdf text
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Create the knowledge base object
        knowledgeBase = process_text(text)

        query = st.text_input('Ask a question to the PDF')
        cancel_button = st.button('Cancel')

        if cancel_button:
            st.stop()

        if query:
            docs = knowledgeBase.similarity_search(query)
            llm = OpenAI(api_key=api_key)
            chain = load_qa_chain(llm, chain_type='stuff')

            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                print(cost)

            st.write(response)


if __name__ == "__main__":
    main()
