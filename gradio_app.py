import os
import openai
import pinecone
import gradio as gr
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain


# Function to load documents from a directory
def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

# Function to split documents into chunks
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

# Function to load embeddings into Pinecone vector store
def load_embeddings(docs):
    embeddings = OpenAIEmbeddings()
    index_name = "resume"
    index = Pinecone.from_documents(docs, embeddings, index_name=PINECONE_INDEX_NAME)
    return index

# Function to get similar documents based on a query
def get_similar_docs(query, index, k=2, score=False):
    if score:
        similar_docs = index.similarity_search_with_score(query, k=k)
    else:
        similar_docs = index.similarity_search(query, k=k)
    return similar_docs

# Function to load a Large Language Model (LLM) and run the question answering chain
def get_answer(query, index, llm, chain_type='stuff', k=2):
    similar_docs = get_similar_docs(query, index, k=k)
    chain = load_qa_chain(llm, chain_type=chain_type)
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer

# Gradio app function for user interaction
def qa_gradio_app():
    # Load documents
    directory = '/home/kushal/Documents/DocumentBasedQALangChain/data'
    documents = load_docs(directory)

    # Split documents
    docs = split_docs(documents)

    # Load embeddings
    index = load_embeddings(docs)

    # Load Large Language Model
    model_name = 'text-curie-001'
    llm = OpenAI(model_name=model_name)

    # Create the Gradio Interface
    gr_interface = gr.Interface(
        fn=lambda query: get_answer(query, index, llm, k=3),
        inputs=gr.inputs.Textbox(),
        outputs=gr.outputs.Textbox(),
        title="Question Answering with LLM",
        description="Enter your question to get answers from the documents.",
        examples=[
            ["What is Machine Learning?"],
            ["What tasks had been done in Machine Learning?"]
        ]
    )

    # Launch the Gradio Interface
    gr_interface.launch()

# Launch the Gradio app
qa_gradio_app()
