import os
import io
import sys

from PyPDF2 import PdfReader
from pinecone import Pinecone
from dotenv import load_dotenv
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain_pinecone import PineconeVectorStore

from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from langchain_core.prompts import (
    ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder)

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.chains import create_retrieval_chain


def init_langchain_clients():

    load_dotenv()

    openai_api_key = os.getenv('OPENAI_API_KEY')
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')
    pinecone_index = os.getenv('PINECONE_INDEX')

    # Inicializar OpenAIEmbeddings desde langchain
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Inicializar cliente Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index)

    # Inicializar Pinecone usando langchain y pasando el embedding
    pinecone_vectorstore = PineconeVectorStore(
        index=index, embedding=embeddings)

    return pinecone_vectorstore


def download_document(file_name):
    with open(file_name, 'rb') as f:
        pdf_content = f.read()

    # Usar PdfReader para extraer el texto del PDF
    # Utilizamos BytesIO para procesar el contenido binario
    pdf_reader = PdfReader(io.BytesIO(pdf_content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  # Extraer texto de cada página

    return text


def split_text_with_langchain(text, max_length, chunk_overlap, source):
    # Usar el TextSplitter de langchain para dividir el texto en fragmentos
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length, chunk_overlap=chunk_overlap)

    # Crear metadata con el nombre del archivo para cada fragmento
    metadata = [{"filename": source} for _ in range(len(text))]

    # Dividir el texto en fragmentos y asignar metadatos a cada fragmento
    documents = text_splitter.create_documents([text], metadatas=metadata)

    indices = [f"{'CV'}_{i+1}" for i in range(len(documents))]

    return documents, indices


def main_embeddings():
    # Inicializacion de variables
    pinecone_vectorstore = init_langchain_clients()

    # Cargamos el documento
    file_name = 'Curriculum.pdf'
    text = download_document(file_name)

    # Dividimos el texto en documentos
    documents, indices = split_text_with_langchain(text, 150, 0, 'CV')

    # Elimina todos los vectores del index actual
    pinecone_vectorstore._index.delete(delete_all=True)

    # Subo lo vectores de embeddings a pinecone
    pinecone_vectorstore.add_documents(documents=documents, ids=indices)


def main_request():

    st.title("RAG con Memoria - TP1 MIA")
    st.write("Asistente especializado")

    # Inicializar vectorstore
    pinecone_vectorstore = init_langchain_clients()
    retriever = pinecone_vectorstore.as_retriever(search_kwargs={"k": 3})

    # Inicializar el modelo
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # Inicializar memoria
    memory = ConversationBufferWindowMemory(
        k=5,
        return_messages=True,
        memory_key="historial_chat"
    )

    # Prompt con historial y contexto
    system_prompt = (
        "Sos un asistente especializado en Curriculum Vitae. "
        "Respondé en forma clara y concisa usando el contexto provisto. "
        "Si no hay información suficiente, respondé: 'No existe esa información'.\n"
        "Contexto: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="historial_chat"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    # Crear la chain completa
    combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=combine_chain)

    # Manejar estado en Streamlit
    if 'memory' not in st.session_state:
        st.session_state.memory = memory

    pregunta = st.text_input("Haz una pregunta:")

    if pregunta:
        # Cargar historial desde memoria
        memory_variables = st.session_state.memory.load_memory_variables({})

        # Ejecutar consulta
        inputs = {
            "input": pregunta,
            # LangChain lo espera así
            "historial_chat": memory_variables["historial_chat"]
        }
        respuesta = rag_chain.invoke(inputs)

        # Guardar nueva entrada en memoria
        st.session_state.memory.save_context(
            {"input": pregunta}, {"output": respuesta["answer"]})

        # Mostrar respuesta
        st.text_area("Respuesta del Chatbot:",
                     value=respuesta["answer"], height=100)
        # st.text_area("Contexto de la respuesta:",
        #             value=respuesta["context"], height=150)
        # st.text_area(
        #    "Historial:", value=respuesta["historial_chat"], height=150)

        st.write("Respuesta completa del Chatbot con todas sus partes:", respuesta)


if __name__ == "__main__":
    if sys.argv[1] == "embeddings":
        main_embeddings()
    elif sys.argv[1] == "request":
        main_request()
