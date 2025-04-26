from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph
from typing import TypedDict, List
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.prompts import (
    ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder)
import operator
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import re
import networkx as nx
import matplotlib.pyplot as plt
import sys
import streamlit as st


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')


def init_langchain_clients(pinecone_index):

    # Inicializar OpenAIEmbeddings desde langchain
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Inicializar cliente Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index)

    # Inicializar Pinecone usando langchain y pasando el embedding
    pinecone_vectorstore = PineconeVectorStore(
        index=index, embedding=embeddings)

    return pinecone_vectorstore


def agente_juan(state):
    # Obtenemos el último mensaje
    input_text = state["messages"][-1].content.lower()

    input_text = re.sub(r"laura", "", input_text, flags=re.IGNORECASE).strip()
    input_text = re.sub(r"marcos", "", input_text, flags=re.IGNORECASE).strip()

    # Inicializar vectorstore
    pinecone_vectorstore = init_langchain_clients('cvjuan')
    retriever = pinecone_vectorstore.as_retriever(search_kwargs={"k": 3})

    # Inicializar el modelo
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)

    # Prompt con historial y contexto
    system_prompt = (
        "Sos un asistente especializado en responder preguntas sobre Curriculum Vitae."
        "Respondé en forma clara y concisa usando el contexto provisto."
        "Si no hay información suficiente, respondé: 'No existe esa información sobre Juan'.\n"
        "Contexto: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    # Crear la chain completa
    combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=combine_chain)

    respuesta = rag_chain.invoke({"input": input_text})

    respuesta_final = respuesta["answer"] if isinstance(
        respuesta, dict) else respuesta

    print('respuesta agente juan: ', respuesta_final)

    return {"messages": [AIMessage(content=respuesta_final)], "respuestas_agentes": [respuesta_final]}


def agente_marcos(state):
    input_text = state["messages"][-1].content.lower()

    input_text = re.sub(r"laura", "", input_text, flags=re.IGNORECASE).strip()
    input_text = re.sub(r"juan", "", input_text, flags=re.IGNORECASE).strip()

    # Inicializar vectorstore
    pinecone_vectorstore = init_langchain_clients('cvmarcos')
    retriever = pinecone_vectorstore.as_retriever(search_kwargs={"k": 3})

    # Inicializar el modelo
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)

    # Prompt con historial y contexto
    system_prompt = (
        "Sos un asistente especializado en responder preguntas sobre Curriculum Vitae."
        "Respondé en forma clara y concisa usando el contexto provisto."
        "Si no hay información suficiente, respondé: 'No existe esa información sobre Marcos'.\n"
        "Contexto: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    # Crear la chain completa
    combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=combine_chain)

    respuesta = rag_chain.invoke({"input": input_text})

    respuesta_final = respuesta["answer"] if isinstance(
        respuesta, dict) else respuesta

    print('respuesta agente Marcos: ', respuesta_final)

    return {"messages": [AIMessage(content=respuesta_final)], "respuestas_agentes": [respuesta_final]}


def agente_laura(state):
    input_text = state["messages"][-1].content.lower()

    input_text = re.sub(r"marcos", "", input_text, flags=re.IGNORECASE).strip()
    input_text = re.sub(r"juan", "", input_text, flags=re.IGNORECASE).strip()

    # Inicializar vectorstore
    pinecone_vectorstore = init_langchain_clients('cvlaura')
    retriever = pinecone_vectorstore.as_retriever(search_kwargs={"k": 3})

    # Inicializar el modelo
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)

    # Prompt con historial y contexto
    system_prompt = (
        "Sos un asistente especializado en responder preguntas sobre Curriculum Vitae."
        "Respondé en forma clara y concisa usando el contexto provisto."
        "Si no hay información suficiente, respondé: 'No existe esa información sobre Laura'.\n"
        "Contexto: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    # Crear la chain completa
    combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=combine_chain)

    respuesta = rag_chain.invoke({"input": input_text})

    respuesta_final = respuesta["answer"] if isinstance(
        respuesta, dict) else respuesta

    print('respuesta agente Laura: ', respuesta_final)

    return {"messages": [AIMessage(content=respuesta_final)], "respuestas_agentes": [respuesta_final]}


def llm(state):
    input_text = state["messages"][-1].content.lower()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    system = (
        "Eres un asistente especializado en responder preguntas sobre datos de Curriculum Vitae. "
        "Vas a recibir respuestas sobre varias personas. Tu tarea es proporcionar toda la información "
        "de manera clara y concisa, mencionando todos los datos de todas las personas de forma unificada, "
        "en un solo párrafo. Si no tienes información adicional sobre un tema o una persona, omite hacer mención de la falta de datos. "
        "Simplemente responde con los datos que tengas sobre cada persona mencionada."
        "Contexto: {context}"
    )

    messages = [SystemMessage(content=system),
                HumanMessage(content=input_text)]

    message = llm.invoke(messages)

    return {'messages': [message]}


def decidir_agentes(state):
    mensaje = state["messages"][-1].content.lower()
    agentes = []

    if re.search(r'\bjuan\b', mensaje):
        agentes.append("agente_juan")
    if re.search(r'\bmarcos\b', mensaje):
        agentes.append("agente_marcos")
    if re.search(r'\blaura\b', mensaje):
        agentes.append("agente_laura")

    # Si no se encontró ningún agente explícitamente, se activa Juan por defecto
    if not agentes:
        agentes.append("agente_juan")

    return {"agentes_seleccionados": agentes}


def branching(state):
    return state["agentes_seleccionados"]


def construir_respuesta_final(state):
    # Extraer solo los AIMessage que respondieron los agentes
    mensajes_agentes = [
        m.content for m in state["messages"]
        if isinstance(m, AIMessage)
    ]

    # Concatenar los contenidos
    respuesta_concatenada = "\n".join(mensajes_agentes)

    # Retornar como un nuevo mensaje que después va a ir al LLM
    return {"messages": [AIMessage(content=respuesta_concatenada)]}


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


# Crear el grafo con edges condicionales
builder = StateGraph(AgentState)

# Definir el nodo de entrada como la decisión condicional de qué agente usar
builder.set_entry_point("decidir_agente")

# Agregar nodos para los agentes
builder.add_node("decidir_agente", decidir_agentes)
builder.add_node("agente_juan", agente_juan)
builder.add_node("agente_marcos", agente_marcos)
builder.add_node("agente_laura", agente_laura)
builder.add_node("llm", llm)

builder.add_conditional_edges(
    "decidir_agente",
    branching,
    {
        "agente_juan": "agente_juan",
        "agente_marcos": "agente_marcos",
        "agente_laura": "agente_laura",
    }
)

builder.add_edge("agente_juan", "construir_respuesta")
builder.add_edge("agente_marcos", "construir_respuesta")
builder.add_edge("agente_laura", "construir_respuesta")

builder.add_node("construir_respuesta", construir_respuesta_final)

builder.add_edge("construir_respuesta", "llm")

graph = builder.compile()


def main():
    st.title("Agentes - TP2 MIA")
    st.write("Agentes especializados en responder sobre CV")

    # Solicitar al usuario que ingrese una pregunta
    pregunta = st.text_input("Haz una pregunta:")

    # Verificar si se ha ingresado una pregunta
    if pregunta:
        # Crear el mensaje para el agente
        messages = [HumanMessage(content=pregunta)]

        # Llamar a la función graph.invoke con el mensaje
        result = graph.invoke({"messages": messages})

        # Obtener la respuesta del último AIMessage
        respuesta = result['messages'][-1].content

        # Mostrar la respuesta en un cuadro de texto
        st.text_area("Respuesta del Chatbot:", value=respuesta, height=300)
    else:
        st.write("Por favor ingresa una pregunta para obtener una respuesta.")


if __name__ == "__main__":
    # Verificar si el script se ejecuta con el argumento correcto
    if sys.argv[1] == "main":
        main()
