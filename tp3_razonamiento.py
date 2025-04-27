from typing import Annotated, TypedDict
import operator
from langgraph.graph import StateGraph
from langchain_core.messages import AnyMessage
import openai
from dotenv import load_dotenv
import os
import sys
import streamlit as st

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


# Configurar el cliente de OpenAI
client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Definir la variables globales
tokens_razonamiento = 0


# Agente especialista en el Mercado
def agente_mercado(state: AgentState) -> AgentState:
    global tokens_razonamiento

    pregunta = state["messages"][-1]

    prompt = f"""
    Sos un agente especializado en estudios de mercado. Tu tarea es proporcionar un análisis profundo y detallado sobre la situación del mercado relacionada con la siguiente consulta. Analiza las tendencias, los comportamientos de los consumidores, la competencia y cualquier otro factor relevante.

    Consulta: "{pregunta}"

    Proporciona una respuesta clara, concisa y basada en datos de mercado actuales.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    respuesta = response.choices[0].message.content

    # Sumar tokens de razonamiento
    consumo = response.usage
    tokens = consumo.prompt_tokens + consumo.completion_tokens
    tokens_razonamiento += tokens

    # Imprimir la pregunta, respuesta y tokens (para usar en debugger)
    if False:
        print('### AGENTE MERCADO')
        print(f"Pregunta para Agente Mercado: {pregunta}")
        print(f"Respuesta de Agente Mercado: {respuesta}")
        print(f"Tokens consumidos Mercado: {tokens}\n")

    return {"messages": [f"Respuesta Mercado: {respuesta}"]}

# Agente especialista en Marketing


def agente_marketing(state: AgentState) -> AgentState:
    global tokens_razonamiento

    pregunta = state["messages"][-1]

    prompt = f"""
    Sos un agente especializado en marketing. Tu tarea es proporcionar recomendaciones sobre estrategias de marketing efectivas relacionadas con la consulta que se te presenta. Ten en cuenta las mejores prácticas, las tendencias de marketing actuales y los canales más efectivos.

    Consulta: "{pregunta}"

    Responde con ideas claras sobre cómo se puede aplicar una estrategia de marketing para abordar esta situación.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    respuesta = response.choices[0].message.content

    # Sumar tokens de razonamiento
    consumo = response.usage
    tokens = consumo.prompt_tokens + consumo.completion_tokens
    tokens_razonamiento += tokens

    # Imprimir la pregunta, respuesta y tokens (para usar en debugger)
    if False:
        print('### AGENTE MARKETING')
        print(f"Pregunta para Agente Marketing: {pregunta}")
        print(f"Respuesta de Agente Marketing: {respuesta}")
        print(f"Tokens consumidos Marketing: {tokens}\n")

    return {"messages": [f"Respuesta Marketing: {respuesta}"]}

# Agente especialista en Distribucion


def agente_distribucion(state: AgentState) -> AgentState:
    global tokens_razonamiento

    pregunta = state["messages"][-1]

    prompt = f"""
    Sos un agente especializado en distribución. Tu tarea es proporcionar un análisis sobre cómo distribuir un producto de manera efectiva, teniendo en cuenta aspectos como canales de distribución, logística, alianzas estratégicas y mercados relevantes.

    Consulta: "{pregunta}"

    Brinda recomendaciones sobre cómo planificar y ejecutar la distribución de un producto en el contexto de esta consulta.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    respuesta = response.choices[0].message.content

    # Sumar tokens de razonamiento
    consumo = response.usage
    tokens = consumo.prompt_tokens + consumo.completion_tokens
    tokens_razonamiento += tokens

    # Imprimir la pregunta, respuesta y tokens (para usar en debugger)
    if False:
        print('### AGENTE DISTRIBUCION')
        print(f"Pregunta para Agente Distribucion: {pregunta}")
        print(f"Respuesta de Agente Distribucion: {respuesta}")
        print(f"Tokens consumidos Distribucion: {tokens}\n")

    return {"messages": [f"Respuesta Distribución: {respuesta}"]}


# Agente en el caso que no se consulte ningun de los anteriores, cuando el usuario consulta por otro tema
def sin_agente(state: AgentState) -> AgentState:
    return {"messages": ["No estoy preparado para responder sobre ese tema."]}


# Agente decisor. Toma la pregunta, la interpreta y decide que agentes deben intervenir.
def decidir_agentes(state: AgentState) -> dict:
    global tokens_razonamiento, agentes_activados, tokens_entrada

    pregunta = state["messages"][0]

    # Prompt para que el LLM decida los agentes adecuados según la pregunta
    prompt = f"""
    La siguiente es una pregunta que un Asistente de Estrategia Empresarial debe responder. Dependiendo de la naturaleza de la pregunta, selecciona qué agentes deben involucrarse. Los agentes disponibles son:
    1. Agente de Mercado: especializado en análisis de tendencias de mercado, comportamiento del consumidor, competencia y situación actual del mercado. Busca palabras clave como 'mercado', 'tendencias', 'consumidor', 'competencia'.
    2. Agente de Marketing: especializado en estrategias de marketing, promoción de productos, segmentación de clientes y tácticas publicitarias. Busca palabras clave como 'marketing', 'estrategias', 'promoción', 'clientes'.
    3. Agente de Distribución: especializado en la distribución de productos, planificación de canales de distribución, alianzas estratégicas y expansión a nuevos mercados. Busca palabras clave como 'distribución', 'canales', 'expansión', 'mercados'.

    La pregunta es: "{pregunta}"

    Devuelve una lista solo con los nombres de los agentes (mercado, marketing o distribucion) que consideres relevantes para responder la pregunta. Si la pregunta no está relacionada con ninguno de estos temas, responde "No hay agentes relevantes para esta pregunta".
"""

    # Realizar la invocación al LLM para obtener los agentes
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    # Extraer los agentes de la respuesta del LLM
    agentes_respuesta = response.choices[0].message.content

    # Aquí buscamos las palabras clave y asignamos los agentes correspondientes
    agentes_activados = []

    if 'mercado' in agentes_respuesta.lower():
        agentes_activados.append('agente_mercado')
    if 'distribucion' in agentes_respuesta.lower() or 'distribución' in agentes_respuesta.lower():
        agentes_activados.append('agente_distribucion')
    if 'marketing' in agentes_respuesta.lower():
        agentes_activados.append('agente_marketing')
    if not agentes_activados:
        agentes_activados.append('sin_agente')

    # Sumar tokens de razonamiento
    consumo = response.usage
    tokens_entrada = consumo.prompt_tokens
    tokens_razonamiento += consumo.completion_tokens

    # Imprimir la pregunta, respuesta y tokens (para usar en debugger)
    if True:
        print('### AGENTE DECISOR')
        print(f"Pregunta para Agente decisor: {pregunta}")
        print(f"Respuesta de Agente decisor: {agentes_respuesta}")
        print(f"Agentes seleccionados: {agentes_activados}")
        print(f"Tokens consumidos de entrada: {tokens_entrada}\n")

    # Devolver los agentes a activar
    return {"agentes": agentes_activados}


# Función de branching (ramificación)
def branching(result: dict) -> str:
    # Se utiliza el nombre del agente para decidir a cuales nodos ir.
    return result["agentes"]


# Función que compila la respuesta final
def construir_respuesta_final(state: AgentState) -> AgentState:
    mensajes = state["messages"]
    respuesta_final = "\n\n".join(mensajes[1:])
    return {"messages": [respuesta_final]}


# Agente que genera la respuesta final. Recibe las respuestas de todos los agentes y genera una unica respuesta final.
def llm_final(state: AgentState) -> AgentState:
    global tokens_salida, tokens_razonamiento, agentes_activados

    pregunta = state["messages"][0]
    respuestas_agentes = state["messages"][-1]

    prompt = (
        f"Esta es la pregunta original: {pregunta}\n\n"
        f"Estas son las respuestas parciales:\n{respuestas_agentes}\n\n"
        "Generá una respuesta final y bien estructurada usando toda esta información."
        "Si en las respuesta parciales dice No estoy preparado para responder sobre ese tema, reponde:No estoy preparado para responder sobre ese tema."
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    respuesta_final = response.choices[0].message.content

    # Guardar los tokens usados en la etapa final
    consumo = response.usage
    tokens_razonamiento = consumo.prompt_tokens
    tokens_salida = consumo.completion_tokens

    # Imprimir la pregunta, respuesta y tokens
    if False:
        print('### RESPUESTA Y RESUMEN FINAL')
        print(f"Pregunta : {pregunta}\n")
        print(f"Respuesta : {respuesta_final}\n")
        print(f"Tokens consumidos de entrada: {tokens_entrada}")
        print(f"Tokens consumidos de salida: {tokens_salida}")
        print(f"Tokens consumidos de razonamiento: {tokens_razonamiento}")
        print(f"Agentes involucrados en la respuesta: {agentes_activados}")

    return {"messages": [respuesta_final]}


# Crear el grafo
builder = StateGraph(AgentState)

# Definir los nodos
builder.set_entry_point("decidir_agente")
builder.add_node("decidir_agente", decidir_agentes)
builder.add_node("agente_mercado", agente_mercado)
builder.add_node("agente_marketing", agente_marketing)
builder.add_node("agente_distribucion", agente_distribucion)
builder.add_node("sin_agente", sin_agente)
builder.add_node("construir_respuesta", construir_respuesta_final)
builder.add_node("llm_final", llm_final)

# Definir las conexiones
builder.add_conditional_edges(
    "decidir_agente",
    branching,
    {
        "agente_mercado": "agente_mercado",
        "agente_marketing": "agente_marketing",
        "agente_distribucion": "agente_distribucion",
        "sin_agente": "sin_agente"
    }
)

builder.add_edge("agente_mercado", "construir_respuesta")
builder.add_edge("agente_marketing", "construir_respuesta")
builder.add_edge("agente_distribucion", "construir_respuesta")
builder.add_edge("sin_agente", "construir_respuesta")

builder.add_edge("construir_respuesta", "llm_final")

# Compilar el grafo
graph = builder.compile()


def main():
    st.title("Agentes con razonamiento - TP3 MIA")
    st.write(
        "Asistente de Estrategia Empresarial: Mercado, Marketing y Distribución")

    # Solicitar al usuario que ingrese una pregunta. Ejemplo:¿Cómo lanzar un nuevo producto de tecnología al mercado?
    pregunta = st.text_input("Haz una pregunta:")

    # Verificar si se ha ingresado una pregunta
    if pregunta:
        # Llamar a la función graph.invoke con el mensaje
        result = graph.invoke({"messages": [pregunta]})

        # Obtener la respuesta del último AIMessage
        respuesta = result['messages'][-1]

        # Mostrar la respuesta en un cuadro de texto
        st.text_area("Respuesta del Chatbot:", value=respuesta, height=600)

        # Mostrar información adicional
        st.markdown("### Información del proceso")
        st.write(f"**Tokens de entrada:** {tokens_entrada}")
        st.write(f"**Tokens de salida:** {tokens_salida}")
        st.write(f"**Tokens de razonamiento:** {tokens_razonamiento}")
        st.write(f"**Agentes involucrados:** {', '.join(agentes_activados)}")
    else:
        st.write("Por favor ingresa una pregunta para obtener una respuesta.")


if __name__ == "__main__":
    if sys.argv[1] == "main":
        main()
