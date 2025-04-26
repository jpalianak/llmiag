# TP1 - MIA
## RAG con memoria utilizando OpenIA, Pinecone, Langchain y Streamlit

##### 1) El RAG permite generar embeddings en Pinecone a partir de un pdf. Para ellos es necesario configurar, ademas del nombre del archivo, los parametros del split dentro de la funcion main_embeddings()
##### 2) Con la funcion main_request() se ejecuta Streamlit local para realizar la consulta. Ademas de la respuesta, devuelve el contexto y el historial.


# TP2 - MIA
## Agentes utilizando OpenIA, Pinecone, Langchain, LangGraph y Streamlit
##### El agente es capaz de responder preguntas sobre 3 CV que tiene precargados en Pinecone, a nombre de Laura, Marcos y Juan. Cada uno de estos CV es manegado por un agente. Es necesario ser explicito en la pregunta con los nombres de las personas que se desea consultar. En caso que no exista ningun nombre, responde por el CV de Juan.

![Estructura del grafo](tp2_agent.jpg)