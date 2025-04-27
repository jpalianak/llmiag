# 📚 TP1 - Sistema RAG con Memoria

Este proyecto implementa un sistema de recuperación aumentada con generación (RAG) utilizando embeddings creados a partir de documentos PDF, permitiendo realizar consultas contextuales sobre el contenido.

---

## 🚀 ¿Cómo funciona?

Cuando el usuario carga un documento:

1. Se generan embeddings a partir del contenido del PDF, utilizando OpenAI.
2. Los embeddings se almacenan en Pinecone para consultas futuras.
3. A través de la aplicación en Streamlit, el usuario puede hacer preguntas relacionadas al documento.
4. El sistema responde basándose en el contexto recuperado del documento y muestra además el historial de conversación.

---

## 🛠️ Tecnologías utilizadas

- Python 3
- OpenAI API (text-embedding-ada-002)
- Pinecone
- Langchain
- Streamlit

---

## 📋 Funcionalidades principales

- Carga de nuevos documentos y generación de embeddings (`main_embeddings()`).
- Consulta interactiva con recuperación de contexto (`main_request()`).
- Visualización del historial de respuestas.

---

# 👥 TP2 - Agentes de Consulta sobre Currículums

Este proyecto desarrolla un sistema multiagente que responde preguntas específicas sobre tres perfiles laborales almacenados en Pinecone.

---

## 🚀 ¿Cómo funciona?

Cuando el usuario ingresa una pregunta:

1. El sistema analiza la pregunta para identificar a qué persona (Laura, Marcos o Juan) hace referencia.
2. Activa el agente correspondiente para responder en base al currículum almacenado.
3. Si no se detecta ningún nombre, el sistema responde utilizando por defecto el CV de Juan.

---

## 🧩 Agentes disponibles

- **Agente Laura**:  
  Especializado en el perfil profesional de Laura.

- **Agente Marcos**:  
  Especializado en el perfil profesional de Marcos.

- **Agente Juan**:  
  Especializado en el perfil profesional de Juan (también responde por defecto si no se especifica un nombre).

---

## 🛠️ Tecnologías utilizadas

- Python 3
- OpenAI API
- Pinecone
- Langchain
- LangGraph
- Streamlit

---

## 🖼️ Diagrama del flujo de agentes

![Estructura del grafo TP2](tp2_agent_diagrama.jpg)

---

# 🧠 TP3 - Asistente de Estrategia Empresarial

Este proyecto implementa un sistema basado en agentes capaces de colaborar para responder preguntas relacionadas con la estrategia de lanzamiento y gestión de productos en el mercado.

---

## 🚀 ¿Cómo funciona?

Cuando el usuario ingresa una pregunta, el sistema:

1. Analiza la pregunta y selecciona automáticamente qué agentes especializados deben intervenir.
2. Cada agente aporta su respuesta en base a su área de especialización.
3. Las respuestas se combinan para construir una respuesta final completa y coherente.
4. Si la pregunta no está relacionada con los temas disponibles, el sistema responde:  
   _"No estoy preparado para responder sobre ese tema."_

---

## 🧩 Agentes disponibles

- **Agente de Mercado**:  
  Especialista en análisis de tendencias de mercado, comportamiento del consumidor y evaluación de la competencia.

- **Agente de Marketing**:  
  Especialista en estrategias de promoción, segmentación de clientes y tácticas de comunicación comercial.

- **Agente de Distribución**:  
  Especialista en planificación de canales de distribución, logística comercial y expansión a nuevos mercados.

- **Agente Sin-Especialización**:  
  Responde cuando la pregunta no corresponde a ninguna de las áreas anteriores.

---

## 🛠️ Tecnologías utilizadas

- Python 3
- OpenAI API (gpt-3.5-turbo)
- Streamlit

---

## 📋 Ejemplos de preguntas

- ¿Cómo lanzar un nuevo producto al mercado?
- ¿Qué estrategia de marketing me recomiendan para un producto tecnológico?
- ¿Cómo puedo organizar la distribución de un nuevo producto en diferentes regiones?
- ¿Me podés explicar cómo resolver una ecuación matemática? _(En este caso el sistema indicará que no está preparado para responder)

---
