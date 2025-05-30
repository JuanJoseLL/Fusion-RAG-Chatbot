import streamlit as st
import requests
import json

# --- Configuración de la Página ---
st.set_page_config(page_title="Coomeva Seguros - Asistente Virtual Interno", layout="wide")

# --- Título y Descripción ---
st.title("🤖 Asistente Virtual Interno de Coomeva Seguros")
st.caption("Este asistente te ayudará a responder preguntas sobre productos, servicios y procedimientos internos de Coomeva Seguros.")

# --- URL del Backend FastAPI ---
# Asegúrate de que tu backend FastAPI (main.py) esté ejecutándose.
# Si ejecutas Streamlit y FastAPI en la misma máquina, esta URL debería funcionar.
# Si están en diferentes máquinas o contenedores, ajusta la URL.
FASTAPI_URL = "http://localhost:8000/v1/chat/completions"

# --- Inicializar el estado de la sesión para el historial del chat ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hola, soy tu asistente virtual de Coomeva Seguros. ¿En qué puedo ayudarte hoy?"}
    ]

# --- Mostrar mensajes del chat ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Entrada del Usuario ---
if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    # Añadir mensaje del usuario al historial y mostrarlo
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Preparar el payload para el backend
    # El backend espera una lista de mensajes, siendo el último el del usuario.
    # Para simplificar, aquí solo enviamos el mensaje actual del usuario
    # en el formato que espera el backend.
    # Si tuvieras un historial más complejo o un formato de payload diferente, ajústalo.
    payload = {
        "messages": [{"role": "user", "content": prompt}]
        # Puedes añadir otros parámetros que tu API espere, como 'model', 'stream', etc.
        # "model": "qwen" # Ejemplo si necesitaras especificar el modelo
    }

    # Llamar al backend FastAPI
    try:
        with st.spinner("Pensando..."):
            response = requests.post(FASTAPI_URL, json=payload, timeout=180) # Timeout de 3 minutos
            response.raise_for_status()  # Lanza una excepción para errores HTTP (4xx o 5xx)
            
            response_data = response.json()
            
            # Extraer la respuesta del asistente del JSON
            # Esto asume que tu backend devuelve un JSON con la estructura OpenAI
            assistant_reply = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if not assistant_reply:
                assistant_reply = "No he podido obtener una respuesta clara del backend."

    except requests.exceptions.RequestException as e:
        assistant_reply = f"Error al conectar con el backend: {e}"
        st.error(assistant_reply)
    except json.JSONDecodeError:
        assistant_reply = "Error al decodificar la respuesta del backend. ¿Está el backend devolviendo un JSON válido?"
        st.error(assistant_reply)
    except Exception as e:
        assistant_reply = f"Ha ocurrido un error inesperado: {e}"
        st.error(assistant_reply)

    # Añadir respuesta del asistente al historial y mostrarla
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)

# --- Barra lateral (Opcional) ---
st.sidebar.header("Acerca de")
st.sidebar.info(
    "Este es un asistente virtual para funcionarios de Coomeva Seguros. "
    "Utiliza un modelo de lenguaje avanzado y recuperación de información para responder tus preguntas."
)
st.sidebar.markdown("---")
st.sidebar.markdown("Desarrollado con Streamlit y FastAPI.") 