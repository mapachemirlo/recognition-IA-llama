import requests
import os
from dotenv import load_dotenv  # Importa dotenv para cargar variables de entorno

# Carga las variables de entorno desde el archivo .env
load_dotenv()

def get_description_from_groq(object_name):
    """
    Envía una solicitud a la API de Groq para obtener una descripción del objeto detectado.
    """
    # URL de la API de Groq configurada para Llama
    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    # Obtén la clave de API desde las variables de entorno
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("La clave de API (GROQ_API_KEY) no está configurada en las variables de entorno.")

    # Datos de la solicitud: modelo y mensaje con el objeto detectado
    payload = {
        "model": "llama3-8b-8192",  # El modelo de Llama que estás utilizando
        "messages": [
            {
                "role": "user",
                "content": f"Describe el objeto {object_name}"
            }
        ]
    }

    # Encabezados para la solicitud, incluyendo la API Key
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        # Realiza la solicitud POST a la API
        response = requests.post(API_URL, json=payload, headers=headers)
        response.raise_for_status()  # Levanta una excepción si hay errores HTTP
        
        # Extrae la respuesta de la API
        data = response.json()

        # Verifica si la respuesta contiene una descripción
        description = data.get("choices", [{}])[0].get("message", {}).get("content", "Descripción no disponible.")
        return description

    except requests.exceptions.RequestException as e:
        print(f"Error al conectar con la API de Groq: {e}")
        return "No se pudo obtener una descripción."

# Ejemplo de uso
if __name__ == "__main__":
    object_name = "dog"  # Este es el nombre del objeto que quieres que describa
    description = get_description_from_groq(object_name)
    print(description)


# ------------------------------- V1
# import requests
# import os  # Para manejar variables de entorno

# def get_description_from_groq(object_name):
#     """
#     Envía una solicitud a la API de Groq para obtener una descripción del objeto detectado.
#     """
#     # URL de la API de Groq configurada para Llama
#     API_URL = "https://api.groq.com/openai/v1/chat/completions"

#     # Obtén la clave de API desde las variables de entorno para mayor seguridad
#     GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")  # Asegúrate de configurar tu clave de API en las variables de entorno

#     # Datos de la solicitud: modelo y mensaje con el objeto detectado
#     payload = {
#         "model": "llama3-8b-8192",  # El modelo de Llama que estás utilizando
#         "messages": [
#             {
#                 "role": "user",
#                 "content": f"Describe el objeto {object_name}"
#             }
#         ]
#     }

#     # Encabezados para la solicitud, incluyendo la API Key
#     headers = {
#         "Authorization": f"Bearer {GROQ_API_KEY}",
#         "Content-Type": "application/json"
#     }

#     try:
#         # Realiza la solicitud POST a la API
#         response = requests.post(API_URL, json=payload, headers=headers)
#         response.raise_for_status()  # Levanta una excepción si hay errores HTTP
        
#         # Extrae la respuesta de la API
#         data = response.json()

#         # Verifica si la respuesta contiene una descripción
#         description = data.get("choices", [{}])[0].get("message", {}).get("content", "Descripción no disponible.")
#         return description

#     except requests.exceptions.RequestException as e:
#         print(f"Error al conectar con la API de Groq: {e}")
#         return "No se pudo obtener una descripción."

# # Ejemplo de uso
# if __name__ == "__main__":
#     object_name = "dog"  # Este es el nombre del objeto que quieres que describa
#     description = get_description_from_groq(object_name)
#     print(description)
