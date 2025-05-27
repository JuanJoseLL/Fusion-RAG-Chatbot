# Configuración de Google Vertex AI para Embeddings

## Pasos para configurar Vertex AI:

### 1. Crear un proyecto en Google Cloud
1. Ve a [Google Cloud Console](https://console.cloud.google.com/)
2. Crea un nuevo proyecto o selecciona uno existente
3. Anota el Project ID

### 2. Habilitar las APIs necesarias
```bash
gcloud services enable aiplatform.googleapis.com
gcloud services enable vertexai.googleapis.com
```

### 3. Crear una cuenta de servicio
1. Ve a IAM & Admin > Service Accounts
2. Crea una nueva cuenta de servicio
3. Asigna los roles:
   - Vertex AI User
   - AI Platform Developer
4. Descarga la clave JSON

### 4. Configurar variables de entorno
Actualiza tu archivo `.env` con:

```bash
# Google Vertex AI Configuration
GOOGLE_PROJECT_ID=tu-project-id-real
GOOGLE_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/ruta/completa/a/tu/service-account-key.json
```

### 5. Ubicaciones disponibles para text-embedding-005
- us-central1
- us-west1
- us-east4
- europe-west1
- asia-southeast1

### 6. Alternativa: Usar autenticación por defecto
Si estás ejecutando en Google Cloud (Compute Engine, Cloud Run, etc.), puedes omitir `GOOGLE_APPLICATION_CREDENTIALS` y usar la autenticación por defecto.

### 7. Probar la configuración
```bash
cd /home/juan/uni/semestre8/ia/Fusion-RAG-Chatbot/rag-backend
python3 -c "from embedder import VertexAIEmbeddings; emb = VertexAIEmbeddings(); print('✅ Vertex AI configurado correctamente')"
```

## Dimensiones del modelo text-embedding-005
- Dimensiones: 768
- Contexto máximo: 3072 tokens
- Idiomas soportados: Multilenguaje (incluye español)
