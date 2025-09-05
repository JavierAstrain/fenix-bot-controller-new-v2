# Fénix Automotriz — Agente de Negocio (RAG) con Streamlit + Google Sheets

Este proyecto implementa un **agente de negocio** basado en **RAG (Retrieval-Augmented Generation)** para Fénix Automotriz.
La app lee la hoja **MODELO_BOT** desde Google Sheets, indexa registros en **ChromaDB** y responde preguntas con **OpenAI**
usando únicamente lo recuperado de la planilla.

## Estructura
```
app.py
requirements.txt
.streamlit/
  └── secrets.toml   # (plantilla) usa Streamlit Secrets en la nube
```

## Requisitos
- Python 3.10+
- Cuenta de OpenAI con API Key
- **Service Account** de Google (compartir la planilla con el email del servicio):
  `controller-bot@controller-bot-20.iam.gserviceaccount.com`

## Instalación (local)
```bash
pip install -r requirements.txt
streamlit run app.py
```

> **Nota**: Si corres localmente, completa `.streamlit/secrets.toml` con tus credenciales reales.
> En **Streamlit Cloud**, configura los Secrets del proyecto (recomendado) y no subas llaves al repo.

### Secrets esperados
```toml
OPENAI_API_KEY = "sk-..."

[gcp_service_account]
type = "service_account"
project_id = "controller-bot-20"
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "controller-bot@controller-bot-20.iam.gserviceaccount.com"
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "..."
universe_domain = "googleapis.com"
```

## Configuración de la planilla
- ID de la planilla: `1SaXuzhY_sJ9Tk9MOLDLAI4OVdsNbCP-X4L8cP15yTqo`
- Hoja: `MODELO_BOT`
- La app usa de forma preferente los siguientes campos (si existen):  
  `OT, PATENTE, MARCA, MODELO, ESTADO SERVICIO, ESTADO PRESUPUESTO, FECHA INGRESO PLANTA, FECHA SALIDA PLANTA,
  PROCESO, PIEZAS DESABOLLADAS, PIEZAS PINTADAS, ASIGNACIÓN DESARME, ASIGNACIÓN DESABOLLADURA, ASIGNACIÓN PINTURA,
  FECHA INSPECCIÓN, TIPO CLIENTE, NOMBRE CLIENTE, SINIESTRO, TIPO VEHÍCULO, FECHA RECEPCION, FECHA ENTREGA,
  MONTO PRINCIPAL NETO, IVA PRINCIPAL [F], MONTO PRINCIPAL BRUTO [F], NUMERO DE FACTURA, FECHA DE FACTURACION,
  FECHA DE PAGO FACTURA, FACTURADO, NUMERO DE DIAS EN PLANTA, DIAS EN DOMINIO, CANTIDAD DE VEHICULO, DIAS DE PAGO DE FACTURA`.

## ¿Cómo funciona?
1. **Carga de datos**: se lee la hoja via `gspread` con credenciales de servicio (solo lectura).
2. **Limpieza + fragmentos**: se normalizan columnas y se genera un texto por fila con los campos clave.
3. **Embeddings + Chroma**: se convierten los fragmentos a vectores (OpenAI `text-embedding-3-small`) y se indexan en ChromaDB.
4. **Consulta**: la pregunta del usuario se vectoriza y se buscan los `top_k` fragmentos más similares.
5. **Respuesta**: se construye un prompt con esos fragmentos + contexto de negocio y se envía al modelo `gpt-4o-mini`.
6. **UI**: chat con historial, tabla de registros relevantes y visualización automática simple (Plotly).

## Despliegue en Streamlit Cloud
1. Crea un repositorio con estos archivos.
2. En Streamlit Cloud, crea la app apuntando a `app.py`.
3. En **Secrets**, pega tu `OPENAI_API_KEY` y el JSON del Service Account bajo la clave `gcp_service_account`.
4. Asegúrate de compartir la planilla con el email del servicio.

## Comandos útiles
- Ejecutar local: `streamlit run app.py`
- Reconstruir índice (UI): botón **🔁 Reconstruir índice vectorial** en el sidebar.
- Los embeddings se recalculan automáticamente si cambian los datos (hash del contenido).

## Soporte
Creado por: Equipo de IA — Fénix Automotriz / NEXA  
Contacto: soporte@nexa-ai.example