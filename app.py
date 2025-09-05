# -*- coding: utf-8 -*-
"""
Fénix Automotriz — Agente de Negocio (RAG) en Streamlit
Refactor RAG (estilo Gemini): chunking narrativo por fila + vectorización robusta + recuperación y prompt claros.

- La hoja MODELO_BOT se carga en un DataFrame.
- Cada FILA se convierte en un único FRAGMENTO DE TEXTO narrativo (multi-línea) con todos los campos (empezando por la OT).
- Se vectorizan esos fragmentos (OpenAI Embeddings) y se almacenan (ChromaDB si está disponible; si no, fallback local con .npz).
- La consulta del usuario se embebe y se recuperan los K fragmentos más relevantes.
- El prompt final incluye sección “Contexto proporcionado” con los fragmentos recuperados (no inventar; si falta info, decirlo).
"""

import os
import re
import json
import hashlib
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from unidecode import unidecode
from openai import OpenAI

# Chroma opcional
CHROMA_AVAILABLE = True
try:
    import chromadb
except Exception:
    CHROMA_AVAILABLE = False

st.set_page_config(page_title="Fénix | Agente de Negocio (RAG)", page_icon="🔥", layout="wide")

# ----------------- Constantes del proyecto -----------------
SHEET_ID = "1SaXuzhY_sJ9Tk9MOLDLAI4OVdsNbCP-X4L8cP15yTqo"
WORKSHEET = "MODELO_BOT"

# Orden recomendado de campos para armar el fragmento (los que existan en la hoja):
CANONICAL_FIELDS = [
    "OT","PATENTE","MARCA","MODELO","ESTADO SERVICIO","ESTADO PRESUPUESTO",
    "FECHA INGRESO PLANTA","FECHA SALIDA PLANTA","PROCESO","PIEZAS DESABOLLADAS","PIEZAS PINTADAS",
    "ASIGNACIÓN DESARME","ASIGNACIÓN DESABOLLADURA","ASIGNACIÓN PINTURA","FECHA INSPECCIÓN",
    "TIPO CLIENTE","NOMBRE CLIENTE","SINIESTRO","TIPO VEHÍCULO","FECHA RECEPCION","FECHA ENTREGA",
    "MONTO PRINCIPAL NETO","IVA PRINCIPAL [F]","MONTO PRINCIPAL BRUTO [F]","NUMERO DE FACTURA",
    "FECHA DE FACTURACION","FECHA DE PAGO FACTURA","FACTURADO","NUMERO DE DIAS EN PLANTA","DIAS EN DOMINIO",
    "CANTIDAD DE VEHICULO","DIAS DE PAGO DE FACTURA",
]

# Aliases para normalizar encabezados
COL_ALIASES = {
    "dias en planta": "NUMERO DE DIAS EN PLANTA",
    "días en planta": "NUMERO DE DIAS EN PLANTA",
    "dias de pago": "DIAS DE PAGO DE FACTURA",
    "fecha facturacion": "FECHA DE FACTURACION",
    "fecha facturación": "FECHA DE FACTURACION",
    "monto bruto": "MONTO PRINCIPAL BRUTO [F]",
    "monto neto": "MONTO PRINCIPAL NETO",
    "iva": "IVA PRINCIPAL [F]",
    "fecha recepcion": "FECHA RECEPCION",
    "fecha recepción": "FECHA RECEPCION",
    "fecha ingreso": "FECHA INGRESO PLANTA",
    "fecha salida": "FECHA SALIDA PLANTA",
    "entrega": "FECHA ENTREGA",
    "numero factura": "NUMERO DE FACTURA",
    "nro factura": "NUMERO DE FACTURA",
    "n° factura": "NUMERO DE FACTURA",
}

DATE_FIELDS = [
    "FECHA INGRESO PLANTA","FECHA SALIDA PLANTA","FECHA INSPECCIÓN",
    "FECHA RECEPCION","FECHA ENTREGA","FECHA DE FACTURACION","FECHA DE PAGO FACTURA"
]

NUM_FIELDS = [
    "MONTO PRINCIPAL NETO","IVA PRINCIPAL [F]","MONTO PRINCIPAL BRUTO [F]",
    "NUMERO DE DIAS EN PLANTA","DIAS EN DOMINIO","CANTIDAD DE VEHICULO","DIAS DE PAGO DE FACTURA",
]

BUSINESS_CONTEXT = """
Compañía: Fénix Automotriz (Chile). Empresa familiar dedicada a la reparación de vehículos, fundada en 2017.
Temas estratégicos: Experiencia excepcional, Excelencia operacional, Transformación tecnológica, Innovación y Expansión nacional.
Misión: “Entregar un servicio de reparación transparente, de calidad y puntual para transformar la insatisfacción de nuestros clientes en una agradable experiencia”.
Visión (2026): “Ofrecer el servicio de reparación automotriz preferido de nuestros clientes, colaboradores y proveedores dentro del territorio nacional”.
Etapas productivas: Presupuesto → Recepción del vehículo → Desarme → Desabolladura → Preparación → Pintura → Pulido → Lavado y entrega.
Cargos: Gerencia General, Planificación y Control, Administración y Finanzas, Gestión Comercial y Ventas, Marketing,
Líder Unidad de Negocio, Jefe de Taller, Desarmador, Desabollador, Pintor, Preparador, etc.
""".strip()

# ----------------- Credenciales Google -----------------
def _safe_secret_keys():
    try:
        return list(st.secrets._secrets.keys())
    except Exception:
        try:
            return list(st.secrets.keys())
        except Exception:
            return []

def _load_service_account_from_secrets() -> Dict[str, Any]:
    # TOML dict directo
    for k in ["gcp_service_account", "service_account", "google_service_account"]:
        try:
            obj = st.secrets[k]
            if isinstance(obj, dict) and "client_email" in obj and "private_key" in obj:
                return dict(obj)
        except Exception:
            pass
    # String JSON en secrets
    for k in ["GOOGLE_CREDENTIALS", "GOOGLE_CREDENTIALS_JSON", "SERVICE_ACCOUNT_JSON", "gcp_service_account_json"]:
        try:
            js = st.secrets.get(k, "")
            if isinstance(js, str) and js.strip().startswith("{"):
                return json.loads(js)
        except Exception:
            pass
    # Variables de entorno
    for k in ["GOOGLE_CREDENTIALS", "GOOGLE_CREDENTIALS_JSON"]:
        try:
            js = os.getenv(k, "")
            if isinstance(js, str) and js.strip().startswith("{"):
                return json.loads(js)
        except Exception:
            pass
    return {}

def get_gspread_client():
    info = _load_service_account_from_secrets()
    if not info:
        st.error("Faltan credenciales de Google Service Account. Claves presentes en secrets: **{}**".format(", ".join(_safe_secret_keys())))
        st.stop()
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)

# ----------------- Utilidades DataFrame -----------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", unidecode(str(s)).strip().lower())

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        key = _norm(c)
        # match exacto (sin acentos) con la lista canónica
        for can in CANONICAL_FIELDS:
            if _norm(can) == key:
                mapping[c] = can
        # alias
        if key in COL_ALIASES:
            mapping[c] = COL_ALIASES[key]
    return df.rename(columns=mapping)

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    # Fechas
    for col in DATE_FIELDS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True, infer_datetime_format=True)
    # Números (limpia símbolos y convierte)
    for col in NUM_FIELDS:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(r"[.$ ]", "", regex=True)
                .str.replace(",", ".", regex=True)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Normalización de FACTURADO
    if "FACTURADO" in df.columns:
        df["FACTURADO"] = (
            df["FACTURADO"].astype(str).str.strip().str.upper()
            .replace({"TRUE": "SI", "FALSE": "NO", "1": "SI", "0": "NO"})
        )
    return df

def _fmt_value(val: Any) -> str:
    if pd.isna(val):
        return ""
    if isinstance(val, pd.Timestamp):
        if pd.isna(val): return ""
        return val.strftime("%Y-%m-%d")
    if isinstance(val, float) or isinstance(val, int):
        # Evita notación científica y respeta decimales
        return f"{val:.2f}".rstrip("0").rstrip(".")
    return str(val)

def row_to_narrative(row: pd.Series, present_fields: List[str]) -> str:
    """
    Convierte UNA fila en un fragmento narrativo multi-línea:
    OT: ...
    PATENTE: ...
    ...
    """
    lines = []
    for col in present_fields:
        lines.append(f"{col}: {_fmt_value(row.get(col, ''))}")
    return "\n".join(lines)

def build_fragments(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Devuelve:
      - fragments: lista de textos (uno por fila) en formato narrativo
      - ids: lista de row_id (string) alineada con fragments
    """
    cols = [c for c in CANONICAL_FIELDS if c in df.columns]
    fragments = []
    ids = []
    for idx, row in df.iterrows():
        fragments.append(row_to_narrative(row, cols))
        ids.append(str(idx))
    return fragments, ids

@st.cache_data(show_spinner=False)
def load_sheet_dataframe() -> pd.DataFrame:
    gc = get_gspread_client()
    ws = gc.open_by_key(SHEET_ID).worksheet(WORKSHEET)
    records = ws.get_all_records()
    df = pd.DataFrame(records) if records else pd.DataFrame()
    df = normalize_columns(df)
    df = coerce_types(df)
    return df

def _hash_fragments(frags: List[str]) -> str:
    h = hashlib.md5()
    for t in frags:
        h.update(t.encode("utf-8"))
    return h.hexdigest()

# ----------------- OpenAI -----------------
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
    if not api_key:
        st.error("OPENAI_API_KEY no está configurada.")
        st.stop()
    return OpenAI(api_key=api_key)

def embed_texts(client: OpenAI, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    out = []
    B = 96
    for i in range(0, len(texts), B):
        chunk = texts[i:i+B]
        resp = client.embeddings.create(model=model, input=chunk)
        out.extend([d.embedding for d in resp.data])
    return out

# ----------------- Índice vectorial -----------------
def ensure_index(fragments: List[str], ids: List[str]):
    """
    Crea/carga el índice vectorial sobre los FRAGMENTOS narrativos.
    Guarda hash en session_state para evitar recomputar si la hoja no cambió.
    """
    current_hash = _hash_fragments(fragments)
    if st.session_state.get("index_hash") == current_hash:
        return

    if CHROMA_AVAILABLE:
        client = chromadb.PersistentClient(path="./.chroma")
        coll_name = f"fenix_modelo_bot_{current_hash[:10]}"
        try:
            collection = client.get_collection(coll_name)
        except Exception:
            collection = client.get_or_create_collection(coll_name)
            openai_client = get_openai_client()
            with st.spinner("Construyendo índice vectorial (Chroma)…"):
                embs = embed_texts(openai_client, fragments)
                metas = [{"row_id": rid} for rid in ids]
                collection.add(ids=ids, documents=fragments, metadatas=metas, embeddings=embs)
        st.session_state.index_backend = "chroma"
        st.session_state.collection = collection
    else:
        os.makedirs("./.cache", exist_ok=True)
        cache_file = f"./.cache/simple_index_{current_hash}.npz"
        if not os.path.exists(cache_file):
            openai_client = get_openai_client()
            with st.spinner("Construyendo índice vectorial (fallback)…"):
                embs = np.array(embed_texts(openai_client, fragments), dtype=np.float32)
            np.savez(cache_file, embs=embs, ids=np.array(ids, dtype=object), docs=np.array(fragments, dtype=object))
        data = np.load(cache_file, allow_pickle=True)
        st.session_state.simple_embs = data["embs"]
        st.session_state.simple_ids = data["ids"].tolist()
        st.session_state.simple_docs = data["docs"].tolist()
        st.session_state.index_backend = "simple"

    st.session_state.index_hash = current_hash

def search_top_k(query: str, k: int = 12) -> Tuple[List[str], List[str]]:
    """
    Devuelve (docs, row_ids) de los K fragmentos más relevantes.
    """
    client = get_openai_client()
    q_emb = np.array(embed_texts(client, [query])[0], dtype=np.float32)

    if st.session_state.get("index_backend") == "chroma":
        res = st.session_state.collection.query(query_embeddings=[q_emb.tolist()], n_results=k, include=["documents","metadatas"])
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        row_ids = [m.get("row_id","") for m in metas]
        return docs, row_ids

    # Fallback simple (cosine sim sobre arrays)
    A = st.session_state.simple_embs
    qn = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    sims = (An @ qn).ravel()
    idx = np.argsort(-sims)[:k]
    docs = [st.session_state.simple_docs[i] for i in idx]
    row_ids = [st.session_state.simple_ids[i] for i in idx]
    return docs, row_ids

# ----------------- Prompts -----------------
def system_prompt() -> str:
    return f"""
Eres un CONSULTOR DE GESTIÓN y ANALISTA DE DATOS para Fénix Automotriz.
Tu única fuente de verdad son los datos provistos desde la hoja MODELO_BOT.
Contexto de negocio:
{BUSINESS_CONTEXT}

Reglas:
- Usa EXCLUSIVAMENTE la información del apartado "Contexto proporcionado".
- Si la información no está, responde literalmente: "No tengo la información necesaria en los datos".
- Puedes realizar cálculos simples (sumas, promedios, min/max) sobre los registros del contexto.
- Analiza fechas, montos y estados. Sé conciso y accionable.
- Si corresponde, sugiere próximos pasos operativos (sin inventar datos).
""".strip()

def build_user_prompt(question: str, context_docs: List[str]) -> str:
    ctx = "\n\n-----\n\n".join(context_docs) if context_docs else "(sin contexto disponible)"
    return f"""
Responde en español y no inventes datos.
Pregunta del usuario:
{question}

Contexto proporcionado (fragmentos de la hoja, uno por fila):
{ctx}
""".strip()

def llm_answer(question: str, context_docs: List[str]) -> str:
    client = get_openai_client()
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": build_user_prompt(question, context_docs)},
    ]
    with st.spinner("Generando respuesta con LLM…"):
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.2)
    return resp.choices[0].message.content

# ----------------- UI -----------------
st.title("🔥 Fénix Automotriz — Agente de Negocio (RAG)")
st.caption("Google Sheets → Fragmentos narrativos → Recuperación semántica → LLM (sin alucinaciones)")

with st.sidebar:
    st.subheader("⚙️ Parámetros")
    top_k = st.slider("Fragmentos de contexto (top_k)", 4, 25, 12, 1)
    force_reindex = st.button("🔁 Reconstruir índice")
    show_debug = st.checkbox("Mostrar contexto recuperado y coincidencias", value=True)
    st.markdown("---")
    st.subheader("📦 Estado")
    st.write("Backend vectorial: **{}**".format("Chroma" if CHROMA_AVAILABLE else "Fallback simple"))

# Historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Carga de datos y construcción del índice
df = load_sheet_dataframe()
if df.empty:
    st.warning("La hoja MODELO_BOT no tiene registros o no se pudo leer.")
    st.stop()

fragments, row_ids = build_fragments(df)
if force_reindex:
    for k in ["index_hash","collection","simple_embs","simple_ids","simple_docs"]:
        st.session_state.pop(k, None)
ensure_index(fragments, row_ids)

# Render historial previo
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Entrada del usuario
question = st.chat_input("Ej.: 'no facturados de agosto', 'OT 1234 estado', 'promedio de días en planta de Toyota'")
if question:
    docs, rid_list = search_top_k(question, k=top_k)
    answer = llm_answer(question, docs)

    # Guardar y mostrar
    st.session_state.messages += [
        {"role":"user","content":question},
        {"role":"assistant","content":answer}
    ]

    with st.chat_message("assistant"):
        if show_debug:
            with st.expander("🧩 Contexto recuperado (fragmentos)"):
                for i, d in enumerate(docs, 1):
                    st.markdown(f"**Fragmento {i}**\n\n```\n{d}\n```")
        st.markdown(answer)

        # Mostrar tabla de las filas recuperadas
        try:
            idxs = [int(x) for x in rid_list if str(x).isdigit()]
            subset = df.loc[idxs] if len(idxs) else pd.DataFrame()
        except Exception:
            subset = pd.DataFrame()
        if not subset.empty:
            st.markdown("**Coincidencias (filas de MODELO_BOT):**")
            show_cols = [c for c in CANONICAL_FIELDS if c in subset.columns]
            st.dataframe(subset[show_cols] if show_cols else subset, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Descargar coincidencias (CSV)",
                               data=subset.to_csv(index=False).encode("utf-8"),
                               file_name="coincidencias.csv",
                               mime="text/csv")
        else:
            st.info("No se pudieron mapear filas exactas para visualizar en tabla (solo contexto narrativo).")
