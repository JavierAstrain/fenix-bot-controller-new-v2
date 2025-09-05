# -*- coding: utf-8 -*-
"""
Fénix Automotriz — Agente de Negocio (RAG) en Streamlit
-------------------------------------------------------
- Lee la hoja "MODELO_BOT" desde Google Sheets (con credenciales de servicio en st.secrets).
- Construye fragmentos de conocimiento y crea un índice vectorial en ChromaDB.
- Usa OpenAI (embeddings + chat) para responder preguntas basadas EXCLUSIVAMENTE en lo recuperado.
- Muestra tabla de registros relevantes y visualizaciones automáticas simples.
- Mantiene historial de conversación con st.session_state.

Requisitos en Streamlit Cloud:
  - Configurar Secrets con OPENAI_API_KEY y gcp_service_account (JSON del service account).
  - Compartir la planilla con el email del servicio.
"""
import os
import re
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

import gspread
from google.oauth2.service_account import Credentials

import chromadb
from openai import OpenAI
from unidecode import unidecode

# ---------------------------
# Configuración básica
# ---------------------------
st.set_page_config(page_title="Fénix | Agente de Negocio (RAG)", page_icon="🔥", layout="wide")

# IDs y constantes
SHEET_ID = "1SaXuzhY_sJ9Tk9MOLDLAI4OVdsNbCP-X4L8cP15yTqo"
WORKSHEET = "MODELO_BOT"

# Contexto de negocio (NO consultado, solo enriquece las respuestas)
BUSINESS_CONTEXT = """
Compañía: Fénix Automotriz (Chile). Empresa familiar dedicada a la reparación de vehículos, fundada en 2017.
Temas estratégicos: Experiencia excepcional, Excelencia operacional, Transformación tecnológica, Innovación y Expansión nacional.
Misión: “Entregar un servicio de reparación transparente, de calidad y puntual para transformar la insatisfacción de nuestros clientes en una agradable experiencia”.
Visión (2026): “Ofrecer el servicio de reparación automotriz preferido de nuestros clientes, colaboradores y proveedores dentro del territorio nacional”.
Etapas productivas: Presupuesto → Recepción del vehículo → Desarme → Desabolladura → Preparación → Pintura → Pulido → Lavado y entrega.
Cargos: Gerencia General, Planificación y Control, Administración y Finanzas, Gestión Comercial y Ventas, Marketing,
Líder Unidad de Negocio, Jefe de Taller, Desarmador, Desabollador, Pintor, Preparador, etc.
"""

# Campos clave a usar en los fragmentos (tal como se solicitaron)
CANONICAL_FIELDS = [
    "OT",
    "PATENTE",
    "MARCA",
    "MODELO",
    "ESTADO SERVICIO",
    "ESTADO PRESUPUESTO",
    "FECHA INGRESO PLANTA",
    "FECHA SALIDA PLANTA",
    "PROCESO",
    "PIEZAS DESABOLLADAS",
    "PIEZAS PINTADAS",
    "ASIGNACIÓN DESARME",
    "ASIGNACIÓN DESABOLLADURA",
    "ASIGNACIÓN PINTURA",
    "FECHA INSPECCIÓN",
    "TIPO CLIENTE",
    "NOMBRE CLIENTE",
    "SINIESTRO",
    "TIPO VEHÍCULO",
    "FECHA RECEPCION",
    "FECHA ENTREGA",
    "MONTO PRINCIPAL NETO",
    "IVA PRINCIPAL [F]",
    "MONTO PRINCIPAL BRUTO [F]",
    "NUMERO DE FACTURA",
    "FECHA DE FACTURACION",
    "FECHA DE PAGO FACTURA",
    "FACTURADO",
    "NUMERO DE DIAS EN PLANTA",
    "DIAS EN DOMINIO",
    "CANTIDAD DE VEHICULO",
    "DIAS DE PAGO DE FACTURA",
]

# Mapeos de nombres normalizados a canonicals (soportar acentos, mayúsculas, sufijos [F], etc.)
# La clave es "normalizada" (lowercase, sin acentos, espacios compactados, etc.)
NORMALIZATION_MAP = {
    "ot": "OT",
    "patente": "PATENTE",
    "marca": "MARCA",
    "modelo": "MODELO",
    "estado servicio": "ESTADO SERVICIO",
    "estado del servicio": "ESTADO SERVICIO",
    "estado presupuesto": "ESTADO PRESUPUESTO",
    "fecha ingreso planta": "FECHA INGRESO PLANTA",
    "fecha de ingreso planta": "FECHA INGRESO PLANTA",
    "fecha salida planta": "FECHA SALIDA PLANTA",
    "proceso": "PROCESO",
    "piezas desabolladas": "PIEZAS DESABOLLADAS",
    "piezas pintadas": "PIEZAS PINTADAS",
    "asignacion desarme": "ASIGNACIÓN DESARME",
    "asignación desarme": "ASIGNACIÓN DESARME",
    "asignacion desabolladura": "ASIGNACIÓN DESABOLLADURA",
    "asignación desabolladura": "ASIGNACIÓN DESABOLLADURA",
    "asignacion pintura": "ASIGNACIÓN PINTURA",
    "asignación pintura": "ASIGNACIÓN PINTURA",
    "fecha inspeccion": "FECHA INSPECCIÓN",
    "fecha inspección": "FECHA INSPECCIÓN",
    "tipo cliente": "TIPO CLIENTE",
    "nombre cliente": "NOMBRE CLIENTE",
    "siniestro": "SINIESTRO",
    "tipo vehiculo": "TIPO VEHÍCULO",
    "tipo vehículo": "TIPO VEHÍCULO",
    "fecha recepcion": "FECHA RECEPCION",
    "fecha recepción": "FECHA RECEPCION",
    "fecha entrega": "FECHA ENTREGA",
    "monto principal neto": "MONTO PRINCIPAL NETO",
    "iva principal": "IVA PRINCIPAL [F]",
    "iva principal f": "IVA PRINCIPAL [F]",
    "monto principal bruto": "MONTO PRINCIPAL BRUTO [F]",
    "monto principal bruto f": "MONTO PRINCIPAL BRUTO [F]",
    "numero de factura": "NUMERO DE FACTURA",
    "n° de factura": "NUMERO DE FACTURA",
    "n de factura": "NUMERO DE FACTURA",
    "fecha de facturacion": "FECHA DE FACTURACION",
    "fecha de facturación": "FECHA DE FACTURACION",
    "fecha pago factura": "FECHA DE PAGO FACTURA",
    "fecha de pago factura": "FECHA DE PAGO FACTURA",
    "facturado": "FACTURADO",
    "numero de dias en planta": "NUMERO DE DIAS EN PLANTA",
    "numero de días en planta": "NUMERO DE DIAS EN PLANTA",
    "dias en dominio": "DIAS EN DOMINIO",
    "días en dominio": "DIAS EN DOMINIO",
    "cantidad de vehiculo": "CANTIDAD DE VEHICULO",
    "cantidad de vehículo": "CANTIDAD DE VEHICULO",
    "dias de pago de factura": "DIAS DE PAGO DE FACTURA",
    "días de pago de factura": "DIAS DE PAGO DE FACTURA",
}

DATE_FIELDS_CANDIDATES = [
    "FECHA INGRESO PLANTA",
    "FECHA SALIDA PLANTA",
    "FECHA INSPECCIÓN",
    "FECHA RECEPCION",
    "FECHA ENTREGA",
    "FECHA DE FACTURACION",
    "FECHA DE PAGO FACTURA",
]

NUMERIC_FIELDS_CANDIDATES = [
    "MONTO PRINCIPAL NETO",
    "IVA PRINCIPAL [F]",
    "MONTO PRINCIPAL BRUTO [F]",
    "NUMERO DE DIAS EN PLANTA",
    "DIAS EN DOMINIO",
    "CANTIDAD DE VEHICULO",
    "DIAS DE PAGO DE FACTURA",
]

# ---------------------------
# Utilidades
# ---------------------------
def norm_text(s: str) -> str:
    s = unidecode(str(s)).lower().strip()
    s = re.sub(r"[\[\]\(\)]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Renombra columnas según NORMALIZATION_MAP
    new_cols = {}
    for c in df.columns:
        key = norm_text(c)
        mapped = NORMALIZATION_MAP.get(key)
        if mapped:
            new_cols[c] = mapped
    df = df.rename(columns=new_cols)
    return df

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    # Fechas
    for col in DATE_FIELDS_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True, infer_datetime_format=True)
    # Numéricos (permitir separadores de miles y coma)
    for col in NUMERIC_FIELDS_CANDIDATES:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[.$ ]", "", regex=True)
                .str.replace(",", ".", regex=True)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def build_fragments(df: pd.DataFrame) -> pd.DataFrame:
    # Asegurar que solo usamos la lista de CANONICAL_FIELDS si existen
    cols = [c for c in CANONICAL_FIELDS if c in df.columns]
    def row_to_fragment(row: pd.Series) -> str:
        pairs = [f"{c}: {row.get(c, '')}" for c in cols]
        return " | ".join(pairs)

    df = df.copy()
    df["__fragment__"] = df.apply(row_to_fragment, axis=1)
    df["__row_id__"] = df.index.astype(str)
    df["__ot__"] = df[cols[0]] if cols else df.index.astype(str)
    df["__patente__"] = df["PATENTE"] if "PATENTE" in df.columns else ""
    return df

@st.cache_data(show_spinner=False)
def load_sheet_dataframe() -> pd.DataFrame:
    # Credenciales desde Secrets
    try:
        service_info = dict(st.secrets["gcp_service_account"])
    except Exception as e:
        st.error("No se encontró 'gcp_service_account' en st.secrets. Revise la configuración de Secrets.")
        raise

    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_info(service_info, scopes=scopes)
    gc = gspread.authorize(creds)
    ws = gc.open_by_key(SHEET_ID).worksheet(WORKSHEET)
    records = ws.get_all_records()
    if not records:
        return pd.DataFrame()
    raw_df = pd.DataFrame(records)

    df = normalize_columns(raw_df)
    df = coerce_types(df)
    df = build_fragments(df)
    return df

def hash_dataframe(df: pd.DataFrame) -> str:
    # Hash robusto para detectar cambios
    data_bytes = df.drop(columns=[c for c in df.columns if c.startswith("__")], errors="ignore").to_csv(index=False).encode("utf-8")
    return hashlib.md5(data_bytes).hexdigest()

# ---------------------------
# Vector Store (Chroma)
# ---------------------------
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
    if not api_key:
        st.error("OPENAI_API_KEY no está configurado en el entorno ni en st.secrets.")
        st.stop()
    return OpenAI(api_key=api_key)

def embed_texts(client: OpenAI, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    # Lotea por seguridad (Chroma recomienda embeddings como lista de floats)
    out = []
    B = 96
    for i in range(0, len(texts), B):
        chunk = texts[i:i+B]
        resp = client.embeddings.create(model=model, input=chunk)
        for d in resp.data:
            out.append(d.embedding)
    return out

def ensure_vector_index(df: pd.DataFrame):
    """
    Crea/actualiza la colección en Chroma usando un hash del contenido para evitar duplicados.
    Guarda nombre y hash en st.session_state.
    """
    if "vector_index" not in st.session_state:
        st.session_state.vector_index = None
    if "vector_index_hash" not in st.session_state:
        st.session_state.vector_index_hash = None
    if "vector_collection_name" not in st.session_state:
        st.session_state.vector_collection_name = None

    client = chromadb.PersistentClient(path="./.chroma")
    current_hash = hash_dataframe(df)
    coll_name = f"fenix_modelo_bot_{current_hash[:10]}"

    if st.session_state.vector_index_hash == current_hash and st.session_state.vector_collection_name == coll_name:
        # Ya existe para este contenido
        try:
            st.session_state.vector_index = client.get_collection(coll_name)
            return st.session_state.vector_index
        except Exception:
            pass  # Si falló, reconstruimos

    # Crear/actualizar colección
    try:
        client.delete_collection(coll_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(coll_name)
    # Agregar documentos + embeddings
    openai_client = get_openai_client()
    documents = df["__fragment__"].fillna("").astype(str).tolist()
    ids = df["__row_id__"].tolist()
    metadatas = [{"row_id": rid, "ot": str(df.loc[int(rid), "OT"]) if "OT" in df.columns else str(rid),
                  "patente": str(df.loc[int(rid), "PATENTE"]) if "PATENTE" in df.columns else ""} for rid in ids]

    with st.spinner("Generando embeddings y construyendo el índice vectorial…"):
        embs = embed_texts(openai_client, documents, model="text-embedding-3-small")
        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embs)

    st.session_state.vector_index = collection
    st.session_state.vector_index_hash = current_hash
    st.session_state.vector_collection_name = coll_name
    return collection

def retrieve_top_k(collection, query: str, k: int = 8):
    openai_client = get_openai_client()
    q_emb = embed_texts(openai_client, [query], model="text-embedding-3-small")[0]
    result = collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas", "distances"])
    return result

def build_prompt(context_chunks: List[str], question: str) -> str:
    ctx = "\n\n".join([f"- {c}" for c in context_chunks if c])
    prompt = f"""
    Eres un agente de negocio experto para Fénix Automotriz. Responde SIEMPRE en español,
    con precisión, claridad y espíritu analítico, usando EXCLUSIVAMENTE la información de los fragmentos recuperados (Contexto).
    Si la respuesta no está en los datos, indícalo explícitamente y sugiere qué información faltaría.

    Además, considera el siguiente contexto de negocio para enriquecer el tono y las recomendaciones (no inventes datos):
    {BUSINESS_CONTEXT}

    Contexto (fragmentos recuperados del Google Sheet "MODELO_BOT"):
    {ctx}

    Pregunta del usuario:
    {question}

    Instrucciones de estilo:
    - Contesta de forma directa y accionable.
    - Cuando aplique, incluye KPIs, riesgos y recomendaciones operativas.
    - Evita redundancias y no inventes información fuera del contexto.
    - Si hay ambigüedad, explica supuestos razonables.
    """
    return prompt.strip()

def llm_answer(question: str, context_chunks: List[str]) -> str:
    client = get_openai_client()
    messages = [
        {"role": "system", "content": "Eres un analista de negocio senior. Respondes en español, con precisión y foco en decisiones."},
        {"role": "user", "content": build_prompt(context_chunks, question)},
    ]
    with st.spinner("Generando respuesta con LLM…"):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
        )
    return resp.choices[0].message.content

def auto_visualize(df_rows: pd.DataFrame, question: str):
    if df_rows.empty:
        return
    q = unidecode(question).lower()

    # Heurísticas simples
    date_pref = None
    for c in ["FECHA DE FACTURACION", "FECHA RECEPCION", "FECHA INGRESO PLANTA", "FECHA ENTREGA"]:
        if c in df_rows.columns:
            date_pref = c
            break

    # 1) Si pregunta por "factura" o "facturado", agrupar por mes y sumar montos si existen.
    if ("factur" in q or "boleta" in q) and date_pref and date_pref in df_rows.columns:
        dft = df_rows.copy()
        dft = dft.dropna(subset=[date_pref])
        if dft.empty:
            return
        dft["__mes__"] = dft[date_pref].dt.to_period("M").dt.to_timestamp()
        y_col = None
        for c in ["MONTO PRINCIPAL BRUTO [F]", "MONTO PRINCIPAL NETO", "IVA PRINCIPAL [F]"]:
            if c in dft.columns:
                y_col = c
                break
        if y_col:
            agg = dft.groupby("__mes__", as_index=False)[y_col].sum()
            fig = px.bar(agg, x="__mes__", y=y_col, title="Montos facturados por mes (suma)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            agg = dft.groupby("__mes__", as_index=False).size()
            agg = agg.rename(columns={"size": "cantidad"})
            fig = px.bar(agg, x="__mes__", y="cantidad", title="Registros por mes")
            st.plotly_chart(fig, use_container_width=True)
        return

    # 2) Si menciona "dias" y existe "NUMERO DE DIAS EN PLANTA", mostrar histograma
    if "dia" in q and "NUMERO DE DIAS EN PLANTA" in df_rows.columns:
        dft = df_rows.copy()
        dft = dft.dropna(subset=["NUMERO DE DIAS EN PLANTA"])
        if not dft.empty:
            fig = px.histogram(dft, x="NUMERO DE DIAS EN PLANTA", nbins=20, title="Distribución de días en planta")
            st.plotly_chart(fig, use_container_width=True)
        return

    # 3) Si menciona "facturado", torta por estado FACTURADO
    if "facturad" in q and "FACTURADO" in df_rows.columns:
        dft = df_rows.copy()
        agg = dft["FACTURADO"].fillna("Desconocido").value_counts().reset_index()
        agg.columns = ["FACTURADO", "cantidad"]
        fig = px.pie(agg, names="FACTURADO", values="cantidad", title="Distribución FACTURADO")
        st.plotly_chart(fig, use_container_width=True)
        return

    # 4) Fallback: si hay columnas numéricas, mostrar top 1 por suma
    numeric_candidates = [c for c in df_rows.columns if c in NUMERIC_FIELDS_CANDIDATES]
    if numeric_candidates:
        y_col = numeric_candidates[0]
        dft = df_rows.copy()
        dft["__idx__"] = range(1, len(dft) + 1)
        fig = px.bar(dft, x="__idx__", y=y_col, title=f"Valores de {y_col} (registros recuperados)")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# UI
# ---------------------------
st.title("🔥 Fénix Automotriz — Agente de Negocio (RAG)")
st.caption("Basado en Google Sheets → Recuperación semántica (Chroma) → Respuesta con OpenAI")

with st.sidebar:
    st.subheader("⚙️ Parámetros")
    top_k = st.slider("Máx. fragmentos a recuperar", 3, 15, 8, 1)
    force_reindex = st.button("🔁 Reconstruir índice vectorial")

    st.markdown("---")
    st.subheader("📘 Información")
    st.write("Hoja de cálculo: **MODELO_BOT**")
    st.write("Email de servicio esperado: `controller-bot@controller-bot-20.iam.gserviceaccount.com`")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Cargar datos
df = load_sheet_dataframe()
if df.empty:
    st.warning("La hoja MODELO_BOT no tiene registros o no se pudo leer correctamente.")
    st.stop()

# Verificación de columnas disponibles
missing = [c for c in CANONICAL_FIELDS if c not in df.columns]
if missing:
    with st.expander("Columnas no encontradas en la hoja (informativo)"):
        st.write(missing)

# Indexar en Chroma
if force_reindex:
    # Invalida hash para obligar a reindexar
    st.session_state.pop("vector_index_hash", None)
    st.session_state.pop("vector_collection_name", None)

collection = ensure_vector_index(df)

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Entrada de usuario
question = st.chat_input("Haz tu pregunta (ej.: ¿Cuántos vehículos facturados hubo en agosto? ¿Tiempos en planta?)")

if question:
    # Retrieve
    res = retrieve_top_k(collection, question, k=top_k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]

    # Tabla con filas originales
    row_indices = [int(m.get("row_id", "0")) for m in metas]
    sel = df.iloc[row_indices].copy() if row_indices else pd.DataFrame()

    # LLM
    answer = llm_answer(question, docs)

    # Persistir y mostrar
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)

        # Mostrar tabla de soporte (los registros que fundamentan la respuesta)
        if not sel.empty:
            st.markdown("**Registros relevantes (desde MODELO_BOT):**")
            show_cols = [c for c in CANONICAL_FIELDS if c in sel.columns]
            if show_cols:
                st.dataframe(sel[show_cols], use_container_width=True, hide_index=True)
            else:
                st.dataframe(sel, use_container_width=True, hide_index=True)

            # Visualización automática simple
            auto_visualize(sel, question)

            # Descarga CSV
            st.download_button(
                "⬇️ Descargar registros recuperados (CSV)",
                data=sel.to_csv(index=False).encode("utf-8"),
                file_name="registros_relevantes.csv",
                mime="text/csv",
            )
        else:
            st.info("No se encontraron registros suficientemente relevantes para mostrar.")