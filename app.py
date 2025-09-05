# -*- coding: utf-8 -*-
"""
Fénix Automotriz — Agente de Negocio (RAG) en Streamlit
Versión con fallback si ChromaDB no está disponible (p. ej., entornos Python 3.13).
- Intenta usar ChromaDB (Persistente). Si falla la importación, usa un índice simple en memoria
  con similitud coseno (NumPy), persistiendo a disco como .npz para acelerar siguientes cargas.
"""
import os, re, json, hashlib
from typing import List, Dict
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI
from unidecode import unidecode

# --- Intentar importar ChromaDB ---
CHROMA_AVAILABLE = True
try:
    import chromadb
except Exception as e:
    CHROMA_AVAILABLE = False

st.set_page_config(page_title="Fénix | Agente de Negocio (RAG)", page_icon="🔥", layout="wide")

SHEET_ID = "1SaXuzhY_sJ9Tk9MOLDLAI4OVdsNbCP-X4L8cP15yTqo"
WORKSHEET = "MODELO_BOT"

BUSINESS_CONTEXT = """
Compañía: Fénix Automotriz (Chile). Empresa familiar dedicada a la reparación de vehículos, fundada en 2017.
Temas estratégicos: Experiencia excepcional, Excelencia operacional, Transformación tecnológica, Innovación y Expansión nacional.
Misión: “Entregar un servicio de reparación transparente, de calidad y puntual para transformar la insatisfacción de nuestros clientes en una agradable experiencia”.
Visión (2026): “Ofrecer el servicio de reparación automotriz preferido de nuestros clientes, colaboradores y proveedores dentro del territorio nacional”.
Etapas productivas: Presupuesto → Recepción del vehículo → Desarme → Desabolladura → Preparación → Pintura → Pulido → Lavado y entrega.
Cargos: Gerencia General, Planificación y Control, Administración y Finanzas, Gestión Comercial y Ventas, Marketing,
Líder Unidad de Negocio, Jefe de Taller, Desarmador, Desabollador, Pintor, Preparador, etc.
"""

CANONICAL_FIELDS = [
    "OT","PATENTE","MARCA","MODELO","ESTADO SERVICIO","ESTADO PRESUPUESTO",
    "FECHA INGRESO PLANTA","FECHA SALIDA PLANTA","PROCESO","PIEZAS DESABOLLADAS","PIEZAS PINTADAS",
    "ASIGNACIÓN DESARME","ASIGNACIÓN DESABOLLADURA","ASIGNACIÓN PINTURA","FECHA INSPECCIÓN",
    "TIPO CLIENTE","NOMBRE CLIENTE","SINIESTRO","TIPO VEHÍCULO","FECHA RECEPCION","FECHA ENTREGA",
    "MONTO PRINCIPAL NETO","IVA PRINCIPAL [F]","MONTO PRINCIPAL BRUTO [F]","NUMERO DE FACTURA",
    "FECHA DE FACTURACION","FECHA DE PAGO FACTURA","FACTURADO","NUMERO DE DIAS EN PLANTA","DIAS EN DOMINIO",
    "CANTIDAD DE VEHICULO","DIAS DE PAGO DE FACTURA",
]

NORMALIZATION_MAP = {
    "ot":"OT","patente":"PATENTE","marca":"MARCA","modelo":"MODELO",
    "estado servicio":"ESTADO SERVICIO","estado del servicio":"ESTADO SERVICIO",
    "estado presupuesto":"ESTADO PRESUPUESTO",
    "fecha ingreso planta":"FECHA INGRESO PLANTA","fecha de ingreso planta":"FECHA INGRESO PLANTA",
    "fecha salida planta":"FECHA SALIDA PLANTA","proceso":"PROCESO",
    "piezas desabolladas":"PIEZAS DESABOLLADAS","piezas pintadas":"PIEZAS PINTADAS",
    "asignacion desarme":"ASIGNACIÓN DESARME","asignación desarme":"ASIGNACIÓN DESARME",
    "asignacion desabolladura":"ASIGNACIÓN DESABOLLADURA","asignación desabolladura":"ASIGNACIÓN DESABOLLADURA",
    "asignacion pintura":"ASIGNACIÓN PINTURA","asignación pintura":"ASIGNACIÓN PINTURA",
    "fecha inspeccion":"FECHA INSPECCIÓN","fecha inspección":"FECHA INSPECCIÓN",
    "tipo cliente":"TIPO CLIENTE","nombre cliente":"NOMBRE CLIENTE","siniestro":"SINIESTRO",
    "tipo vehiculo":"TIPO VEHÍCULO","tipo vehículo":"TIPO VEHÍCULO",
    "fecha recepcion":"FECHA RECEPCION","fecha recepción":"FECHA RECEPCION",
    "fecha entrega":"FECHA ENTREGA","monto principal neto":"MONTO PRINCIPAL NETO",
    "iva principal":"IVA PRINCIPAL [F]","iva principal f":"IVA PRINCIPAL [F]",
    "monto principal bruto":"MONTO PRINCIPAL BRUTO [F]","monto principal bruto f":"MONTO PRINCIPAL BRUTO [F]",
    "numero de factura":"NUMERO DE FACTURA","n° de factura":"NUMERO DE FACTURA","n de factura":"NUMERO DE FACTURA",
    "fecha de facturacion":"FECHA DE FACTURACION","fecha de facturación":"FECHA DE FACTURACION",
    "fecha pago factura":"FECHA DE PAGO FACTURA","fecha de pago factura":"FECHA DE PAGO FACTURA",
    "facturado":"FACTURADO","numero de dias en planta":"NUMERO DE DIAS EN PLANTA",
    "numero de días en planta":"NUMERO DE DIAS EN PLANTA","dias en dominio":"DIAS EN DOMINIO",
    "días en dominio":"DIAS EN DOMINIO","cantidad de vehiculo":"CANTIDAD DE VEHICULO",
    "cantidad de vehículo":"CANTIDAD DE VEHICULO","dias de pago de factura":"DIAS DE PAGO DE FACTURA",
    "días de pago de factura":"DIAS DE PAGO DE FACTURA",
}

DATE_FIELDS_CANDIDATES = [
    "FECHA INGRESO PLANTA","FECHA SALIDA PLANTA","FECHA INSPECCIÓN","FECHA RECEPCION",
    "FECHA ENTREGA","FECHA DE FACTURACION","FECHA DE PAGO FACTURA",
]

NUMERIC_FIELDS_CANDIDATES = [
    "MONTO PRINCIPAL NETO","IVA PRINCIPAL [F]","MONTO PRINCIPAL BRUTO [F]",
    "NUMERO DE DIAS EN PLANTA","DIAS EN DOMINIO","CANTIDAD DE VEHICULO","DIAS DE PAGO DE FACTURA",
]

def norm_text(s: str) -> str:
    s = unidecode(str(s)).lower().strip()
    s = re.sub(r"[\[\]\(\)]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    for c in df.columns:
        key = norm_text(c)
        mapped = NORMALIZATION_MAP.get(key)
        if mapped:
            new_cols[c] = mapped
    return df.rename(columns=new_cols)

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in DATE_FIELDS_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True, infer_datetime_format=True)
    for col in NUMERIC_FIELDS_CANDIDATES:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(r"[.$ ]", "", regex=True)
                .str.replace(",", ".", regex=True)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def build_fragments(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in CANONICAL_FIELDS if c in df.columns]
    def row_to_fragment(row: pd.Series) -> str:
        return " | ".join([f"{c}: {row.get(c, '')}" for c in cols])
    df = df.copy()
    df["__fragment__"] = df.apply(row_to_fragment, axis=1)
    df["__row_id__"] = df.index.astype(str)
    return df

@st.cache_data(show_spinner=False)
def load_sheet_dataframe() -> pd.DataFrame:
    try:
        service_info = dict(st.secrets["gcp_service_account"])
    except Exception:
        st.error("Faltan credenciales en st.secrets['gcp_service_account'].")
        st.stop()
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_info(service_info, scopes=scopes)
    gc = gspread.authorize(creds)
    ws = gc.open_by_key(SHEET_ID).worksheet(WORKSHEET)
    records = ws.get_all_records()
    df = pd.DataFrame(records) if records else pd.DataFrame()
    df = normalize_columns(df)
    df = coerce_types(df)
    df = build_fragments(df)
    return df

def hash_dataframe(df: pd.DataFrame) -> str:
    data_bytes = df.drop(columns=[c for c in df.columns if c.startswith("__")], errors="ignore")\
                   .to_csv(index=False).encode("utf-8")
    return hashlib.md5(data_bytes).hexdigest()

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

# ---------------- Chroma o Fallback ----------------
def ensure_index(df: pd.DataFrame):
    current_hash = hash_dataframe(df)
    if "index_hash" not in st.session_state or st.session_state.index_hash != current_hash:
        st.session_state.index_hash = None  # invalida

    if CHROMA_AVAILABLE:
        # Chroma persistente
        client = chromadb.PersistentClient(path="./.chroma")
        coll_name = f"fenix_modelo_bot_{current_hash[:10]}"
        try:
            collection = client.get_collection(coll_name)
        except Exception:
            collection = client.get_or_create_collection(coll_name)
            # poblar colección
            openai_client = get_openai_client()
            docs = df["__fragment__"].fillna("").astype(str).tolist()
            ids = df["__row_id__"].tolist()
            metas = [{"row_id": rid} for rid in ids]
            with st.spinner("Construyendo índice vectorial (Chroma)…"):
                embs = embed_texts(openai_client, docs)
                collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        st.session_state.index_hash = current_hash
        st.session_state.index_backend = "chroma"
        st.session_state.collection = collection
    else:
        # Fallback simple: guardamos embeddings en .cache
        os.makedirs("./.cache", exist_ok=True)
        cache_file = f"./.cache/simple_index_{current_hash}.npz"
        if not os.path.exists(cache_file):
            openai_client = get_openai_client()
            docs = df["__fragment__"].fillna("").astype(str).tolist()
            ids = df["__row_id__"].tolist()
            with st.spinner("Construyendo índice vectorial (fallback)…"):
                embs = np.array(embed_texts(openai_client, docs), dtype=np.float32)
            np.savez(cache_file, embs=embs, ids=np.array(ids, dtype=object), docs=np.array(docs, dtype=object))
        data = np.load(cache_file, allow_pickle=True)
        st.session_state.simple_embs = data["embs"]
        st.session_state.simple_ids = data["ids"].tolist()
        st.session_state.simple_docs = data["docs"].tolist()
        st.session_state.index_hash = current_hash
        st.session_state.index_backend = "simple"

def retrieve_top_k(query: str, k: int = 8):
    if st.session_state.index_backend == "chroma":
        client = get_openai_client()
        q_emb = embed_texts(client, [query])[0]
        res = st.session_state.collection.query(query_embeddings=[q_emb], n_results=k, include=["documents","metadatas","distances"])
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        return docs, metas
    else:
        client = get_openai_client()
        q_emb = np.array(embed_texts(client, [query])[0], dtype=np.float32)
        A = st.session_state.simple_embs
        # cos sim
        qn = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
        sims = (An @ qn).ravel()
        idx = np.argsort(-sims)[:k]
        docs = [st.session_state.simple_docs[i] for i in idx]
        metas = [{"row_id": st.session_state.simple_ids[i]} for i in idx]
        return docs, metas

def build_prompt(context_chunks: List[str], question: str) -> str:
    ctx = "\n\n".join([f"- {c}" for c in context_chunks if c])
    return f"""
    Eres un agente de negocio experto para Fénix Automotriz. Responde SIEMPRE en español,
    con precisión, claridad y espíritu analítico, usando EXCLUSIVAMENTE la información de los fragmentos recuperados (Contexto).
    Si la respuesta no está en los datos, indícalo explícitamente y sugiere qué información faltaría.

    Además, considera el siguiente contexto de negocio para enriquecer el tono y las recomendaciones (no inventes datos):
    {BUSINESS_CONTEXT}

    Contexto recuperado:
    {ctx}

    Pregunta del usuario:
    {question}

    Instrucciones de estilo:
    - Respuesta directa y accionable.
    - Incluir KPIs y recomendaciones cuando aplique.
    - No inventar información fuera del contexto.
    """.strip()

def llm_answer(question: str, context_chunks: List[str]) -> str:
    client = get_openai_client()
    messages = [
        {"role": "system", "content": "Eres un analista de negocio senior. Respondes en español, con precisión y foco en decisiones."},
        {"role": "user", "content": build_prompt(context_chunks, question)},
    ]
    with st.spinner("Generando respuesta con LLM…"):
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.2)
    return resp.choices[0].message.content

def auto_visualize(df_rows: pd.DataFrame, question: str):
    if df_rows.empty:
        return
    q = unidecode(question).lower()
    date_pref = None
    for c in ["FECHA DE FACTURACION", "FECHA RECEPCION", "FECHA INGRESO PLANTA", "FECHA ENTREGA"]:
        if c in df_rows.columns:
            date_pref = c; break
    if ("factur" in q) and date_pref and date_pref in df_rows.columns:
        dft = df_rows.dropna(subset=[date_pref]).copy()
        if dft.empty: return
        dft["__mes__"] = dft[date_pref].dt.to_period("M").dt.to_timestamp()
        y_col = None
        for c in ["MONTO PRINCIPAL BRUTO [F]","MONTO PRINCIPAL NETO","IVA PRINCIPAL [F]"]:
            if c in dft.columns: y_col = c; break
        if y_col:
            agg = dft.groupby("__mes__", as_index=False)[y_col].sum()
            fig = px.bar(agg, x="__mes__", y=y_col, title="Montos facturados por mes (suma)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            agg = dft.groupby("__mes__", as_index=False).size().rename(columns={"size":"cantidad"})
            fig = px.bar(agg, x="__mes__", y="cantidad", title="Registros por mes")
            st.plotly_chart(fig, use_container_width=True)
        return
    if "dia" in q and "NUMERO DE DIAS EN PLANTA" in df_rows.columns:
        dft = df_rows.dropna(subset=["NUMERO DE DIAS EN PLANTA"]).copy()
        if not dft.empty:
            fig = px.histogram(dft, x="NUMERO DE DIAS EN PLANTA", nbins=20, title="Distribución de días en planta")
            st.plotly_chart(fig, use_container_width=True)
        return
    if "facturad" in q and "FACTURADO" in df_rows.columns:
        agg = df_rows["FACTURADO"].fillna("Desconocido").value_counts().reset_index()
        agg.columns = ["FACTURADO","cantidad"]
        fig = px.pie(agg, names="FACTURADO", values="cantidad", title="Distribución FACTURADO")
        st.plotly_chart(fig, use_container_width=True)
        return
    numeric_candidates = [c for c in df_rows.columns if c in NUMERIC_FIELDS_CANDIDATES]
    if numeric_candidates:
        y_col = numeric_candidates[0]
        dft = df_rows.copy()
        dft["__idx__"] = range(1, len(dft)+1)
        fig = px.bar(dft, x="__idx__", y=y_col, title=f"Valores de {y_col} (registros recuperados)")
        st.plotly_chart(fig, use_container_width=True)

# ---------------- UI ----------------
st.title("🔥 Fénix Automotriz — Agente de Negocio (RAG)")
st.caption("Google Sheets → Recuperación semántica → LLM (con fallback si Chroma no está disponible)")

with st.sidebar:
    st.subheader("⚙️ Parámetros")
    top_k = st.slider("Máx. fragmentos a recuperar", 3, 15, 8, 1)
    force_reindex = st.button("🔁 Reconstruir índice")

    st.markdown("---")
    st.subheader("📦 Estado del backend vectorial")
    st.write("Disponible ChromaDB: **{}**".format("Sí" if CHROMA_AVAILABLE else "No"))

if "messages" not in st.session_state: st.session_state.messages = []
df = load_sheet_dataframe()
if df.empty:
    st.warning("La hoja MODELO_BOT no tiene registros o no se pudo leer.")
    st.stop()

if force_reindex:
    for k in ["index_hash","collection","simple_embs","simple_ids","simple_docs"]:
        st.session_state.pop(k, None)

ensure_index(df)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Haz tu pregunta (ej.: vehículos facturados por mes, días en planta, etc.)")
if question:
    docs, metas = retrieve_top_k(question, k=top_k)
    row_indices = [int(m.get("row_id","0")) for m in metas if "row_id" in m]
    sel = df.iloc[row_indices].copy() if row_indices else pd.DataFrame()
    answer = llm_answer(question, docs)
    st.session_state.messages += [{"role":"user","content":question},{"role":"assistant","content":answer}]
    with st.chat_message("assistant"):
        st.markdown(answer)
        if not sel.empty:
            show_cols = [c for c in CANONICAL_FIELDS if c in sel.columns]
            st.markdown("**Registros relevantes (MODELO_BOT):**")
            st.dataframe(sel[show_cols] if show_cols else sel, use_container_width=True, hide_index=True)
            auto_visualize(sel, question)
