# -*- coding: utf-8 -*-
"""
Fénix Automotriz — Agente de Negocio (RAG) en Streamlit
Fix6: Prompts refinados + planner con guía de dominio + ejecución determinista.
- System prompt con rol explícito (consultor/analista de Fénix) + misión/visión/procesos.
- Prompt de consulta con instrucciones estrictas de uso de datos (No inventar / solo planilla).
- Planner (function-calling) instruido con reglas de negocio para seleccionar columnas de fecha
  (p.ej., facturación -> FECHA DE FACTURACION) y filtros comunes.
- Mantiene: credenciales robustas, fallback sin Chroma, visualizaciones, descarga CSV.
"""
import os, re, json, hashlib, math
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI
from unidecode import unidecode

CHROMA_AVAILABLE = True
try:
    import chromadb
except Exception:
    CHROMA_AVAILABLE = False

st.set_page_config(page_title="Fénix | Agente de Negocio (RAG + Planner)", page_icon="🔥", layout="wide")

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
    "facturado?": "FACTURADO",
    "numero factura": "NUMERO DE FACTURA",
    "nro factura": "NUMERO DE FACTURA",
    "n° factura": "NUMERO DE FACTURA",
}

DATE_FIELDS_CANDIDATES = [
    "FECHA INGRESO PLANTA","FECHA SALIDA PLANTA","FECHA INSPECCIÓN","FECHA RECEPCION",
    "FECHA ENTREGA","FECHA DE FACTURACION","FECHA DE PAGO FACTURA",
]

NUMERIC_FIELDS_CANDIDATES = [
    "MONTO PRINCIPAL NETO","IVA PRINCIPAL [F]","MONTO PRINCIPAL BRUTO [F]",
    "NUMERO DE DIAS EN PLANTA","DIAS EN DOMINIO","CANTIDAD DE VEHICULO","DIAS DE PAGO DE FACTURA",
]

# ----------------- Credenciales Google -----------------
def _try_load_sa_from_secrets() -> Dict:
    for key in ["gcp_service_account", "service_account", "google_service_account"]:
        try:
            obj = st.secrets[key]
            if isinstance(obj, dict) and "client_email" in obj and "private_key" in obj:
                return dict(obj)
        except Exception:
            pass
    for key in [
        "gcp_service_account_json",
        "GOOGLE_CREDENTIALS_JSON",
        "GOOGLE_APPLICATION_CREDENTIALS_JSON",
        "GOOGLE_CREDENTIALS",
        "GCP_CREDENTIALS_JSON",
        "SERVICE_ACCOUNT_JSON",
    ]:
        try:
            s = st.secrets.get(key, "")
            if isinstance(s, str) and "{" in s and "}" in s:
                return json.loads(s)
        except Exception:
            pass
    for key in ["GOOGLE_CREDENTIALS_JSON", "GOOGLE_APPLICATION_CREDENTIALS_JSON", "GOOGLE_CREDENTIALS"]:
        try:
            s = os.getenv(key, "")
            if isinstance(s, str) and "{" in s and "}" in s:
                return json.loads(s)
        except Exception:
            pass
    return {}

def _secrets_keys_safe():
    try:
        return list(st.secrets._secrets.keys())
    except Exception:
        try:
            return list(st.secrets.keys())
        except Exception:
            return []

def get_gspread_client():
    service_info = _try_load_sa_from_secrets()
    if not service_info:
        st.error("Faltan credenciales de Google Service Account. Claves de secrets presentes: **{}**"
                 .format(", ".join(_secrets_keys_safe())))
        st.stop()
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_info(service_info, scopes=scopes)
    return gspread.authorize(creds)

# ----------------- Utilidades DF -----------------
def norm_text(s: str) -> str:
    s = unidecode(str(s)).lower().strip()
    s = re.sub(r"[\[\]\(\)]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    for c in df.columns:
        key = norm_text(c)
        for can in CANONICAL_FIELDS:
            if norm_text(can) == key:
                new_cols[c] = can
        if key in COL_ALIASES:
            new_cols[c] = COL_ALIASES[key]
    return df.rename(columns=new_cols)

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in DATE_FIELDS_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True, infer_datetime_format=True)
    for col in NUMERIC_FIELDS_CANDIDATES:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r"[.$ ]", "", regex=True).str.replace(",", ".", regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "FACTURADO" in df.columns:
        df["FACTURADO"] = df["FACTURADO"].astype(str).str.strip().str.upper().replace({
            "TRUE":"SI","FALSE":"NO","1":"SI","0":"NO"
        })
    return df

def build_fragments(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in CANONICAL_FIELDS if c in df.columns]
    def row_to_fragment(row: pd.Series) -> str:
        return " | ".join([f\"{c}: {row.get(c, '')}\" for c in cols])
    df = df.copy()
    df["__fragment__"] = df.apply(row_to_fragment, axis=1)
    df["__row_id__"] = df.index.astype(str)
    return df

@st.cache_data(show_spinner=False)
def load_sheet_dataframe() -> pd.DataFrame:
    gc = get_gspread_client()
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

# ----------------- OpenAI -----------------
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
    if not api_key:
        st.error("OPENAI_API_KEY no está configurada.")
        st.stop()
    return OpenAI(api_key=api_key)

# System Prompt reforzado (rol + reglas)
def system_prompt() -> str:
    return f"""
    Eres un CONSULTOR DE GESTIÓN y ANALISTA DE DATOS para Fénix Automotriz.
    Tu única fuente de verdad son los datos provistos desde la hoja MODELO_BOT.
    Contexto de negocio:
    {BUSINESS_CONTEXT}

    Reglas de oro:
    - Usa EXCLUSIVAMENTE los datos proporcionados; NO inventes ni alucines.
    - Si la información no está en los datos, di claramente: "No tengo la información necesaria en los datos".
    - Puedes realizar cálculos simples (sumas, promedios, min/max) sobre los registros filtrados que se te entregan.
    - Analiza fechas, montos y estados; al responder, sé conciso y directo.
    - Cuando el usuario pida tablas, listados o detalle, preséntalo de forma ordenada.
    - Si el usuario no especifica columna de fecha, prefiere:
        * facturación/ingresos → FECHA DE FACTURACION
        * ingresos a planta → FECHA INGRESO PLANTA
        * entregas → FECHA ENTREGA
        * recepción → FECHA RECEPCION
    - No contradigas los totales ya calculados por la aplicación si se te proveen.
    """.strip()

def embed_texts(client: OpenAI, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    out = []
    B = 96
    for i in range(0, len(texts), B):
        chunk = texts[i:i+B]
        resp = client.embeddings.create(model=model, input=chunk)
        out.extend([d.embedding for d in resp.data])
    return out

# ----------------- Vector index -----------------
def ensure_index(df: pd.DataFrame):
    current_hash = hash_dataframe(df)
    if "index_hash" not in st.session_state or st.session_state.index_hash != current_hash:
        st.session_state.index_hash = None
    if CHROMA_AVAILABLE:
        client = chromadb.PersistentClient(path="./.chroma")
        coll_name = f"fenix_modelo_bot_{current_hash[:10]}"
        try:
            collection = client.get_collection(coll_name)
        except Exception:
            collection = client.get_or_create_collection(coll_name)
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

def retrieve_top_k(query: str, k: int = 12):
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
        qn = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
        sims = (An @ qn).ravel()
        idx = np.argsort(-sims)[:k]
        docs = [st.session_state.simple_docs[i] for i in idx]
        metas = [{"row_id": st.session_state.simple_ids[i]} for i in idx]
        return docs, metas

# ----------------- Planner (function-calling) -----------------
def infer_schema(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    schema = {}
    for c in df.columns:
        if c.startswith("__"): continue
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            dtype = "date"; ex = [str(x.date()) for x in s.dropna().unique()[:5]]
        elif pd.api.types.is_numeric_dtype(s):
            dtype = "numeric"; ex = [float(x) for x in s.dropna().unique()[:5]]
        else:
            dtype = "text"; ex = [str(x) for x in s.dropna().astype(str).unique()[:5]]
        schema[c] = {"dtype": dtype, "examples": ex}
    return schema

def llm_make_plan(question: str, schema: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    client = get_openai_client()
    rules = (
        "Reglas para elegir columna de fecha:\n"
        "- Si la pregunta es sobre facturas/facturación/ingresos: usar 'FECHA DE FACTURACION'.\n"
        "- Si es sobre vehículos en planta o ingreso: 'FECHA INGRESO PLANTA'.\n"
        "- Si es sobre entregas: 'FECHA ENTREGA'.\n"
        "- Si es sobre recepción: 'FECHA RECEPCION'.\n"
        "Reglas de filtros:\n"
        "- 'sin facturar' => (FACTURADO in ['NO','PENDIENTE'] OR NUMERO DE FACTURA vacío).\n"
        "- 'entregados' => 'FECHA ENTREGA' not empty.\n"
        "- reconocer operadores >, >=, <, <= para 'NUMERO DE DIAS EN PLANTA' y montos.\n"
        "- no inventes columnas; usa solo nombres del esquema.\n"
    )
    schema_json = json.dumps(schema)[:6000]
    system = (
        "Eres un planificador de consultas para datos tabulares de Fénix Automotriz. "
        "Debes devolver un PLAN JSON (filters/aggregations/group_by) que otro componente ejecutará en Pandas. "
        "No incluyas explicación; solo el JSON a través de la función."
    )
    user = f"Esquema:\n{schema_json}\n\n{rules}\n\nPregunta del usuario:\n{question}"
    tools = [{
        "type": "function",
        "function": {
            "name": "emitir_plan",
            "description": "Devuelve el plan de consulta.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {"type":"string"},
                                "op": {"type": "string", "enum": ["eq","neq","gt","gte","lt","lte","contains","not_contains","in","not_in","empty","not_empty","between_dates"]},
                                "value": {"type":"array","items":{"type":"string"}}
                            },
                            "required": ["column","op"]
                        }
                    },
                    "date_column_preference": {"type": ["string","null"]},
                    "aggregations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "op": {"type": "string", "enum": ["count","sum","avg","min","max"]},
                                "column": {"type":"string"},
                                "alias": {"type":"string"}
                            },
                            "required": ["op"]
                        }
                    },
                    "group_by": {"type": ["string","null"]}
                },
                "required": ["filters","aggregations"]
            }
        }
    }]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        tools=tools,
        tool_choice={"type":"function","function":{"name":"emitir_plan"}}
    )
    msg = resp.choices[0].message
    plan = {"filters": [], "aggregations": [{"op":"count","alias":"cantidad"}], "date_column_preference": None, "group_by": None}
    try:
        if msg.tool_calls:
            args = msg.tool_calls[0].function.arguments
            cand = json.loads(args)
            if isinstance(cand, dict):
                plan.update(cand)
    except Exception:
        pass
    return plan

def map_column_name(col: str, available: List[str]) -> Optional[str]:
    if col in available: return col
    key = norm_text(col)
    for a in available:
        if norm_text(a) == key: return a
    if key in COL_ALIASES:
        cand = COL_ALIASES[key]
        if cand in available: return cand
    for a in available:
        if key in norm_text(a): return a
    return None

def execute_plan(df: pd.DataFrame, plan: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    mask = pd.Series(True, index=df.index)
    info = {"plan": plan, "applied_filters": []}
    cols = [c for c in df.columns if not c.startswith("__")]

    # aplicar filtros
    for f in plan.get("filters", []):
        col = map_column_name(str(f.get("column","")), cols)
        op = f.get("op","")
        vals = f.get("value", [])
        if not col or col not in df.columns:
            continue
        s = df[col]
        m = pd.Series(True, index=df.index)
        if op in ["eq","neq","contains","not_contains","in","not_in"]:
            sv = s.astype(str).str.upper().str.strip()
            vlist = [str(x).upper().strip() for x in vals] if isinstance(vals, list) else [str(vals).upper().strip()]
            if op == "eq": m = sv.isin(vlist)
            elif op == "neq": m = ~sv.isin(vlist)
            elif op == "contains":
                pat = "|".join([re.escape(v) for v in vlist]) if vlist else ""
                m = sv.str.contains(pat, na=False)
            elif op == "not_contains":
                pat = "|".join([re.escape(v) for v in vlist]) if vlist else ""
                m = ~sv.str.contains(pat, na=False)
            elif op == "in": m = sv.isin(vlist)
            elif op == "not_in": m = ~sv.isin(vlist)
        elif op in ["gt","gte","lt","lte"]:
            sn = pd.to_numeric(s, errors="coerce")
            v = float(vals[0]) if isinstance(vals, list) and vals else math.nan
            if op == "gt": m = sn > v
            if op == "gte": m = sn >= v
            if op == "lt": m = sn < v
            if op == "lte": m = sn <= v
        elif op in ["empty","not_empty"]:
            if op == "empty":
                m = s.isna() | (s.astype(str).str.strip() == "") | (s.astype(str).str.upper().isin(["NAN","NONE","NULL"]))
            else:
                m = ~(s.isna() | (s.astype(str).str.strip() == "") | (s.astype(str).str.upper().isin(["NAN","NONE","NULL"])))
        elif op == "between_dates":
            sd = pd.to_datetime(vals[0], dayfirst=True, errors="coerce")
            ed = pd.to_datetime(vals[1], dayfirst=True, errors="coerce")
            if pd.notna(sd) and pd.notna(ed):
                if pd.api.types.is_datetime64_any_dtype(s):
                    m = s.between(sd, ed)
                else:
                    s2 = pd.to_datetime(s, dayfirst=True, errors="coerce")
                    m = s2.between(sd, ed)
        mask &= m
        info["applied_filters"].append({"column": col, "op": op, "value": vals, "remaining": int(mask.sum())})

    filtered = df[mask].copy()

    # agregaciones
    aggs = plan.get("aggregations") or [{"op":"count","alias":"cantidad"}]
    out = {"cantidad": len(filtered)}
    for a in aggs:
        op = a.get("op")
        col = map_column_name(str(a.get("column","")), cols) if a.get("column") else None
        alias = a.get("alias") or (f"{op}_{col}" if col else op)
        if op == "count":
            out[alias] = int(len(filtered))
        elif op in ["sum","avg","min","max"] and col and col in filtered.columns:
            sn = pd.to_numeric(filtered[col], errors="coerce")
            if op == "sum": out[alias] = float(sn.sum())
            if op == "avg": out[alias] = float(sn.mean())
            if op == "min": out[alias] = float(sn.min())
            if op == "max": out[alias] = float(sn.max())
    info["metrics"] = out
    return filtered, info

# ----------------- Respuesta -----------------
def build_answer_prompt(context_chunks: List[str], question: str, metrics: Dict[str, Any]) -> str:
    ctx = "\n\n".join([f"- {c}" for c in context_chunks if c])
    metrics_lines = [f"{k}: {v}" for k,v in metrics.items()]
    return f"""
    Actúa como consultor/analista de Fénix Automotriz.
    Usa **únicamente** los datos proporcionados; si falta info, responde: "No tengo la información necesaria en los datos".
    Puedes realizar cálculos simples sobre los registros filtrados que se te entregan.
    Sé conciso y al punto. Si el usuario pide tablas o detalle, indícalo en tu respuesta; la app ya las mostrará aparte.

    Totales calculados por la app (no los contradigas):
    {chr(10).join(metrics_lines) if metrics_lines else '—'}

    Contexto (muestras de registros):
    {ctx}

    Pregunta:
    {question}
    """.strip()

def llm_answer(question: str, context_chunks: List[str], metrics: Dict[str, Any]) -> str:
    client = get_openai_client()
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": build_answer_prompt(context_chunks, question, metrics)},
    ]
    with st.spinner("Generando respuesta con LLM…"):
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.1)
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

# ----------------- UI -----------------
st.title("🔥 Fénix Automotriz — Agente de Negocio (RAG + Planner + Prompts refinados)")
st.caption("Planner (function-calling) → Filtros deterministas → Contexto semántico → Respuesta guiada por System Prompt")

with st.sidebar:
    st.subheader("⚙️ Parámetros")
    top_k = st.slider("Muestras de contexto (top_k)", 6, 30, 12, 1)
    force_reindex = st.button("🔁 Reconstruir índice")
    show_debug = st.checkbox("Mostrar plan y métricas", value=True)
    st.markdown("---")
    st.subheader("📦 Estado")
    st.write("Backend vectorial: **{}**".format("Chroma" if CHROMA_AVAILABLE else "Fallback simple"))

if "messages" not in st.session_state: st.session_state.messages = []
if force_reindex:
    for k in ["index_hash","collection","simple_embs","simple_ids","simple_docs"]:
        st.session_state.pop(k, None)

df = load_sheet_dataframe()
if df.empty:
    st.warning("La hoja MODELO_BOT no tiene registros o no se pudo leer.")
    st.stop()

ensure_index(df)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ej.: 'no facturados en agosto 2024', 'promedio días en planta > 15', 'monto bruto este mes'")
if question:
    schema = infer_schema(df)
    plan = llm_make_plan(question, schema)
    filtered, info = execute_plan(df, plan)

    # contexto
    if not filtered.empty:
        sample = filtered.sample(min(len(filtered), top_k), random_state=42)
        docs = sample["__fragment__"].tolist()
        base_to_show = filtered
    else:
        docs, _ = retrieve_top_k(question, k=top_k)
        base_to_show = pd.DataFrame()

    metrics = info.get("metrics", {"cantidad": int(len(filtered))})
    answer = llm_answer(question, docs, metrics)

    st.session_state.messages += [{"role":"user","content":question},{"role":"assistant","content":answer}]

    with st.chat_message("assistant"):
        if show_debug:
            with st.expander("🧩 Plan de consulta y métricas"):
                st.json(plan)
                st.write(info.get("applied_filters", []))
                st.json(metrics)
        st.markdown(answer)

        if not base_to_show.empty:
            st.markdown("**Registros relevantes (MODELO_BOT):**")
            show_cols = [c for c in CANONICAL_FIELDS if c in base_to_show.columns]
            st.dataframe(base_to_show[show_cols] if show_cols else base_to_show, use_container_width=True, hide_index=True)
            auto_visualize(base_to_show, question)
            st.download_button("⬇️ Descargar registros (CSV)",
                data=base_to_show.to_csv(index=False).encode("utf-8"),
                file_name="registros_filtrados.csv", mime="text/csv")
        else:
            st.info("No se encontraron registros para esos criterios.")
