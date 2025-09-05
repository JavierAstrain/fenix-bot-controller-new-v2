# -*- coding: utf-8 -*-
"""
Fénix Automotriz — Agente de Negocio (RAG) en Streamlit
Fix4: Hibridación semántica + filtros deterministas (Pandas) para respuestas NUMÉRICAS correctas.
- Interpreta consultas comunes ("sin facturar", "entregados", meses, rangos de fechas, etc.)
- Aplica filtros sobre el DataFrame y calcula métricas (conteos/sumas/promedios)
- Luego usa RAG (Chroma o fallback) solo para contexto textual y explicación con LLM
"""
import os, re, json, hashlib
from typing import List, Dict, Tuple
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

# ---------------- Credenciales Google ----------------
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
        if mapped: new_cols[c] = mapped
    return df.rename(columns=new_cols)

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in DATE_FIELDS_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True, infer_datetime_format=True)
    for col in NUMERIC_FIELDS_CANDIDATES:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r"[.$ ]", "", regex=True).str.replace(",", ".", regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # normalizar FACTURADO (string)
    if "FACTURADO" in df.columns:
        df["FACTURADO"] = df["FACTURADO"].astype(str).str.strip().str.upper().replace({
            "TRUE":"SI","FALSE":"NO","1":"SI","0":"NO"
        })
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

# ------------- Backend vectorial -------------
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

# ------------- Intérprete de consultas (filtros deterministas) -------------
SP_MONTHS = {
    "enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,
    "julio":7,"agosto":8,"septiembre":9,"setiembre":9,"octubre":10,"noviembre":11,"diciembre":12
}

def parse_date_mentions(q: str) -> Tuple[pd.Timestamp, pd.Timestamp, str]:
    """Devuelve (start, end, label) si detecta un mes/rango; si no, (None,None,"")."""
    ql = unidecode(q).lower()
    # Rango explícito dd/mm/yyyy - dd/mm/yyyy
    m = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}).{0,10}(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", ql)
    if m:
        s = pd.to_datetime(m.group(1), dayfirst=True, errors="coerce")
        e = pd.to_datetime(m.group(2), dayfirst=True, errors="coerce")
        if pd.notna(s) and pd.notna(e):
            return s, e, f"entre {s.date()} y {e.date()}"
    # Mes + año
    for name, num in SP_MONTHS.items():
        if name in ql:
            # ¿año?
            m2 = re.search(rf"{name}\s+(\d{{4}})", ql)
            if m2:
                y = int(m2.group(1))
            else:
                y = pd.Timestamp.today().year
            start = pd.Timestamp(year=y, month=num, day=1)
            end = (start + pd.offsets.MonthEnd(0))
            return start, end, f"{name.capitalize()} {y}"
    # Este mes / mes pasado
    if "este mes" in ql or "actual" in ql:
        today = pd.Timestamp.today()
        start = pd.Timestamp(year=today.year, month=today.month, day=1)
        end = (start + pd.offsets.MonthEnd(0))
        return start, end, "este mes"
    if "mes pasado" in ql or "ultimo mes" in ql or "último mes" in ql:
        today = pd.Timestamp.today() - pd.offsets.MonthBegin(1)
        start = pd.Timestamp(year=today.year, month=today.month, day=1)
        end = (start + pd.offsets.MonthEnd(0))
        return start, end, "mes pasado"
    return None, None, ""

def build_mask(df: pd.DataFrame, q: str) -> Tuple[pd.Series, Dict]:
    """Devuelve (mask, info) con filtros aplicados y metadatos para debug."""
    qn = unidecode(q).lower()
    mask = pd.Series(True, index=df.index)
    info = {}

    # Sin facturar
    if any(k in qn for k in ["sin factur", "no factur", "no facturados", "pendiente de factur"]):
        info["sin_factura"] = True
        m1 = df["NUMERO DE FACTURA"].astype(str).str.strip().isin(["", "nan", "None", "0"]) if "NUMERO DE FACTURA" in df.columns else True
        m2 = df["FACTURADO"].astype(str).str.upper().isin(["NO", "PENDIENTE", "NAN"]) if "FACTURADO" in df.columns else True
        mask &= (m1 | m2)

    # Entregados
    if any(k in qn for k in ["entregado", "entregados"]):
        if "FECHA ENTREGA" in df.columns:
            info["entregados"] = True
            mask &= df["FECHA ENTREGA"].notna()

    # En planta
    if "en planta" in qn or "dias en planta" in qn or "días en planta" in qn:
        if "NUMERO DE DIAS EN PLANTA" in df.columns:
            info["en_planta"] = True
            # si pide umbrales: >, <, >=, <= N
            m = re.search(r"(?:>=|<=|>|<)\s*(\d+)", qn)
            if m:
                val = int(m.group(1))
                if ">=" in qn: mask &= (df["NUMERO DE DIAS EN PLANTA"] >= val)
                elif "<=" in qn: mask &= (df["NUMERO DE DIAS EN PLANTA"] <= val)
                elif ">" in qn: mask &= (df["NUMERO DE DIAS EN PLANTA"] > val)
                elif "<" in qn: mask &= (df["NUMERO DE DIAS EN PLANTA"] < val)

    # Por mes/rango usando fechas de facturación si habla de facturas; sino recepción
    ds, de, label = parse_date_mentions(q)
    if ds is not None and de is not None:
        info["periodo"] = label
        date_col = "FECHA DE FACTURACION" if "factur" in qn else ("FECHA RECEPCION" if "recepcion" in qn or "recepción" in qn else None)
        if date_col is None:
            # fallback: preferimos FECHA DE FACTURACION y si no existe, FECHA INGRESO PLANTA
            date_col = "FECHA DE FACTURACION" if "FECHA DE FACTURACION" in df.columns else "FECHA INGRESO PLANTA"
        if date_col in df.columns:
            mask &= df[date_col].between(ds, de)

    # Por patente específica
    mpat = re.search(r"\b([A-Z]{2,3}-?\d{2,3})\b", q.upper())
    if mpat and "PATENTE" in df.columns:
        info["patente"] = mpat.group(1)
        mask &= df["PATENTE"].astype(str).str.upper().str.replace("-", "") == info["patente"].replace("-", "")

    return mask, info

def compute_metrics(df: pd.DataFrame) -> Dict:
    out = {"cantidad": len(df)}
    if "MONTO PRINCIPAL BRUTO [F]" in df.columns:
        out["monto_bruto"] = float(df["MONTO PRINCIPAL BRUTO [F]"].fillna(0).sum())
    if "MONTO PRINCIPAL NETO" in df.columns:
        out["monto_neto"] = float(df["MONTO PRINCIPAL NETO"].fillna(0).sum())
    if "IVA PRINCIPAL [F]" in df.columns:
        out["iva"] = float(df["IVA PRINCIPAL [F]"].fillna(0).sum())
    if "NUMERO DE DIAS EN PLANTA" in df.columns:
        col = df["NUMERO DE DIAS EN PLANTA"].dropna()
        if not col.empty:
            out["dias_planta_prom"] = float(col.mean())
            out["dias_planta_p95"] = float(col.quantile(0.95))
    return out

def metrics_to_text(m: Dict) -> str:
    if not m: return ""
    parts = [f"- Registros: {m.get('cantidad', 0):,.0f}"]
    if "monto_bruto" in m: parts.append(f"- Monto bruto: ${m['monto_bruto']:,.0f}".replace(",", "."))
    if "monto_neto" in m: parts.append(f"- Monto neto: ${m['monto_neto']:,.0f}".replace(",", "."))
    if "iva" in m: parts.append(f"- IVA: ${m['iva']:,.0f}".replace(",", "."))
    if "dias_planta_prom" in m: parts.append(f"- Días en planta (prom): {m['dias_planta_prom']:.1f}")
    if "dias_planta_p95" in m: parts.append(f"- Días en planta (p95): {m['dias_planta_p95']:.1f}")
    return "\n".join(parts)

# ------------- Prompting -------------
def build_prompt(context_chunks: List[str], question: str, metrics_text: str, filters_info: Dict) -> str:
    ctx = "\n\n".join([f"- {c}" for c in context_chunks if c])
    filtros = ", ".join([f"{k}: {v}" for k, v in filters_info.items()]) if filters_info else "—"
    return f"""
    Eres un analista de negocio de Fénix Automotriz. Responde SIEMPRE en español y
    **usa los siguientes MÉTRICOS ya calculados** como verdad de referencia. No los contradigas.

    Filtros aplicados (deterministas sobre DataFrame): {filtros}
    Métricos calculados:
    {metrics_text or '—'}

    Usa el siguiente contexto (muestras de registros) solo para enriquecer y ejemplificar, no para cambiar los totales:
    {ctx}

    Pregunta del usuario:
    {question}

    Instrucciones:
    - Da respuesta directa con los totales y, si corresponde, agrega insights, riesgos y recomendaciones.
    - Si no hay registros para el filtro, dilo y sugiere alternativas.
    - Sé claro y breve, con bullets cuando ayude.
    """.strip()

def llm_answer(question: str, context_chunks: List[str], metrics_text: str, filters_info: Dict) -> str:
    client = get_openai_client()
    messages = [
        {"role": "system", "content": "Eres un analista financiero/operacional senior. Respondes en español con precisión."},
        {"role": "user", "content": build_prompt(context_chunks, question, metrics_text, filters_info)},
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
st.caption("Google Sheets → Filtros deterministas + Recuperación semántica → LLM")

with st.sidebar:
    st.subheader("⚙️ Parámetros")
    top_k = st.slider("Máx. fragmentos a recuperar (contexto)", 6, 30, 12, 1)
    force_reindex = st.button("🔁 Reconstruir índice")
    st.markdown("---")
    st.subheader("🧪 Debug")
    show_debug = st.checkbox("Mostrar detalles de filtros y métricas", value=False)
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

question = st.chat_input("Haz tu pregunta (p. ej.: 'vehículos entregados no facturados en agosto 2024', 'promedio días en planta > 15', 'monto facturado este mes')")
if question:
    # 1) Filtros deterministas
    mask, finfo = build_mask(df, question)
    filtered = df[mask].copy()
    metrics = compute_metrics(filtered)
    metrics_text = metrics_to_text(metrics)

    # 2) Contexto: usa fragmentos de la muestra filtrada si hay; si no, RAG puro
    if not filtered.empty:
        sample = filtered.sample(min(len(filtered), top_k), random_state=42)
        docs = sample["__fragment__"].tolist()
        metas = [{"row_id": rid} for rid in sample["__row_id__"].tolist()]
        row_indices = sample.index.tolist()
    else:
        docs, metas = retrieve_top_k(question, k=top_k)
        row_indices = [int(m.get("row_id","0")) for m in metas if "row_id" in m]

    sel = df.loc[row_indices].copy() if row_indices else pd.DataFrame()

    # 3) Responder
    answer = llm_answer(question, docs, metrics_text, finfo)
    st.session_state.messages += [{"role":"user","content":question},{"role":"assistant","content":answer}]

    with st.chat_message("assistant"):
        if show_debug:
            with st.expander("🔎 Filtros aplicados / Métricas"):
                st.write(finfo)
                st.code(metrics_text or "—")
        st.markdown(answer)

        # Tabla y gráficos
        base_to_show = filtered if not filtered.empty else sel
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
