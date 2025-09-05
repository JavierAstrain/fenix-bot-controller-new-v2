# -*- coding: utf-8 -*-
"""
Fénix Automotriz — Agente de Negocio (RAG) con diagnóstico
- Chunking narrativo por fila para retrieval de alta calidad.
- Motor de reglas deterministas para 6 casos críticos.
- Modo diagnóstico: Top-3 fragmentos recuperados + prompt completo + filtros aplicados.
- RAG solo para contexto; los totales/listados se calculan en Pandas sobre TODA la planilla.
"""
import os, re, json, hashlib, datetime as dt
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

st.set_page_config(page_title="Fénix | Agente de Negocio (RAG + Diagnóstico)", page_icon="🔥", layout="wide")

SHEET_ID = "1SaXuzhY_sJ9Tk9MOLDLAI4OVdsNbCP-X4L8cP15yTqo"
WORKSHEET = "MODELO_BOT"

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
    "numero factura": "NUMERO DE FACTURA",
    "nro factura": "NUMERO DE FACTURA",
    "n° factura": "NUMERO DE FACTURA",
}
DATE_FIELDS = [
    "FECHA INGRESO PLANTA","FECHA SALIDA PLANTA","FECHA INSPECCIÓN",
    "FECHA RECEPCION","FECHA ENTREGA","FECHA DE FACTURACION","FECHA DE PAGO FACTURA",
]
NUM_FIELDS = [
    "MONTO PRINCIPAL NETO","IVA PRINCIPAL [F]","MONTO PRINCIPAL BRUTO [F]",
    "NUMERO DE DIAS EN PLANTA","DIAS EN DOMINIO","CANTIDAD DE VEHICULO","DIAS DE PAGO DE FACTURA",
]

BUSINESS_CONTEXT = """
Compañía: Fénix Automotriz (Chile). Empresa familiar fundada en 2017, dedicada a reparación de carrocería y pintura.
Ejes: Experiencia excepcional, Excelencia operacional, Transformación tecnológica, Innovación, Expansión nacional.
Misión: “Entregar un servicio de reparación transparente, de calidad y puntual...”.
Visión 2026: “Ser el servicio de reparación automotriz preferido...”.
Proceso: Presupuesto → Recepción → Desarme → Desabolladura → Preparación → Pintura → Pulido → Lavado/entrega.
Roles: Gerencia, Planificación y Control, Adm/Finanzas, Comercial, Marketing, Líder Unidad, Jefe de Taller, Desarmador, Desabollador, Pintor, etc.
""".strip()

# ---------- Credenciales ----------
def _safe_secret_keys():
    try:
        return list(st.secrets._secrets.keys())
    except Exception:
        try: return list(st.secrets.keys())
        except Exception: return []

def _load_sa() -> Dict[str,Any]:
    for k in ["gcp_service_account","service_account","google_service_account"]:
        try:
            obj = st.secrets[k]
            if isinstance(obj, dict) and "client_email" in obj and "private_key" in obj:
                return dict(obj)
        except Exception: pass
    for k in ["GOOGLE_CREDENTIALS","GOOGLE_CREDENTIALS_JSON","SERVICE_ACCOUNT_JSON","gcp_service_account_json"]:
        try:
            s = st.secrets.get(k,"")
            if isinstance(s,str) and s.strip().startswith("{"): return json.loads(s)
        except Exception: pass
    for k in ["GOOGLE_CREDENTIALS","GOOGLE_CREDENTIALS_JSON"]:
        try:
            s = os.getenv(k,"")
            if isinstance(s,str) and s.strip().startswith("{"): return json.loads(s)
        except Exception: pass
    return {}

def get_gspread_client():
    info = _load_sa()
    if not info:
        st.error("Faltan credenciales de Google Service Account. Claves presentes: **{}**".format(", ".join(_safe_secret_keys())))
        st.stop()
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)

# ---------- Utilidades DF ----------
def _norm(s: str) -> str:
    return re.sub(r"\s+"," ", unidecode(str(s)).strip().lower())

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        key = _norm(c)
        for can in CANONICAL_FIELDS:
            if _norm(can) == key: mapping[c]=can
        if key in COL_ALIASES: mapping[c]=COL_ALIASES[key]
    return df.rename(columns=mapping)

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in DATE_FIELDS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True, infer_datetime_format=True)
    for col in NUM_FIELDS:
        if col in df.columns:
            df[col] = (df[col].astype(str)
                        .str.replace(r"[.$ ]","",regex=True)
                        .str.replace(",",".",regex=True))
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "FACTURADO" in df.columns:
        df["FACTURADO"] = (df["FACTURADO"].astype(str).str.strip().str.upper()
                           .replace({"TRUE":"SI","FALSE":"NO","1":"SI","0":"NO"}))
    return df

def _fmt(val: Any) -> str:
    if pd.isna(val): return ""
    if isinstance(val, pd.Timestamp): return val.strftime("%Y-%m-%d")
    if isinstance(val, (int,float)): return f"{val:.2f}".rstrip("0").rstrip(".")
    return str(val)

def row_to_fragment(row: pd.Series, present: List[str]) -> str:
    return "\n".join([f"{c}: {_fmt(row.get(c,''))}" for c in present])

def build_fragments(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cols = [c for c in CANONICAL_FIELDS if c in df.columns]
    frags, ids = [], []
    for idx, row in df.iterrows():
        frags.append(row_to_fragment(row, cols))
        ids.append(str(idx))
    return frags, ids

@st.cache_data(show_spinner=False)
def load_df() -> pd.DataFrame:
    gc = get_gspread_client()
    ws = gc.open_by_key(SHEET_ID).worksheet(WORKSHEET)
    recs = ws.get_all_records()
    df = pd.DataFrame(recs) if recs else pd.DataFrame()
    df = normalize_columns(df)
    df = coerce_types(df)
    return df

# ---------- OpenAI ----------
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY",""))
    if not api_key:
        st.error("OPENAI_API_KEY no está configurada.")
        st.stop()
    return OpenAI(api_key=api_key)

def embed_texts(client: OpenAI, texts: List[str], model="text-embedding-3-small") -> List[List[float]]:
    out=[]; B=96
    for i in range(0,len(texts),B):
        chunk=texts[i:i+B]
        resp=client.embeddings.create(model=model, input=chunk)
        out.extend([d.embedding for d in resp.data])
    return out

# ---------- Index vectorial ----------
def _hash_frags(frags: List[str]) -> str:
    h = hashlib.md5()
    for t in frags: h.update(t.encode("utf-8"))
    return h.hexdigest()

def ensure_index(frags: List[str], ids: List[str]):
    h = _hash_frags(frags)
    if st.session_state.get("index_hash")==h: return
    if CHROMA_AVAILABLE:
        client = chromadb.PersistentClient(path="./.chroma")
        name=f"fenix_{h[:10]}"
        try: coll = client.get_collection(name)
        except Exception:
            coll = client.get_or_create_collection(name)
            oai = get_openai_client()
            with st.spinner("Construyendo índice (Chroma)…"):
                embs = embed_texts(oai, frags)
                coll.add(ids=ids, documents=frags, metadatas=[{"row_id":i} for i in ids], embeddings=embs)
        st.session_state.index_backend="chroma"; st.session_state.collection=coll
    else:
        os.makedirs("./.cache", exist_ok=True)
        path=f"./.cache/simple_{h}.npz"
        if not os.path.exists(path):
            oai=get_openai_client()
            with st.spinner("Construyendo índice (fallback)…"):
                embs=np.array(embed_texts(oai, frags), dtype=np.float32)
            np.savez(path, embs=embs, ids=np.array(ids,dtype=object), docs=np.array(frags,dtype=object))
        d=np.load(path, allow_pickle=True)
        st.session_state.simple_embs=d["embs"]; st.session_state.simple_ids=d["ids"].tolist(); st.session_state.simple_docs=d["docs"].tolist()
        st.session_state.index_backend="simple"
    st.session_state.index_hash=h

def retrieve_top(query: str, k=3) -> Tuple[List[str], List[str]]:
    oai=get_openai_client()
    q=np.array(embed_texts(oai,[query])[0], dtype=np.float32)
    if st.session_state.get("index_backend")=="chroma":
        res=st.session_state.collection.query(query_embeddings=[q.tolist()], n_results=k, include=["documents","metadatas"])
        docs=res.get("documents",[[]])[0]; metas=res.get("metadatas",[[]])[0]; ids=[m.get("row_id","") for m in metas]
        return docs, ids
    A=st.session_state.simple_embs
    qn=q/(np.linalg.norm(q)+1e-9); An=A/(np.linalg.norm(A,axis=1,keepdims=True)+1e-9)
    idx=np.argsort(-(An@qn).ravel())[:k]
    return [st.session_state.simple_docs[i] for i in idx], [st.session_state.simple_ids[i] for i in idx]

# ---------- Prompts / LLM ----------
def system_prompt() -> str:
    return f"""
Eres un CONSULTOR/ANALISTA de Fénix Automotriz.
Usa SOLO el “Contexto proporcionado”. Si falta info, responde exactamente: "No tengo la información necesaria en los datos".
Puedes hacer sumas, promedios y min/max sobre los registros del contexto. Sé conciso y accionable.
Contexto de negocio:
{BUSINESS_CONTEXT}
""".strip()

def build_user_prompt(question: str, frags: List[str]) -> str:
    ctx="\n\n-----\n\n".join(frags) if frags else "(sin contexto)"
    return f"Pregunta: {question}\n\nContexto proporcionado:\n{ctx}"

def llm_answer(question: str, frags: List[str]) -> str:
    client=get_openai_client()
    msgs=[{"role":"system","content":system_prompt()},
          {"role":"user","content":build_user_prompt(question, frags)}]
    resp=client.chat.completions.create(model="gpt-4o-mini", messages=msgs, temperature=0.2)
    return resp.choices[0].message.content, msgs

# ---------- Motor determinista para 6 casos ----------
SPANISH_MONTHS = {
    "enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,
    "julio":7,"agosto":8,"septiembre":9,"setiembre":9,"octubre":10,"noviembre":11,"diciembre":12
}

def parse_month(q: str) -> Optional[int]:
    qn=_norm(q)
    for m,n in SPANISH_MONTHS.items():
        if m in qn: return n
    return None

def parse_days_window(q: str, default_days=7) -> int:
    qn=_norm(q)
    m=re.search(r"proxim[oa]s?\s+(\d+)\s+dias", qn)
    if m: return int(m.group(1))
    m=re.search(r"siguient[ea]s?\s+(\d+)\s+dias", qn)
    if m: return int(m.group(1))
    return default_days

def rule_engine(question: str, df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    q=_norm(question)
    today=dt.date.today()

    # A: entregados no facturados
    if "entreg" in q and ("no factur" in q or "aun" in q or "aún" in q):
        mask_ent = df.get("FECHA ENTREGA").notna() if "FECHA ENTREGA" in df.columns else pd.Series(False,index=df.index)
        mask_no_fact = pd.Series(True,index=df.index)
        if "FACTURADO" in df.columns:
            mask_no_fact &= df["FACTURADO"].astype(str).str.upper().isin(["NO","PENDIENTE","NAN",""])
        if "NUMERO DE FACTURA" in df.columns:
            mask_no_fact |= df["NUMERO DE FACTURA"].astype(str).str.strip().eq("")
        if "FECHA DE FACTURACION" in df.columns:
            mask_no_fact |= df["FECHA DE FACTURACION"].isna()
        res=df[mask_ent & mask_no_fact].copy()
        cols=[c for c in ["OT","PATENTE","FECHA ENTREGA","FACTURADO","NUMERO DE FACTURA","FECHA DE FACTURACION","MONTO PRINCIPAL BRUTO [F]"] if c in df.columns]
        return "Vehículos entregados aún NO facturados", res[cols] if cols else res

    # B: facturas a pagar próximos días
    if ("pagar" in q or "pago" in q) and "factur" in q:
        days=parse_days_window(q, default_days=30)
        if "FECHA DE PAGO FACTURA" not in df.columns: return "No tengo la información necesaria en los datos", pd.DataFrame()
        mask=df["FECHA DE PAGO FACTURA"].notna() & (df["FECHA DE PAGO FACTURA"].dt.date>=today) & (df["FECHA DE PAGO FACTURA"].dt.date<=today+dt.timedelta(days=days))
        res=df[mask].copy().sort_values("FECHA DE PAGO FACTURA", ascending=True)
        cols=[c for c in ["NUMERO DE FACTURA","FECHA DE PAGO FACTURA","MONTO PRINCIPAL BRUTO [F]","NOMBRE CLIENTE"] if c in df.columns]
        return f"Facturas a pagar en los próximos {days} días", res[cols] if cols else res

    # C: días en taller (en proceso) Top 10
    if ("dias" in q or "días" in q) and ("taller" in q or "planta" in q):
        if "FECHA INGRESO PLANTA" not in df.columns: return "No tengo la información necesaria en los datos", pd.DataFrame()
        mask_proc = df["ESTADO SERVICIO"].astype(str).str.upper().str.contains("PROCES", na=False) if "ESTADO SERVICIO" in df.columns else pd.Series(True,index=df.index)
        base=df[mask_proc & df["FECHA INGRESO PLANTA"].notna()].copy()
        base["DIAS_ACTUALES"]= (pd.Timestamp(today) - base["FECHA INGRESO PLANTA"]).dt.days
        res=base.sort_values("DIAS_ACTUALES", ascending=False).head(10)
        cols=[c for c in ["OT","PATENTE","FECHA INGRESO PLANTA","DIAS_ACTUALES","NOMBRE CLIENTE"] if c in res.columns]
        return "Top 10 vehículos con más días en taller (En Proceso)", res[cols] if cols else res

    # D: facturación por mes / tipo de cliente
    if "factur" in q and ("mes" in q or "marzo" in q or "enero" in q or "febrero" in q or "abril" in q or "mayo" in q or "junio" in q or "julio" in q or "agosto" in q or "septiembre" in q or "octubre" in q or "noviembre" in q or "diciembre" in q) and ("tipo de cliente" in q or "cliente" in q):
        month=parse_month(q) or dt.date.today().month
        if "FECHA DE FACTURACION" not in df.columns or "MONTO PRINCIPAL NETO" not in df.columns or "TIPO CLIENTE" not in df.columns:
            return "No tengo la información necesaria en los datos", pd.DataFrame()
        base=df[df["FECHA DE FACTURACION"].dt.month==month].copy()
        res = base.groupby("TIPO CLIENTE", dropna=False)["MONTO PRINCIPAL NETO"].sum().reset_index().rename(columns={"MONTO PRINCIPAL NETO":"MONTO_NETO"})
        return f"Facturación (monto neto) por TIPO CLIENTE en el mes {month}", res

    # E: entregas próximos días (solo logística)
    if ("entreg" in q) and ("proxim" in q or "siguient" in q or "dias" in q or "días" in q):
        if "FECHA ENTREGA" not in df.columns: return "No tengo la información necesaria en los datos", pd.DataFrame()
        days=parse_days_window(q, default_days=14)
        mask=df["FECHA ENTREGA"].notna() & (df["FECHA ENTREGA"].dt.date>=today) & (df["FECHA ENTREGA"].dt.date<=today+dt.timedelta(days=days))
        res=df[mask].copy().sort_values("FECHA ENTREGA", ascending=True)
        cols=[c for c in ["OT","PATENTE","FECHA ENTREGA","NOMBRE CLIENTE","PROCESO"] if c in res.columns]
        return f"Vehículos a entregar en los próximos {days} días", res[cols] if cols else res

    # F: autos en taller sin aprobación
    if ("sin aprob" in q or ("aprob" in q and "sin" in q)) or ("taller" in q and "aprob" in q):
        mask_serv = df["ESTADO SERVICIO"].astype(str).str.upper().str.contains("PROCES", na=False) if "ESTADO SERVICIO" in df.columns else pd.Series(True,index=df.index)
        mask_pres = ~(df["ESTADO PRESUPUESTO"].astype(str).str.upper().isin(["APROBADO","PERDIDO"])) if "ESTADO PRESUPUESTO" in df.columns else pd.Series(True,index=df.index)
        res=df[mask_serv & mask_pres].copy()
        cols=[c for c in ["OT","PATENTE","ESTADO SERVICIO","ESTADO PRESUPUESTO","NOMBRE CLIENTE"] if c in res.columns]
        return "Autos en taller sin aprobación de presupuesto", res[cols] if cols else res

    return "", pd.DataFrame()

# ---------- UI ----------
st.title("🔥 Fénix Automotriz — Agente de Negocio (RAG + Diagnóstico)")
st.caption("Determinista en Pandas para KPIs + RAG para redacción/soporte. Modo diagnóstico incluido.")

with st.sidebar:
    st.subheader("⚙️ Parámetros")
    top_k = st.slider("Top-K fragmentos para contexto", 3, 15, 6, 1)
    diag = st.checkbox("🔎 Modo diagnóstico (mostrar retrieval y prompt)", value=True)
    force = st.button("🔁 Reconstruir índice")

df = load_df()
if df.empty:
    st.error("La hoja MODELO_BOT no tiene registros o no se pudo leer.")
    st.stop()

frags, ids = build_fragments(df)
if force:
    for k in ["index_hash","collection","simple_embs","simple_ids","simple_docs"]: st.session_state.pop(k, None)
ensure_index(frags, ids)

if "messages" not in st.session_state: st.session_state.messages=[]

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

question = st.chat_input("Escribe tu pregunta…")
if question:
    # 1) Motor determinista (si aplica)
    title, table = rule_engine(question, df)

    # 2) Retrieval para contexto (siempre)
    top_docs, top_ids = retrieve_top(question, k=top_k)

    # 3) LLM para redacción (NO para números)
    answer_text, prompt_msgs = llm_answer(question, top_docs)

    # Guardar conversación
    st.session_state.messages += [{"role":"user","content":question},
                                  {"role":"assistant","content":answer_text}]

    with st.chat_message("assistant"):
        if diag:
            with st.expander("🧩 Top-3 fragmentos recuperados"):
                for i, d in enumerate(top_docs[:3], 1):
                    st.markdown(f"**Fragmento {i}**\n\n```\n{d}\n```")
            with st.expander("🧪 Prompt exacto enviado al LLM"):
                st.write("**System Prompt:**")
                st.code(prompt_msgs[0]["content"])
                st.write("**User Prompt:**")
                st.code(prompt_msgs[1]["content"])

        # Respuesta
        if title: st.markdown(f"### ✅ Resultado determinista: {title}")
        else: st.markdown("### ℹ️ Resultado (sin regla específica)")

        st.markdown(answer_text)

        if not table.empty:
            st.dataframe(table, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Descargar resultado (CSV)",
                               data=table.to_csv(index=False).encode("utf-8"),
                               file_name="resultado.csv", mime="text/csv")
        else:
            st.info("No se encontraron filas que cumplan esos criterios o faltan columnas requeridas.")
