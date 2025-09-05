# -*- coding: utf-8 -*-
"""
Fénix Automotriz — Agente de Negocio (RAG) con Búsqueda Híbrida (Filtros + Semántica)
- get_data_from_gsheet() devuelve el DataFrame CRUDO (sin transformaciones) para cumplir el requisito.
- Intent Parser (OpenAI function-calling) extrae filtros exactos desde la pregunta del usuario.
- Filtrado determinista en Pandas sobre una copia NORMALIZADA del DF crudo.
- RAG y vector search SOLO sobre el subconjunto filtrado (fragmentos narrativos por fila).
- Modo diagnóstico: filtros extraídos, reglas añadidas, filtros aplicados, top-3 fragmentos y prompt exacto.

Requiere:
- streamlit, pandas, numpy, gspread, google-auth, openai, unidecode
- chromadb (opcional; hay fallback simple).
"""

import os
import re
import json
import hashlib
import datetime as dt
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

st.set_page_config(page_title="Fénix | Agente de Negocio (Híbrido)", page_icon="🔥", layout="wide")

# ---------- Constantes ----------
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

# Alias de columnas (para mapear sinónimos → nombre canónico)
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
    "cliente": "NOMBRE CLIENTE",
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
Compañía: Fénix Automotriz (Chile). Empresa familiar dedicada a la reparación de vehículos, fundada en 2017.
Ejes: Experiencia excepcional, Excelencia operacional, Transformación tecnológica, Innovación y Expansión nacional.
Misión: “Entregar un servicio de reparación transparente, de calidad y puntual...”.
Visión 2026: “Ser el servicio de reparación automotriz preferido...”.
Proceso: Presupuesto → Recepción → Desarme → Desabolladura → Preparación → Pintura → Pulido → Lavado/entrega.
Roles: Gerencia, Planificación/Control, Adm/Finanzas, Comercial, Marketing, Líder Unidad, Jefe de Taller, Desarmador, Desabollador, Pintor, etc.
""".strip()

SPANISH_MONTHS = {
    "enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,
    "julio":7,"agosto":8,"septiembre":9,"setiembre":9,"octubre":10,"noviembre":11,"diciembre":12
}

# ---------- Utilidades ----------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", unidecode(str(s)).strip().lower())

def _fmt_value(v: Any) -> str:
    if pd.isna(v): return ""
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, (int, float)):
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return str(v)

def _map_col(col: str, available: List[str]) -> Optional[str]:
    if not col: return None
    if col in available: return col
    key = _norm(col)
    for a in available:
        if _norm(a) == key: return a
    if key in COL_ALIASES and COL_ALIASES[key] in available:
        return COL_ALIASES[key]
    for a in available:
        if key in _norm(a): return a
    return None

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

# ---------- 1) get_data_from_gsheet (CRUDO) ----------
@st.cache_data(show_spinner=False)
def get_data_from_gsheet() -> pd.DataFrame:
    """Devuelve el DataFrame crudo, exactamente como viene de Google Sheets."""
    gc = get_gspread_client()
    ws = gc.open_by_key(SHEET_ID).worksheet(WORKSHEET)
    records = ws.get_all_records()
    return pd.DataFrame(records) if records else pd.DataFrame()

# Copia normalizada (no toca el crudo)
def normalize_df(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty: return raw.copy()
    # Renombrar columnas a canónicas
    mapping = {}
    for c in raw.columns:
        key = _norm(c)
        for can in CANONICAL_FIELDS:
            if _norm(can) == key: mapping[c] = can
        if key in COL_ALIASES: mapping[c] = COL_ALIASES[key]
    df = raw.rename(columns=mapping).copy()

    # Fechas
    for col in DATE_FIELDS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True, infer_datetime_format=True)
    # Numéricos
    for col in NUM_FIELDS:
        if col in df.columns:
            df[col] = (df[col].astype(str)
                        .str.replace(r"[.$ ]","",regex=True)
                        .str.replace(",",".",regex=True))
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # FACTURADO normalizado
    if "FACTURADO" in df.columns:
        df["FACTURADO"] = (df["FACTURADO"].astype(str).str.strip().str.upper()
                           .replace({"TRUE":"SI","FALSE":"NO","1":"SI","0":"NO"}))
    return df

# ---------- Fragmentos narrativos ----------
def row_to_fragment(row: pd.Series, present: List[str]) -> str:
    return "\n".join([f"{c}: {_fmt_value(row.get(c,''))}" for c in present])

def make_fragments(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cols = [c for c in CANONICAL_FIELDS if c in df.columns]
    frags, ids = [], []
    for idx, r in df.iterrows():
        frags.append(row_to_fragment(r, cols))
        ids.append(str(idx))
    return frags, ids

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

# ---------- Vector index ----------
def _hash_frags(frags: List[str]) -> str:
    h=hashlib.md5()
    for t in frags: h.update(t.encode("utf-8"))
    return h.hexdigest()

def ensure_index(frags: List[str], ids: List[str]):
    """Indexa SOLO el subconjunto ya filtrado (búsqueda híbrida)."""
    h=_hash_frags(frags)
    key=f"subset_index_{h}"
    if st.session_state.get("subset_index_hash")==h:
        return
    if CHROMA_AVAILABLE:
        client=chromadb.PersistentClient(path="./.chroma")
        name=f"fenix_subset_{h[:10]}"
        try: coll=client.get_collection(name)
        except Exception:
            coll=client.get_or_create_collection(name)
            oai=get_openai_client()
            with st.spinner("Indexando subconjunto (Chroma)…"):
                embs=embed_texts(oai, frags)
                coll.add(ids=ids, documents=frags, metadatas=[{"row_id":i} for i in ids], embeddings=embs)
        st.session_state.subset_backend="chroma"; st.session_state.subset_collection=coll
    else:
        os.makedirs("./.cache", exist_ok=True)
        path=f"./.cache/{key}.npz"
        if not os.path.exists(path):
            oai=get_openai_client()
            with st.spinner("Indexando subconjunto (fallback)…"):
                embs=np.array(embed_texts(oai, frags), dtype=np.float32)
            np.savez(path, embs=embs, ids=np.array(ids,dtype=object), docs=np.array(frags,dtype=object))
        d=np.load(path, allow_pickle=True)
        st.session_state.subset_embs=d["embs"]; st.session_state.subset_ids=d["ids"].tolist(); st.session_state.subset_docs=d["docs"].tolist()
        st.session_state.subset_backend="simple"
    st.session_state.subset_index_hash=h

def retrieve_top_subset(query: str, k=6) -> Tuple[List[str], List[str]]:
    oai=get_openai_client()
    q=np.array(embed_texts(oai,[query])[0], dtype=np.float32)
    if st.session_state.get("subset_backend")=="chroma":
        res=st.session_state.subset_collection.query(query_embeddings=[q.tolist()], n_results=k, include=["documents","metadatas"])
        docs=res.get("documents",[[]])[0]; metas=res.get("metadatas",[[]])[0]; ids=[m.get("row_id","") for m in metas]
        return docs, ids
    A=st.session_state.subset_embs
    qn=q/(np.linalg.norm(q)+1e-9); An=A/(np.linalg.norm(A,axis=1,keepdims=True)+1e-9)
    idx=np.argsort(-(An@qn).ravel())[:k]
    return [st.session_state.subset_docs[i] for i in idx], [st.session_state.subset_ids[i] for i in idx]

# ---------- System & User Prompt ----------
def system_prompt() -> str:
    return f"""
Eres un CONSULTOR DE GESTIÓN y ANALISTA DE DATOS para Fénix Automotriz.
Usa EXCLUSIVAMENTE el “Contexto proporcionado”. Si falta información, responde exactamente: "No tengo la información necesaria en los datos".
Puedes realizar cálculos simples (sumas, promedios, min/max) sobre los registros del contexto. Sé conciso y accionable.
Contexto de negocio:
{BUSINESS_CONTEXT}
""".strip()

def build_user_prompt(question: str, context_docs: List[str]) -> str:
    ctx = "\n\n-----\n\n".join(context_docs) if context_docs else "(sin contexto)"
    return f"Pregunta: {question}\n\nContexto proporcionado (subconjunto filtrado):\n{ctx}"

def llm_answer(question: str, docs: List[str]) -> Tuple[str, List[Dict[str,str]]]:
    client=get_openai_client()
    messages=[{"role":"system","content":system_prompt()},
              {"role":"user","content":build_user_prompt(question, docs)}]
    resp=client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.2)
    return resp.choices[0].message.content, messages

# ---------- 2) Intent Parser (LLM + reglas) ----------
def infer_schema_for_llm(df_norm: pd.DataFrame) -> Dict[str,Any]:
    schema={}
    for c in df_norm.columns:
        if c.startswith("__"): continue
        s=df_norm[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            schema[c]={"type":"date"}
        elif pd.api.types.is_numeric_dtype(s):
            schema[c]={"type":"number"}
        else:
            schema[c]={"type":"text"}
    return schema

def llm_parse_intent(question: str, df_norm: pd.DataFrame) -> Dict[str,Any]:
    """
    Devuelve un dict con:
      filters: [{column, op, value: [..]}]
      date_window_days: int | null
    """
    client=get_openai_client()
    schema = json.dumps(infer_schema_for_llm(df_norm))
    system = (
        "Eres un parser de intenciones. Extrae filtros TABULARES a partir de una pregunta en español. "
        "Usa solo nombres de columnas del esquema. No inventes columnas. Devuelve JSON vía función."
    )
    user = f"Esquema de columnas (nombre → tipo):\n{schema}\n\nPregunta del usuario:\n{question}"
    tools=[{
        "type":"function",
        "function":{
            "name":"emitir_filtros",
            "description":"Devuelve filtros exactos a aplicar sobre un DataFrame.",
            "parameters":{
                "type":"object",
                "properties":{
                    "filters":{
                        "type":"array",
                        "items":{
                            "type":"object",
                            "properties":{
                                "column":{"type":"string"},
                                "op":{"type":"string","enum":["eq","neq","gt","gte","lt","lte","contains","not_contains","in","not_in","empty","not_empty","between_dates"]},
                                "value":{"type":"array","items":{"type":"string"}}
                            },
                            "required":["column","op"]
                        }
                    },
                    "date_window_days":{"type":["integer","null"]},
                    "notes":{"type":["string","null"]}
                },
                "required":["filters"]
            }
        }
    }]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        tools=tools,
        tool_choice={"type":"function","function":{"name":"emitir_filtros"}}
    )
    msg = resp.choices[0].message
    out={"filters":[], "date_window_days": None, "notes": None}
    try:
        if msg.tool_calls:
            args = msg.tool_calls[0].function.arguments
            cand = json.loads(args)
            if isinstance(cand, dict):
                out.update({k:v for k,v in cand.items() if k in out})
    except Exception:
        pass
    return out

def heuristic_rules(question: str, df_norm: pd.DataFrame) -> List[Dict[str,Any]]:
    """Reglas adicionales para casos comunes (no facturados, en proceso, próximos días, etc.)."""
    q=_norm(question)
    today = dt.date.today()

    rules = []

    # "en proceso"
    if "en proceso" in q or "en proceso" in q.replace("ó","o"):
        if "ESTADO SERVICIO" in df_norm.columns:
            rules.append({"column":"ESTADO SERVICIO","op":"contains","value":["PROCES"]})

    # "no facturado" / "sin facturar"
    if "no factur" in q or "sin factur" in q:
        # interpretamos como: FACTURADO in ["NO","PENDIENTE"] OR NUMERO DE FACTURA vacío OR FECHA DE FACTURACION vacía
        if "FACTURADO" in df_norm.columns:
            rules.append({"column":"FACTURADO","op":"in","value":["NO","PENDIENTE","","NAN"]})
        if "NUMERO DE FACTURA" in df_norm.columns:
            rules.append({"column":"NUMERO DE FACTURA","op":"empty","value":[]})
        if "FECHA DE FACTURACION" in df_norm.columns:
            rules.append({"column":"FECHA DE FACTURACION","op":"empty","value":[]})

    # "sin aprobación"
    if ("sin aprob" in q) or ("no aprob" in q):
        if "ESTADO PRESUPUESTO" in df_norm.columns:
            rules.append({"column":"ESTADO PRESUPUESTO","op":"not_in","value":["APROBADO","PERDIDO"]})

    # "próximos X días" → preferimos FECHA ENTREGA o FECHA DE PAGO FACTURA si la pregunta habla de entrega/factura
    m = re.search(r"(proxim|siguient)[oa]s?\s+(\d+)\s+dias", q)
    days = int(m.group(2)) if m else None
    if ("entreg" in q) and "FECHA ENTREGA" in df_norm.columns and days:
        start = dt.date.today().strftime("%Y-%m-%d")
        end   = (dt.date.today()+dt.timedelta(days=days)).strftime("%Y-%m-%d")
        rules.append({"column":"FECHA ENTREGA","op":"between_dates","value":[start,end]})
    if ("pago" in q or "pagar" in q) and "FECHA DE PAGO FACTURA" in df_norm.columns and days:
        start = dt.date.today().strftime("%Y-%m-%d")
        end   = (dt.date.today()+dt.timedelta(days=days)).strftime("%Y-%m-%d")
        rules.append({"column":"FECHA DE PAGO FACTURA","op":"between_dates","value":[start,end]})

    # Mes explícito (marzo, etc.) para facturación
    for name, num in SPANISH_MONTHS.items():
        if name in q and "FECHA DE FACTURACION" in df_norm.columns:
            # mes = num del año actual
            y = dt.date.today().year
            rules.append({"column":"FECHA DE FACTURACION","op":"between_dates",
                          "value":[f"{y}-{num:02d}-01", f"{y}-{num:02d}-31"]})
            break

    return rules

# ---------- 3) Filtrado determinista (sobre DF normalizado) ----------
def apply_filters(df: pd.DataFrame, filters: List[Dict[str,Any]]) -> Tuple[pd.DataFrame, List[Dict[str,Any]]]:
    """Aplica filtros y devuelve (df_filtrado, log_de_aplicacion)."""
    if df.empty or not filters: return df, []
    cols = list(df.columns)
    mask = pd.Series(True, index=df.index)
    log = []

    for f in filters:
        col = _map_col(str(f.get("column","")), cols)
        op  = f.get("op","")
        vals= f.get("value", [])
        if not col: continue

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
            v = float(vals[0]) if isinstance(vals, list) and vals else np.nan
            if op == "gt": m = sn > v
            elif op == "gte": m = sn >= v
            elif op == "lt": m = sn < v
            elif op == "lte": m = sn <= v

        elif op in ["empty","not_empty"]:
            if op == "empty":
                m = s.isna() | (s.astype(str).str.strip()=="") | (s.astype(str).str.upper().isin(["NAN","NONE","NULL","-"]))
            else:
                m = ~(s.isna() | (s.astype(str).str.strip()=="") | (s.astype(str).str.upper().isin(["NAN","NONE","NULL","-"])))

        elif op == "between_dates":
            sd = pd.to_datetime(vals[0], dayfirst=True, errors="coerce") if vals else pd.NaT
            ed = pd.to_datetime(vals[1], dayfirst=True, errors="coerce") if len(vals)>1 else pd.NaT
            s2 = s if pd.api.types.is_datetime64_any_dtype(s) else pd.to_datetime(s, dayfirst=True, errors="coerce")
            if pd.notna(sd) and pd.notna(ed):
                m = s2.between(sd, ed)
            elif pd.notna(sd):
                m = s2 >= sd
            elif pd.notna(ed):
                m = s2 <= ed

        mask &= m
        log.append({"column": col, "op": op, "value": vals, "remaining": int(mask.sum())})

    return df[mask].copy(), log

# ---------- UI ----------
st.title("🔥 Fénix Automotriz — Búsqueda Híbrida (Filtros exactos + Semántica)")
st.caption("1) Intent Parser → 2) Filtro exacto en Pandas → 3) RAG sobre subconjunto filtrado → 4) Respuesta.")

with st.sidebar:
    st.subheader("⚙️ Parámetros")
    top_k = st.slider("Top-K fragmentos para contexto", 3, 15, 6, 1)
    show_diag = st.checkbox("🔎 Modo diagnóstico", value=True)
    force_index = st.button("🔁 Reindexar subconjunto")

# 1) Cargar DF crudo y crear copia normalizada
raw_df = get_data_from_gsheet()
if raw_df.empty:
    st.error("La hoja MODELO_BOT está vacía o no se pudo leer.")
    st.stop()

df = normalize_df(raw_df)

# Historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ej.: 'vehículos en proceso y no facturados', 'facturas a pagar próximos 7 días', 'sin aprobación'…")
if question:

    # 2) Intent Parser (LLM)
    llm_filters = llm_parse_intent(question, df)
    extracted_filters = llm_filters.get("filters", [])
    date_window_days = llm_filters.get("date_window_days")

    # 2b) Reglas heurísticas (complemento)
    rule_filters = heuristic_rules(question, df)

    # Mezclar, dando prioridad a columnas válidas
    all_filters = []
    for f in (extracted_filters + rule_filters):
        # mapear a canónicas existentes
        f = dict(f)  # copy
        f["column"] = _map_col(f.get("column",""), list(df.columns)) or f.get("column","")
        all_filters.append(f)

    # 3) Aplicar filtros exactos sobre DF normalizado
    filtered_df, filter_log = apply_filters(df, all_filters)

    # Si no hay resultados, relajamos a solo LLM filters o fallback sin filtros
    if filtered_df.empty and extracted_filters:
        filtered_df, filter_log = apply_filters(df, extracted_filters)
    if filtered_df.empty:
        filtered_df = df.copy()

    # 4) RAG SOLO sobre el subconjunto filtrado
    if force_index:
        for k in ["subset_index_hash","subset_collection","subset_embs","subset_ids","subset_docs","subset_backend"]: st.session_state.pop(k, None)

    frags, ids = make_fragments(filtered_df)
    ensure_index(frags, ids)
    docs, row_ids = retrieve_top_subset(question, k=top_k)

    # 5) LLM: respuesta final
    answer, prompt_msgs = llm_answer(question, docs)

    # Guardar conversación
    st.session_state.messages += [{"role":"user","content":question},
                                  {"role":"assistant","content":answer}]

    with st.chat_message("assistant"):
        if show_diag:
            with st.expander("🧰 Diagnóstico de filtros"):
                st.markdown("**Filtros extraídos por LLM:**")
                st.json(extracted_filters)
                st.markdown("**Filtros añadidos por reglas:**")
                st.json(rule_filters)
                st.markdown("**Filtros aplicados (secuencia y remanentes):**")
                st.json(filter_log)

            with st.expander("🧩 Top-3 fragmentos del subconjunto filtrado"):
                for i, d in enumerate(docs[:3], 1):
                    st.markdown(f"**Fragmento {i}**\n\n```\n{d}\n```")

            with st.expander("🧪 Prompt exacto enviado al LLM"):
                st.write("**System Prompt:**")
                st.code(prompt_msgs[0]["content"])
                st.write("**User Prompt:**")
                st.code(prompt_msgs[1]["content"])

        st.markdown(answer)

        # Mostrar filas filtradas (útiles para validar)
        show_cols = [c for c in CANONICAL_FIELDS if c in filtered_df.columns]
        if not filtered_df.empty:
            st.markdown("**Filas del subconjunto filtrado (para auditoría):**")
            st.dataframe(filtered_df[show_cols] if show_cols else filtered_df, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Descargar subconjunto (CSV)",
                               data=filtered_df.to_csv(index=False).encode("utf-8"),
                               file_name="subconjunto_filtrado.csv", mime="text/csv")
        else:
            st.info("No se encontraron filas con los filtros actuales.")

