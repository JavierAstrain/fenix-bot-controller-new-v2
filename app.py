# -*- coding: utf-8 -*-
"""
Fénix Automotriz — Agente de Negocio (RAG Híbrido)
MEJORA: Analizador de intenciones robusto para MES en español (enero..diciembre)
- Si la pregunta especifica un MES (y opcionalmente AÑO), se aplica PRIMERO un filtro estricto por ese mes.
- El índice vectorial SIEMPRE se construye sobre ese subconjunto mensual (sin mezclar otros meses).
- Compatibilidad con guardarraíl de futuro (>= hoy) cuando NO hay mes explícito.
- Mantiene: filtros robustos, OT “Sin asignar”, totales y gráfico por TIPO CLIENTE.

Requiere: streamlit, pandas, numpy, gspread, google-auth, openai, (chromadb opcional)
"""

import os, re, json, hashlib, datetime as dt
from calendar import monthrange
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from unidecode import unidecode
from openai import OpenAI

# Vector DB opcional
CHROMA_AVAILABLE = True
try:
    import chromadb
except Exception:
    CHROMA_AVAILABLE = False

st.set_page_config(page_title="Fénix | Agente (Meses + Fechas estrictas)", page_icon="🔥", layout="wide")

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
TOTAL_CANDIDATES = ["MONTO PRINCIPAL NETO","IVA PRINCIPAL [F]","MONTO PRINCIPAL BRUTO [F]","CANTIDAD DE VEHICULO"]

COL_ALIASES = {
    "dias en planta": "NUMERO DE DIAS EN PLANTA", "días en planta": "NUMERO DE DIAS EN PLANTA",
    "dias de pago": "DIAS DE PAGO DE FACTURA",
    "fecha facturacion": "FECHA DE FACTURACION", "fecha facturación": "FECHA DE FACTURACION",
    "monto bruto": "MONTO PRINCIPAL BRUTO [F]", "monto neto": "MONTO PRINCIPAL NETO", "iva": "IVA PRINCIPAL [F]",
    "fecha recepcion": "FECHA RECEPCION", "fecha recepción": "FECHA RECEPCION",
    "fecha ingreso": "FECHA INGRESO PLANTA", "fecha salida": "FECHA SALIDA PLANTA",
    "entrega": "FECHA ENTREGA", "numero factura": "NUMERO DE FACTURA", "nro factura": "NUMERO DE FACTURA",
    "n° factura": "NUMERO DE FACTURA", "cliente": "NOMBRE CLIENTE",
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
Compañía: Fénix Automotriz (Chile). Empresa familiar fundada en 2017.
Ejes: Experiencia excepcional, Excelencia operacional, Transformación tecnológica, Innovación, Expansión nacional.
Misión: “Entregar un servicio de reparación transparente, de calidad y puntual...”.
Proceso: Presupuesto → Recepción → Desarme → Desabolladura → Preparación → Pintura → Pulido → Lavado/entrega.
Roles: Gerencia, Planificación y Control, Adm/Finanzas, Comercial, Marketing, Jefe de Taller, Desarmador, Desabollador, Pintor, etc.
""".strip()

SPANISH_MONTHS = {
    "enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,
    "julio":7,"agosto":8,"septiembre":9,"setiembre":9,"octubre":10,"noviembre":11,"diciembre":12
}

# ---------- Utilidades ----------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", unidecode(str(s)).strip().lower())

def _fmt_value(v) -> str:
    if pd.isna(v): return ""
    if isinstance(v, pd.Timestamp): return v.strftime("%Y-%m-%d")
    if isinstance(v, (int,float)): return f"{v:.2f}".rstrip("0").rstrip(".")
    return str(v)

def _map_col(col: str, available: List[str]) -> Optional[str]:
    if not col: return None
    if col in available: return col
    key=_norm(col)
    for a in available:
        if _norm(a)==key: return a
    if key in COL_ALIASES and COL_ALIASES[key] in available: return COL_ALIASES[key]
    for a in available:
        if key in _norm(a): return a
    return None

def _to_number(x) -> Optional[float]:
    if x is None: return None
    try:
        if isinstance(x,(int,float)): return float(x)
        s=str(x).strip().replace(" ","").replace("$","").replace(".","").replace(",",".")
        return float(s) if re.match(r"^-?\d+(\.\d+)?$", s) else None
    except Exception:
        return None

def _to_datetime(x) -> Optional[pd.Timestamp]:
    try:
        return pd.to_datetime(x, dayfirst=True, errors="coerce")
    except Exception:
        return None

# ---------- Credenciales ----------
def _safe_secret_keys():
    try: return list(st.secrets._secrets.keys())
    except Exception:
        try: return list(st.secrets.keys())
        except Exception: return []

def _load_sa() -> Dict[str,Any]:
    for k in ["gcp_service_account","service_account","google_service_account"]:
        try:
            obj=st.secrets[k]
            if isinstance(obj,dict) and "client_email" in obj and "private_key" in obj:
                return dict(obj)
        except Exception: pass
    for k in ["GOOGLE_CREDENTIALS","GOOGLE_CREDENTIALS_JSON","SERVICE_ACCOUNT_JSON","gcp_service_account_json"]:
        try:
            s=st.secrets.get(k,"")
            if isinstance(s,str) and s.strip().startswith("{"): return json.loads(s)
        except Exception: pass
    for k in ["GOOGLE_CREDENTIALS","GOOGLE_CREDENTIALS_JSON"]:
        try:
            s=os.getenv(k,"")
            if isinstance(s,str) and s.strip().startswith("{"): return json.loads(s)
        except Exception: pass
    return {}

def get_gspread_client():
    info=_load_sa()
    if not info:
        st.error("Faltan credenciales de Google Service Account. Claves presentes: **{}**".format(", ".join(_safe_secret_keys())))
        st.stop()
    scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds=Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)

# ---------- Datos ----------
@st.cache_data(show_spinner=False)
def get_data_from_gsheet() -> pd.DataFrame:
    gc=get_gspread_client()
    ws=gc.open_by_key(SHEET_ID).worksheet(WORKSHEET)
    rec=ws.get_all_records()
    return pd.DataFrame(rec) if rec else pd.DataFrame()

def normalize_df(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty: return raw.copy()
    mapping={}
    for c in raw.columns:
        key=_norm(c)
        for can in CANONICAL_FIELDS:
            if _norm(can)==key: mapping[c]=can
        if key in COL_ALIASES: mapping[c]=COL_ALIASES[key]
    df=raw.rename(columns=mapping).copy()

    # Fechas → datetime (NaT seguro)
    for col in DATE_FIELDS:
        if col in df.columns:
            df[col]=pd.to_datetime(df[col], errors="coerce", dayfirst=True, infer_datetime_format=True)

    # Numéricos
    for col in NUM_FIELDS:
        if col in df.columns:
            df[col]=(df[col].astype(str)
                      .str.replace(r"[ $]","",regex=True)
                      .str.replace(".","",regex=False)
                      .str.replace(",",".",regex=False))
            df[col]=pd.to_numeric(df[col], errors="coerce")

    if "FACTURADO" in df.columns:
        df["FACTURADO"]=(df["FACTURADO"].astype(str).str.strip().str.upper()
                         .replace({"TRUE":"SI","FALSE":"NO","1":"SI","0":"NO"}))
    return df

# ---------- Fragmentos (OT vacío → “Sin asignar”) ----------
def _display_value_for_fragment(col: str, val) -> str:
    empty=(val is None) or (isinstance(val,float) and pd.isna(val)) or (str(val).strip()=="")
    if col=="OT" and empty: return "Sin asignar"
    if empty: return "N/A"
    return _fmt_value(val)

def row_to_fragment(row: pd.Series, cols: List[str]) -> str:
    return "\n".join([f"{c}: {_display_value_for_fragment(c, row.get(c,''))}" for c in cols])

def make_fragments(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cols=[c for c in CANONICAL_FIELDS if c in df.columns]
    frags, ids=[], []
    for idx, r in df.iterrows():
        frags.append(row_to_fragment(r, cols))
        ids.append(str(idx))
    return frags, ids

# ---------- OpenAI ----------
def get_openai_client() -> OpenAI:
    api_key=os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY",""))
    if not api_key:
        st.error("OPENAI_API_KEY no está configurada."); st.stop()
    return OpenAI(api_key=api_key)

def embed_texts(client: OpenAI, texts: List[str], model="text-embedding-3-small") -> List[List[float]]:
    out=[]; B=96
    for i in range(0,len(texts),B):
        resp=client.embeddings.create(model=model, input=texts[i:i+B])
        out.extend([d.embedding for d in resp.data])
    return out

# ---------- Índice vectorial (SIEMPRE sobre el subconjunto filtrado) ----------
def _hash_frags(frags: List[str]) -> str:
    h=hashlib.md5()
    for t in frags: h.update(t.encode("utf-8"))
    return h.hexdigest()

def ensure_index(frags: List[str], ids: List[str]):
    """No indexa si no hay frags (evita mezclar)."""
    if len(frags)==0:
        st.session_state.subset_backend=None
        st.session_state.subset_index_hash=None
        return
    h=_hash_frags(frags)
    if st.session_state.get("subset_index_hash")==h: return
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
        path=f"./.cache/simple_subset_{h}.npz"
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
    if not st.session_state.get("subset_backend"):
        return [], []
    oai=get_openai_client()
    q=np.array(embed_texts(oai,[query])[0], dtype=np.float32)
    if st.session_state.subset_backend=="chroma":
        r=st.session_state.subset_collection.query(query_embeddings=[q.tolist()], n_results=k, include=["documents","metadatas"])
        docs=r.get("documents",[[]])[0]; metas=r.get("metadatas",[[]])[0]; ids=[m.get("row_id","") for m in metas]
        return docs, ids
    A=st.session_state.subset_embs
    if len(A)==0: return [], []
    qn=q/(np.linalg.norm(q)+1e-9); An=A/(np.linalg.norm(A,axis=1,keepdims=True)+1e-9)
    idx=np.argsort(-(An@qn).ravel())[:k]
    return [st.session_state.subset_docs[i] for i in idx], [st.session_state.subset_ids[i] for i in idx]

# ---------- Prompts ----------
def system_prompt() -> str:
    return f"""
Eres un CONSULTOR DE GESTIÓN y ANALISTA DE DATOS para Fénix Automotriz.
Usa EXCLUSIVAMENTE el “Contexto proporcionado” (ya filtrado: MES cuando aplique y/o FECHAS ≥ HOY).
Si falta información, responde: "No tengo la información necesaria en los datos".
Cuando la pregunta implique montos, calcula y muestra la SUMA TOTAL.
Contexto:
{BUSINESS_CONTEXT}
""".strip()

def build_user_prompt(question: str, context_docs: List[str]) -> str:
    ctx="\n\n-----\n\n".join(context_docs) if context_docs else "(sin contexto)"
    return f"Pregunta: {question}\n\nContexto proporcionado (subconjunto filtrado estrictamente):\n{ctx}"

def llm_answer(question: str, docs: List[str]):
    client=get_openai_client()
    messages=[{"role":"system","content":system_prompt()},
              {"role":"user","content":build_user_prompt(question, docs)}]
    resp=client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.2)
    return resp.choices[0].message.content, messages

# ---------- Parser / reglas genéricas ----------
def infer_schema_for_llm(df: pd.DataFrame) -> Dict[str,Any]:
    schema={}
    for c in df.columns:
        s=df[c]
        if pd.api.types.is_datetime64_any_dtype(s): schema[c]={"type":"date"}
        elif pd.api.types.is_numeric_dtype(s):     schema[c]={"type":"number"}
        else:                                      schema[c]={"type":"text"}
    return schema

def llm_parse_intent(question: str, df: pd.DataFrame) -> Dict[str,Any]:
    client=get_openai_client()
    schema=json.dumps(infer_schema_for_llm(df))
    system=("Eres un parser de intenciones. Extrae filtros TABULARES desde una pregunta en español. "
            "Usa solo nombres de columnas del esquema. Devuelve JSON vía función.")
    user=f"Esquema de columnas (nombre → tipo):\n{schema}\n\nPregunta del usuario:\n{question}"
    tools=[{
        "type":"function",
        "function":{
            "name":"emitir_filtros",
            "description":"Devuelve filtros exactos para un DataFrame.",
            "parameters":{
                "type":"object",
                "properties":{
                    "filters":{"type":"array","items":{"type":"object","properties":{
                        "column":{"type":"string"},
                        "op":{"type":"string","enum":["eq","neq","gt","gte","lt","lte","contains","not_contains","in","not_in","empty","not_empty","between_dates"]},
                        "value":{"type":"array","items":{"type":"string"}}
                    },"required":["column","op"]}},
                    "date_window_days":{"type":["integer","null"]},
                    "notes":{"type":["string","null"]}
                },
                "required":["filters"]
            }
        }
    }]
    resp=client.chat.completions.create(
        model="gpt-4o-mini", temperature=0.1,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        tools=tools, tool_choice={"type":"function","function":{"name":"emitir_filtros"}}
    )
    msg=resp.choices[0].message
    out={"filters":[], "date_window_days": None, "notes": None}
    try:
        if msg.tool_calls:
            cand=json.loads(msg.tool_calls[0].function.arguments)
            if isinstance(cand,dict): out.update({k:v for k,v in cand.items() if k in out})
    except Exception:
        pass
    return out

def choose_date_column(question: str, df: pd.DataFrame) -> Optional[str]:
    q=_norm(question); pref=[]
    if "pago" in q or "pagar" in q: pref+=["FECHA DE PAGO FACTURA"]
    if "factur" in q or "factura" in q: pref+=["FECHA DE FACTURACION"]
    if "entreg" in q: pref+=["FECHA ENTREGA"]
    if "ingres" in q: pref+=["FECHA INGRESO PLANTA"]
    if "recepc" in q: pref+=["FECHA RECEPCION"]
    pref += ["FECHA DE PAGO FACTURA","FECHA DE FACTURACION","FECHA ENTREGA","FECHA INGRESO PLANTA","FECHA RECEPCION"]
    for c in pref:
        if c in df.columns: return c
    return None

def detect_future_intent(question: str) -> bool:
    q=_norm(question)
    keys=["proxim","próxim","pronto","futuro","en adelante","desde hoy","a partir de hoy","hoy en adelante","venider"]
    return any(k in q for k in keys)

# ---------- NUEVO: MES — Analizador robusto y filtro PRIMARIO ----------
def parse_explicit_month_year(question: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """
    Detecta un nombre de mes en español (con acentos normalizados) y un AÑO opcional.
    Retorna (month_num, year, matched_month_name) o (None, None, None).
    Coincidencias por palabra completa para evitar falsos positivos (p.ej. 'marca' ≠ 'marzo').
    """
    q=_norm(question)
    for name, num in SPANISH_MONTHS.items():
        # palabra completa
        if re.search(rf"\b{name}\b", q):
            m=re.search(r"(19|20)\d{2}", q)
            year=int(m.group(0)) if m else None
            return num, year, name
    return None, None, None

def month_to_range(year: int, month: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=year, month=month, day=1)
    end = pd.Timestamp(year=year, month=month, day=monthrange(year, month)[1])
    return start, end

def apply_month_filter_first(df: pd.DataFrame, month: int, year: int, date_col: str) -> pd.DataFrame:
    """
    Filtro PRIMARIO: Se aplica ANTES de cualquier otro filtro.
    Acepta solo filas cuyo date_col pertenece al mes/año indicados.
    No hace fallback. NaT quedan fuera (estricto).
    """
    if date_col not in df.columns: return pd.DataFrame()
    s = df[date_col]
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = pd.to_datetime(s, errors="coerce", dayfirst=True)
    mask = (s.dt.year == year) & (s.dt.month == month)
    return df[mask].copy()

def enforce_future_guardrail(filters: List[Dict[str,Any]], df: pd.DataFrame, question: str) -> Tuple[List[Dict[str,Any]], Optional[str]]:
    """Si intención futura, impone >= HOY en columna de fecha pertinente."""
    if not detect_future_intent(question): return filters, None
    date_col=choose_date_column(question, df)
    if not date_col: return filters, None
    today=dt.date.today().strftime("%Y-%m-%d")
    found=False; out=[]
    for f in filters:
        g=dict(f); col=_map_col(g.get("column",""), list(df.columns)) or g.get("column","")
        if col!=date_col:
            out.append(g); continue
        found=True
        op=g.get("op",""); vals=g.get("value",[])
        if op=="between_dates":
            start=vals[0] if len(vals)>=1 and vals[0] else None
            end=vals[1] if len(vals)>=2 and vals[1] else None
            s = today if not start else max(today, start)
            g["value"]=[s] if not end else [s, end]
            out.append(g)
        elif op in ["gt","gte"]:
            v0=vals[0] if vals else None
            g["op"]="gte"; g["value"]=[max(today, v0) if v0 else today]
            out.append(g)
        else:
            out.append({"column":date_col,"op":"gte","value":[today]})
    if not found:
        out.append({"column":date_col,"op":"gte","value":[today]})
    return out, date_col

# ---------- Filtro robusto (genérico, secundario) ----------
def _ensure_list(x):
    if isinstance(x,list): return x
    if x is None: return []
    return [x]

def apply_filters(df: pd.DataFrame, filters: List[Dict[str,Any]]) -> Tuple[pd.DataFrame, List[Dict[str,Any]]]:
    if df.empty or not filters: return df.copy(), []
    cols=list(df.columns); mask=pd.Series(True,index=df.index); log=[]
    for f in filters:
        col=_map_col(str(f.get("column","")), cols)
        op=str(f.get("op","")).strip()
        vals=_ensure_list(f.get("value", []))
        if not col or col not in df.columns or not op: continue
        s=df[col]; m=pd.Series(True,index=df.index)
        try:
            if op in ["eq","neq","contains","not_contains","in","not_in"]:
                sv=s.astype(str).str.upper().str.strip()
                vlist=[str(v).upper().strip() for v in vals]
                if op=="eq": m=sv.isin(vlist)
                elif op=="neq": m=~sv.isin(vlist)
                elif op=="contains":
                    pat="|".join([re.escape(v) for v in vlist]) if vlist else ""
                    m=sv.str.contains(pat, na=False)
                elif op=="not_contains":
                    pat="|".join([re.escape(v) for v in vlist]) if vlist else ""
                    m=~sv.str.contains(pat, na=False)
                elif op=="in": m=sv.isin(vlist)
                elif op=="not_in": m=~sv.isin(vlist)
            elif op in ["gt","gte","lt","lte"]:
                if pd.api.types.is_numeric_dtype(s):
                    sn=pd.to_numeric(s, errors="coerce"); val=_to_number(vals[0]) if vals else None
                    if val is None: m=pd.Series(False,index=df.index)
                    else:
                        if op=="gt": m=sn>val
                        if op=="gte": m=sn>=val
                        if op=="lt": m=sn<val
                        if op=="lte": m=sn<=val
                elif pd.api.types.is_datetime64_any_dtype(s):
                    sd=_to_datetime(vals[0]) if vals else None
                    if sd is None: m=pd.Series(False,index=df.index)
                    else:
                        if op=="gt": m=s>sd
                        if op=="gte": m=s>=sd
                        if op=="lt": m=s<sd
                        if op=="lte": m=s<=sd
                else:
                    sv=s.astype(str); v=str(vals[0]) if vals else ""
                    if op=="gt": m=sv>v
                    if op=="gte": m=sv>=v
                    if op=="lt": m=sv<v
                    if op=="lte": m=sv<=v
            elif op in ["empty","not_empty"]:
                empty_mask=s.isna() | (s.astype(str).str.strip()=="") | (s.astype(str).str.upper().isin(["NAN","NONE","NULL","-"]))
                m = empty_mask if op=="empty" else ~empty_mask
            elif op=="between_dates":
                sd=_to_datetime(vals[0]) if len(vals)>=1 and vals[0] else None
                ed=_to_datetime(vals[1]) if len(vals)>=2 and vals[1] else None
                s2=s if pd.api.types.is_datetime64_any_dtype(s) else pd.to_datetime(s, dayfirst=True, errors="coerce")
                if sd is not None and ed is not None: m=s2.between(sd, ed)
                elif sd is not None: m=s2>=sd
                elif ed is not None: m=s2<=ed
                else: m=pd.Series(True,index=df.index)
            else:
                m=pd.Series(True,index=df.index)
        except Exception:
            m=pd.Series(True,index=df.index)
        mask &= m
        log.append({"column":col,"op":op,"value":vals,"remaining":int(mask.sum())})
    return df[mask].copy(), log

# ---------- Totales ----------
def requires_totals(question: str) -> bool:
    q=_norm(question)
    return any(k in q for k in ["total","suma","sumar","monto","factur","neto","bruto","iva","ingreso"])

def totals_for_df(df: pd.DataFrame) -> Dict[str,float]:
    totals={}
    for col in TOTAL_CANDIDATES:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            totals[col]=float(pd.to_numeric(df[col], errors="coerce").sum(skipna=True))
    return totals

def append_totals_row(display_df: pd.DataFrame) -> pd.DataFrame:
    if display_df.empty: return display_df
    total_row={c:"" for c in display_df.columns}
    label_col=None
    for c in display_df.columns:
        if not pd.api.types.is_numeric_dtype(display_df[c]): label_col=c; break
    if label_col is None: label_col=display_df.columns[0]
    total_row[label_col]="TOTAL"
    for col in TOTAL_CANDIDATES:
        if col in display_df.columns:
            total_row[col]=pd.to_numeric(display_df[col], errors="coerce").sum(skipna=True)
    return pd.concat([display_df, pd.DataFrame([total_row], columns=display_df.columns)], ignore_index=True)

# ---------- Gráfico: facturación por mes ----------
def show_monthly_facturacion_chart(df: pd.DataFrame, month: int, year: int):
    if df.empty or "TIPO CLIENTE" not in df.columns or "MONTO PRINCIPAL NETO" not in df.columns:
        return
    g = (df.groupby("TIPO CLIENTE", dropna=False)["MONTO PRINCIPAL NETO"]
           .sum(min_count=1)
           .sort_values(ascending=False)
           .reset_index())
    g["MONTO PRINCIPAL NETO"] = g["MONTO PRINCIPAL NETO"].fillna(0.0)
    st.markdown(f"**Facturación (neto) por TIPO CLIENTE — {year}-{month:02d}**")
    st.bar_chart(g.set_index("TIPO CLIENTE"))

# ---------- UI ----------
st.title("🔥 Fénix Automotriz — RAG con MES en español (estricto) + Fechas futuras (≥ hoy)")
st.caption("Si la pregunta especifica mes, se filtra PRIMERO por ese mes y se indexa solo ese subconjunto. Sin mezclas.")

with st.sidebar:
    st.subheader("⚙️ Parámetros")
    top_k=st.slider("Top-K fragmentos para contexto", 3, 15, 6, 1)
    show_diag=st.checkbox("🔎 Modo diagnóstico", value=True)
    force_index=st.button("🔁 Reindexar subconjunto")

raw_df=get_data_from_gsheet()
if raw_df.empty:
    st.error("La hoja MODELO_BOT está vacía o no se pudo leer."); st.stop()
df=normalize_df(raw_df)

if "messages" not in st.session_state: st.session_state.messages=[]
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

question=st.chat_input("Ej.: 'facturación de marzo 2025', 'entregas de septiembre', 'próximas facturas', 'este mes'…")
if question:
    # ============= 1) Analizador de MES — PRIMARIO =============
    month_num, year_num, month_token = parse_explicit_month_year(question)
    month_mode = month_num is not None
    used_month_col = None
    subset_df = pd.DataFrame()

    if month_mode:
        # Seleccionar columna de fecha en función de la intención; por defecto facturación
        if "entreg" in _norm(question) and "FECHA ENTREGA" in df.columns:
            used_month_col="FECHA ENTREGA"
        elif "pago" in _norm(question) and "FECHA DE PAGO FACTURA" in df.columns:
            used_month_col="FECHA DE PAGO FACTURA"
        else:
            used_month_col="FECHA DE FACTURACION" if "FECHA DE FACTURACION" in df.columns else choose_date_column(question, df)

        # Año por defecto: actual
        year_num = year_num if year_num else dt.date.today().year

        # Filtro PRIMARIO por mes (sin fallback, NaT fuera)
        subset_df = apply_month_filter_first(df, month_num, year_num, used_month_col)

    # ============= 2) Analizador de intenciones (secundario) =============
    #    Se aplica sobre subset_df si month_mode, sino sobre df
    base_for_filters = subset_df if month_mode else df
    llm_filters = llm_parse_intent(question, base_for_filters)
    extracted = llm_filters.get("filters", [])

    # Reglas de FUTURO (solo si NO hay mes explícito)
    used_date_col = None
    if not month_mode and detect_future_intent(question):
        extracted, used_date_col = enforce_future_guardrail(extracted, base_for_filters, question)

    # ============= 3) Aplicar filtros exactos (secundarios) =============
    filtered_df, filter_log = apply_filters(base_for_filters, extracted)

    # Post-filter defensivo por mes (si month_mode): garantiza que nada escape del mes
    if month_mode and not subset_df.empty and used_month_col in filtered_df.columns:
        s = filtered_df[used_month_col]
        if not pd.api.types.is_datetime64_any_dtype(s):
            s = pd.to_datetime(s, errors="coerce", dayfirst=True)
        filtered_df = filtered_df[(s.dt.year == year_num) & (s.dt.month == month_num)]

    # Post-filter defensivo de futuro (si aplica)
    if not month_mode and detect_future_intent(question) and used_date_col and used_date_col in filtered_df.columns:
        today_ts = pd.to_datetime(dt.date.today())
        s = filtered_df[used_date_col]
        filtered_df = filtered_df[s.isna() | (s >= today_ts)]

    # ============= 4) Subconjunto final para indexar =============
    # - Si month_mode: NO fallback al DF completo (evita mezclar otros meses).
    # - Si intención futura: NO fallback al DF completo.
    # - En otros casos, si queda vacío, fallback al DF completo.
    if month_mode:
        final_subset = filtered_df  # estricto al mes
    else:
        if detect_future_intent(question):
            final_subset = filtered_df  # sin fallback en futuro
        else:
            final_subset = filtered_df if not filtered_df.empty else df

    # ============= 5) Indexar SOLO el subconjunto final =============
    frags, ids = make_fragments(final_subset)
    if force_index:
        for k in ["subset_index_hash","subset_collection","subset_embs","subset_ids","subset_docs","subset_backend"]:
            st.session_state.pop(k, None)
    ensure_index(frags, ids)

    # ============= 6) Recuperación SOLO del subconjunto y respuesta LLM =============
    docs, row_ids = retrieve_top_subset(question, k=top_k)
    answer, prompt_msgs = llm_answer(question, docs)

    # ============= 7) Totales y salida UI =============
    want_totals = requires_totals(question)
    totals_dict = totals_for_df(final_subset) if want_totals and not final_subset.empty else {}

    st.session_state.messages += [{"role":"user","content":question},
                                  {"role":"assistant","content":answer}]

    with st.chat_message("assistant"):
        if show_diag:
            with st.expander("🧭 Diagnóstico (MES / FECHAS)"):
                st.markdown("**Mes detectado:**"); st.json({"month_num": month_num, "year": year_num, "token": month_token, "date_col": used_month_col})
                st.markdown("**Filtros del LLM (secundarios):**"); st.json(extracted)
                if not month_mode:
                    st.markdown("**Guardarraíl futuro (>= hoy):**"); st.json({"used_date_col": used_date_col})
                st.markdown("**Aplicación secuencial y remanentes:**"); st.json(filter_log)
                st.markdown(f"**Tamaño del subconjunto indexado:** {len(final_subset)} filas")
            with st.expander("🧩 Top-3 fragmentos del subconjunto"):
                for i, d in enumerate(docs[:3], 1):
                    st.markdown(f"**Fragmento {i}**\n\n```\n{d}\n```")
            with st.expander("🧪 Prompt exacto enviado al LLM"):
                st.write("**System Prompt:**"); st.code(prompt_msgs[0]["content"])
                st.write("**User Prompt:**"); st.code(prompt_msgs[1]["content"])

        if month_mode and final_subset.empty:
            st.info("No se encontraron filas para el mes solicitado.")
        elif (not month_mode) and detect_future_intent(question) and final_subset.empty:
            st.info("No se encontraron filas con fechas futuras (≥ hoy) según tu consulta.")

        st.markdown(answer)

        # Tabla + totales
        if not final_subset.empty:
            show_cols=[c for c in CANONICAL_FIELDS if c in final_subset.columns]
            to_show = final_subset[show_cols] if show_cols else final_subset.copy()
            if want_totals:
                if totals_dict:
                    cols = st.columns(max(1,len(totals_dict)))
                    for i,(k,v) in enumerate(totals_dict.items()):
                        with cols[i]: st.metric(label=f"Suma {k}", value=f"{v:,.0f}".replace(",", "."))
                to_show = append_totals_row(to_show)
            st.markdown("**Subconjunto filtrado (auditoría):**")
            st.dataframe(to_show, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Descargar subconjunto (CSV)",
                               data=to_show.to_csv(index=False).encode("utf-8"),
                               file_name="subconjunto_filtrado.csv", mime="text/csv")

            # Gráfico: facturación por mes
            if month_mode and ("factur" in _norm(question)) and ("FECHA DE FACTURACION" in final_subset.columns):
                year_for_chart = year_num if year_num else dt.date.today().year
                # Asegurar que el gráfico use SÓLO el mes/año indicados
                s = final_subset["FECHA DE FACTURACION"]
                if not pd.api.types.is_datetime64_any_dtype(s):
                    s = pd.to_datetime(s, errors="coerce", dayfirst=True)
                subset_for_chart = final_subset[(s.dt.year == year_for_chart) & (s.dt.month == month_num)].copy()
                show_monthly_facturacion_chart(subset_for_chart, month_num, year_for_chart)
