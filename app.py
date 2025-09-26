# -*- coding: utf-8 -*-
"""
Consulta Nexa IA — Fénix Automotriz
Implementa Analizador de Consulta de Dos Pasos:
  Paso 1: get_filters_from_query (LLM traductor → filtros exactos JSON)
  Paso 2: RAG sobre el subconjunto filtrado + respuesta final del LLM
Incluye: normalización de datos, fechas estrictas, meses en español,
agregaciones deterministas cuando aplica, UI con login y marca.
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

# =========================
#  ASSETS (raíz / assets / /mnt/data)
# =========================
ASSETS_DIR = "assets"

def asset(name: str) -> Optional[str]:
    for p in [name, os.path.join(ASSETS_DIR, name), os.path.join("/mnt/data", name)]:
        if p and os.path.exists(p):
            return p
    return None

def safe_page_icon(name: str, default="🔥"):
    return asset(name) or default

def safe_image(name_or_path: str, **kwargs) -> bool:
    p = asset(name_or_path) or name_or_path
    if p and os.path.exists(p):
        kwargs.pop("use_column_width", None)
        kwargs.setdefault("use_container_width", True)
        st.image(p, **kwargs)
        return True
    return False

# =========================
#   CONFIG / THEME
# =========================
st.set_page_config(
    page_title="Consulta Nexa IA",
    page_icon=safe_page_icon("Isotipo_Nexa.png"),  # <- favicon de Nexa (en raíz o assets/)
    layout="wide",
)

PALETTE = {
    "Nexa Blue": {"primary": "#1e88ff"},
    "Fénix Orange": {"primary": "#ff6a00"},
    "Teal": {"primary": "#14b8a6"},
    "Violet": {"primary": "#8b5cf6"},
    "Slate": {"primary": "#475569"},
}
DEFAULT_THEME = "Nexa Blue"

def apply_theme(name: str):
    primary = PALETTE.get(name, PALETTE[DEFAULT_THEME])["primary"]
    st.markdown(
        f"""
        <style>
        :root {{ --nexa-primary: {primary}; }}
        .stButton>button, .stDownloadButton>button {{
            background-color: var(--nexa-primary) !important; color: white !important;
            border-radius: .6rem; border: 0;
        }}
        .stTextInput input, .stSelectbox select, .stNumberInput input {{
            border: 1px solid {primary}33 !important; border-radius: .5rem !important;
        }}
        .nexa-topbar {{ display:flex; align-items:center; justify-content:flex-end; gap:10px; }}
        .nexa-footer {{ text-align:center; opacity:.7; padding: 24px 0; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

if "theme_name" not in st.session_state:
    st.session_state.theme_name = DEFAULT_THEME
apply_theme(st.session_state.theme_name)

# =========================
#  LOGIN
# =========================
if "authed" not in st.session_state:
    st.session_state.authed = False
if "stats" not in st.session_state:
    st.session_state.stats = {"queries": 0, "tokens_est": 0}

def check_login(user: str, pwd: str) -> bool:
    try:
        return (user == st.secrets.get("USER", "")) and (pwd == st.secrets.get("PASSWORD", ""))
    except Exception:
        return False

def login_view():
    col = st.columns([1,1,1])[1]
    with col:
        safe_image("Nexa_logo.png", width=180)
        st.markdown("### Acceso")
        u = st.text_input("Usuario")
        p = st.text_input("Contraseña", type="password")
        if st.button("Ingresar"):
            if check_login(u, p):
                st.session_state.authed = True
                st.rerun()
            else:
                st.error("Usuario o contraseña inválidos.")

# =========================
#  DATA: Google Sheets
# =========================
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
DATE_FIELDS = [
    "FECHA INGRESO PLANTA","FECHA SALIDA PLANTA","FECHA INSPECCIÓN",
    "FECHA RECEPCION","FECHA ENTREGA","FECHA DE FACTURACION","FECHA DE PAGO FACTURA",
]
NUM_FIELDS = [
    "MONTO PRINCIPAL NETO","IVA PRINCIPAL [F]","MONTO PRINCIPAL BRUTO [F]",
    "NUMERO DE DIAS EN PLANTA","DIAS EN DOMINIO","CANTIDAD DE VEHICULO","DIAS DE PAGO DE FACTURA",
]
TOTAL_CANDIDATES = ["MONTO PRINCIPAL NETO","IVA PRINCIPAL [F]","MONTO PRINCIPAL BRUTO [F]","CANTIDAD DE VEHICULO"]

COL_ALIASES = {
    "fecha de entrega": "FECHA ENTREGA",
    "fecha inspeccion": "FECHA INSPECCIÓN",
    "fecha de recepcion": "FECHA RECEPCION",
    "tipo de vehiculo": "TIPO VEHÍCULO",
    "fecha facturacion": "FECHA DE FACTURACION",
    "fecha facturación": "FECHA DE FACTURACION",
    "monto neto": "MONTO PRINCIPAL NETO",
    "monto bruto": "MONTO PRINCIPAL BRUTO [F]",
    "iva": "IVA PRINCIPAL [F]",
    "numero factura": "NUMERO DE FACTURA", "nro factura": "NUMERO DE FACTURA",
    "n° factura": "NUMERO DE FACTURA",
}

BUSINESS_CONTEXT = """
Fénix Automotriz: empresa chilena (2017), reparación de carrocería y mecánica.
Ejes: Experiencia, Excelencia operacional, Transformación tecnológica, Innovación, Expansión nacional.
Misión: Servicio transparente, de calidad y puntual.
Proceso: Presupuesto → Recepción → Desarme → Desabolladura → Preparación → Pintura → Pulido → Lavado/entrega.
Roles: Gerencia, Planificación y Control, Adm/Finanzas, Comercial, Marketing, Jefe de Taller, Desarmador, Desabollador, Pintor, etc.
""".strip()

SPANISH_MONTHS = {
    "enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,
    "julio":7,"agosto":8,"septiembre":9,"setiembre":9,"octubre":10,"noviembre":11,"diciembre":12
}
SPANISH_MONTHS_ABBR = {"ene":1,"feb":2,"mar":3,"abr":4,"may":5,"jun":6,"jul":7,"ago":8,"sep":9,"set":9,"oct":10,"nov":11,"dic":12}

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", unidecode(str(s)).strip().lower())

def norm_series(s: pd.Series) -> pd.Series:
    return s.astype(str).map(lambda x: _norm(x))

def _map_col(col: str, available: List[str]) -> Optional[str]:
    if not col: return None
    if col in available: return col
    key=_norm(col)
    # match exact normalized
    for a in available:
        if _norm(a)==key: return a
    # aliases
    if key in COL_ALIASES and COL_ALIASES[key] in available: return COL_ALIASES[key]
    # partial
    for a in available:
        if key in _norm(a): return a
    return None

def _to_number(x) -> Optional[float]:
    try:
        if pd.isna(x): return None
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

def _load_sa() -> Dict[str,Any]:
    # Busca credenciales en secrets/env
    for k in ["gcp_service_account","service_account","google_service_account"]:
        try:
            obj = st.secrets[k]
            if isinstance(obj, dict) and "client_email" in obj and "private_key" in obj:
                return dict(obj)
        except Exception: pass
    for k in ["GOOGLE_CREDENTIALS","GOOGLE_CREDENTIALS_JSON","SERVICE_ACCOUNT_JSON","gcp_service_account_json"]:
        try:
            s = st.secrets.get(k,"")
            if isinstance(s,str) and s.strip().startswith("{"):
                return json.loads(s)
        except Exception: pass
    s=os.getenv("GOOGLE_CREDENTIALS","")
    if s.strip().startswith("{"): return json.loads(s)
    return {}

def get_gspread_client():
    info=_load_sa()
    if not info:
        st.error("Faltan credenciales de Google Service Account."); st.stop()
    scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds=Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)

@st.cache_data(show_spinner=False)
def get_data_from_gsheet() -> pd.DataFrame:
    gc=get_gspread_client()
    ws=gc.open_by_key(SHEET_ID).worksheet(WORKSHEET)
    rec=ws.get_all_records()
    return pd.DataFrame(rec) if rec else pd.DataFrame()

def normalize_df(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty: return raw.copy()
    # renombrado canónico
    mapping={}
    for c in raw.columns:
        key=_norm(c)
        for can in CANONICAL_FIELDS:
            if _norm(can)==key: mapping[c]=can
        if key in COL_ALIASES: mapping[c]=COL_ALIASES[key]
    df=raw.rename(columns=mapping).copy()
    # fechas
    for col in DATE_FIELDS:
        if col in df.columns:
            df[col]=pd.to_datetime(df[col], errors="coerce", dayfirst=True, infer_datetime_format=True)
    # numéricos
    for col in NUM_FIELDS:
        if col in df.columns:
            df[col]=(df[col].astype(str)
                        .str.replace(r"[ $]","",regex=True)
                        .str.replace(".","",regex=False)
                        .str.replace(",",".",regex=False))
            df[col]=pd.to_numeric(df[col], errors="coerce")
    # facturado normalizado
    if "FACTURADO" in df.columns:
        df["FACTURADO"]=(df["FACTURADO"].astype(str).str.strip().str.upper()
                         .replace({"TRUE":"SI","FALSE":"NO","1":"SI","0":"NO"}))
    return df

# =========================
#  OPENAI utils
# =========================
def get_openai_client() -> OpenAI:
    api_key=os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY",""))
    if not api_key:
        st.error("OPENAI_API_KEY no está configurada."); st.stop()
    return OpenAI(api_key=api_key)

# =========================
#  Paso 1: Analizador de filtros (LLM → JSON)
# =========================
def infer_schema_for_llm(df: pd.DataFrame) -> Dict[str,Any]:
    schema={}
    for c in df.columns:
        s=df[c]
        if pd.api.types.is_datetime64_any_dtype(s): schema[c]="date"
        elif pd.api.types.is_numeric_dtype(s):     schema[c]="number"
        else:                                      schema[c]="text"
    return schema

def get_filters_from_query(question: str, df: pd.DataFrame) -> Dict[str,Any]:
    """
    Usa un LLM como TRADUCTOR DE NEGOCIOS para producir filtros tabulares exactos.
    Devuelve dict:
      {"filters":[{"column":..,"op":..,"value":[...]}],
       "temporal_intent": "future"|"past"|null,
       "preferred_date_column": "FECHA ..."|null }
    Reglas incluidas: 'lista'→estado entregado/finalizado, 'sin aprobar'→presupuesto != Aprobado,
                      'sin factura'→ FACTURADO!='SI' o Num/Fecha vacíos, meses en español, etc.
    """
    client=get_openai_client()
    schema = json.dumps(infer_schema_for_llm(df), ensure_ascii=False)

    system = (
        "Eres un TRADUCTOR DE NEGOCIOS. Tu ÚNICA tarea es convertir una pregunta en español "
        "en filtros tabulares para un DataFrame de la empresa Fénix Automotriz. "
        "Debes usar SOLO los nombres de columna existentes. "
        "Devuelve EXCLUSIVAMENTE un JSON para la función."
    )
    # Reglas de traducción pedidas
    rules = """
- Sinónimo de ESTADO SERVICIO:
  * 'lista', 'listo', 'terminado', 'finalizado', 'entregado' → ESTADO SERVICIO IN ['ENTREGADO','FINALIZADO','TERMINADO'].
- Sinónimo de DINERO:
  * 'cuánto cuesta', 'monto', 'facturación' → usar campos de montos (p.ej. MONTO PRINCIPAL NETO).
- Sinónimo de PENDIENTE:
  * 'sin aprobar', 'no aprobado' → ESTADO PRESUPUESTO != 'APROBADO'.
- SIN FACTURA:
  * 'sin factura', 'no facturado' → FACTURADO != 'SI' OR NUMERO DE FACTURA empty OR FECHA DE FACTURACION empty.
- DUPLICADOS:
  * 'patentes duplicadas/repetidas' → agregación tipo duplicates sobre columna PATENTE.
- Meses en español:
  * 'marzo', 'sep', 'set', etc. Si se pide 'facturación de marzo', aplicar filtro por mes sobre FECHA DE FACTURACION.
- Temporalidad:
  * 'próximos días', 'en adelante', 'por pagar', 'pendiente de pago', 'vencen' → temporal_intent = 'future'.
  * 'anteriores', 'pasados' → temporal_intent = 'past'.
- Fecha por contexto:
  * Si mencionan 'pago' → columna por defecto 'FECHA DE PAGO FACTURA'.
  * Si mencionan 'entrega' → 'FECHA ENTREGA'.
  * Si mencionan 'factura/facturación' → 'FECHA DE FACTURACION'.
    """
    user = f"Esquema de columnas (nombre→tipo):\n{schema}\n\nReglas:\n{rules}\n\nPregunta del usuario:\n{question}\n\nDevuelve SOLO la llamada a función."

    tools=[{
        "type":"function",
        "function":{
            "name":"emitir_filtros",
            "description":"Devuelve filtros exactos para un DataFrame de Fénix Automotriz.",
            "parameters":{
                "type":"object",
                "properties":{
                    "filters":{"type":"array","items":{"type":"object","properties":{
                        "column":{"type":"string"},
                        "op":{"type":"string","enum":["eq","neq","gt","gte","lt","lte","contains","not_contains","in","not_in","empty","not_empty","between_dates","month_eq"]},
                        "value":{"type":"array","items":{"type":"string"}}
                    },"required":["column","op"]}},
                    "temporal_intent":{"type":["string","null"],"enum":["future","past",None]},
                    "preferred_date_column":{"type":["string","null"]}
                },
                "required":["filters"]
            }
        }
    }]

    resp=client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role":"system","content":system},
                  {"role":"user","content":user}],
        tools=tools,
        tool_choice={"type":"function","function":{"name":"emitir_filtros"}}
    )
    msg = resp.choices[0].message
    out={"filters":[], "temporal_intent": None, "preferred_date_column": None}
    try:
        if msg.tool_calls:
            cand=json.loads(msg.tool_calls[0].function.arguments)
            if isinstance(cand, dict):
                out.update({k:v for k,v in cand.items() if k in out})
    except Exception:
        pass
    return out

# =========================
#  Filtros exactos en pandas
# =========================
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
                    # comparación lexicográfica si no hay tipado
                    sv=s.astype(str); v=str(vals[0]) if vals else ""
                    if op=="gt": m=sv>v
                    if op=="gte": m=sv>=v
                    if op=="lt": m=sv<v
                    if op=="lte": m=sv<=v

            elif op in ["empty","not_empty"]:
                empty_mask = (s.isna() | (s.astype(str).str.strip()=="") |
                              (s.astype(str).str.upper().isin(["NAN","NONE","NULL","-"])))
                m = empty_mask if op=="empty" else ~empty_mask

            elif op=="between_dates":
                sd=_to_datetime(vals[0]) if len(vals)>=1 and vals[0] else None
                ed=_to_datetime(vals[1]) if len(vals)>=2 and vals[1] else None
                s2=s if pd.api.types.is_datetime64_any_dtype(s) else pd.to_datetime(s, errors="coerce", dayfirst=True)
                if sd is not None and ed is not None: m=s2.between(sd, ed)
                elif sd is not None: m=s2>=sd
                elif ed is not None: m=s2<=ed
                else: m=pd.Series(True,index=df.index)

            elif op=="month_eq":
                # valor esperado: ["3"] → marzo
                s2=s if pd.api.types.is_datetime64_any_dtype(s) else pd.to_datetime(s, errors="coerce", dayfirst=True)
                try:
                    target=int(vals[0])
                    m = s2.dt.month == target
                except Exception:
                    m = pd.Series(False,index=df.index)

            else:
                m=pd.Series(True,index=df.index)

        except Exception:
            m=pd.Series(True,index=df.index)

        mask &= m
        log.append({"column":col,"op":op,"value":vals,"remaining":int(mask.sum())})
    return df[mask].copy(), log

# =========================
#  RAG utilidades (embeddings, índice, retrieval)
# =========================
CHROMA_AVAILABLE=True
try:
    import chromadb
except Exception:
    CHROMA_AVAILABLE=False

def _fmt_value(v) -> str:
    if pd.isna(v): return ""
    if isinstance(v, pd.Timestamp): return v.strftime("%Y-%m-%d")
    if isinstance(v, (int,float)): return f"{v:.2f}".rstrip("0").rstrip(".")
    return str(v)

def _display_value_for_fragment(col: str, val) -> str:
    if (val is None) or (isinstance(val,float) and pd.isna(val)) or (str(val).strip()==""):
        if col=="OT": return "Sin asignar"
        return "N/A"
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

def get_embeddings_client():
    return get_openai_client()

def embed_texts(client: OpenAI, texts: List[str], model="text-embedding-3-small") -> List[List[float]]:
    out=[]; B=96
    for i in range(0,len(texts),B):
        resp=client.embeddings.create(model=model, input=texts[i:i+B])
        out.extend([d.embedding for d in resp.data])
    return out

def _hash_frags(frags: List[str]) -> str:
    h=hashlib.md5()
    for t in frags: h.update(t.encode("utf-8"))
    return h.hexdigest()

def ensure_index(frags: List[str], ids: List[str]):
    if len(frags)==0:
        st.session_state.subset_backend=None
        st.session_state.subset_index_hash=None
        return
    h=_hash_frags(frags)
    if st.session_state.get("subset_index_hash")==h: return
    if CHROMA_AVAILABLE:
        client=chromadb.PersistentClient(path="./.chroma")
        name=f"fenix_subset_{h[:10]}"
        try:
            coll=client.get_collection(name)
        except Exception:
            coll=client.get_or_create_collection(name)
            oai=get_embeddings_client()
            with st.spinner("Indexando subconjunto (Chroma)…"):
                embs=embed_texts(oai, frags)
                coll.add(ids=ids, documents=frags, embeddings=embs,
                         metadatas=[{"row_id":i} for i in ids])
        st.session_state.subset_backend="chroma"; st.session_state.subset_collection=coll
    else:
        os.makedirs("./.cache", exist_ok=True)
        path=f"./.cache/simple_subset_{h}.npz"
        if not os.path.exists(path):
            oai=get_embeddings_client()
            with st.spinner("Indexando subconjunto (fallback)…"):
                embs=np.array(embed_texts(oai, frags), dtype=np.float32)
            np.savez(path, embs=embs, ids=np.array(ids,dtype=object), docs=np.array(frags,dtype=object))
        d=np.load(path, allow_pickle=True)
        st.session_state.subset_embs=d["embs"]; st.session_state.subset_ids=d["ids"].tolist(); st.session_state.subset_docs=d["docs"].tolist()
        st.session_state.subset_backend="simple"
    st.session_state.subset_index_hash=h

def retrieve_top_subset(query: str, k=6) -> Tuple[List[str], List[str]]:
    if not st.session_state.get("subset_backend"): return [], []
    oai=get_embeddings_client()
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

# =========================
#  Prompts Paso 2 (respuesta)
# =========================
def system_prompt() -> str:
    return f"""
Eres un CONSULTOR DE GESTIÓN y ANALISTA DE DATOS para Fénix Automotriz.

--- INSTRUCCIONES CLAVE DE RAZONAMIENTO ---
1. INTERPRETACIÓN FLEXIBLE: Traduce la intención del usuario a los valores exactos de columnas.
   - "¿Está lista la OT X?" → revisar 'ESTADO SERVICIO' ∈ {{ENTREGADO, FINALIZADO, TERMINADO}}.
   - "¿La patente XYZ ya fue facturada?" → 'FACTURADO'=='SI' o existencia de 'NUMERO DE FACTURA'/'FECHA DE FACTURACION'.
2. SINÓNIMOS/CAMPOS: listo/terminado/entregado → ESTADO SERVICIO; sin aprobar → ESTADO PRESUPUESTO != APROBADO;
   por pagar/pendiente de pago → FECHA DE PAGO FACTURA.
3. SOLO CONTEXTO: Usa EXCLUSIVAMENTE el contexto proporcionado (ya filtrado). Si falta info, di:
   "No tengo la información necesaria en los datos".
4. MONTOS: si aplica, muestra SUMA TOTAL.
5. PASOS: (a) identifica intención y campos, (b) mapea sinónimos, (c) responde conciso.

Contexto de Negocio:
{BUSINESS_CONTEXT}
""".strip()

def build_user_prompt(question: str, context_docs: List[str]) -> str:
    ctx = "\n\n-----\n\n".join(context_docs) if context_docs else "(sin contexto)"
    instruction = ("Analiza el contexto, interpreta la intención y responde profesionalmente. "
                   "Traduce expresiones coloquiales a valores de columnas de Fénix. No salgas del contexto.")
    return f"{instruction}\n\nContexto proporcionado (subconjunto filtrado estrictamente):\n{ctx}\n\nPregunta del usuario: {question}"

# =========================
#  Orquestador Paso 1 → Paso 2
# =========================
def enforce_temporal_guardrail(filters_spec: Dict[str,Any], df: pd.DataFrame) -> Dict[str,Any]:
    """
    Si el LLM detecta 'temporal_intent'='future', fuerza >= hoy en columna de fecha preferida o inferida.
    """
    ti = filters_spec.get("temporal_intent")
    if ti != "future": return filters_spec
    col = filters_spec.get("preferred_date_column") or None
    if not col:
        q_text = ""  # ya no tenemos pregunta aquí; si quieres, pásala y decide por keywords
        # fallback: privilegia pago -> facturación -> entrega
        for c in ["FECHA DE PAGO FACTURA","FECHA DE FACTURACION","FECHA ENTREGA","FECHA INGRESO PLANTA"]:
            if c in df.columns: col=c; break
    if not col or col not in df.columns: return filters_spec
    today = dt.date.today().strftime("%Y-%m-%d")
    extra = {"column": col, "op":"gte", "value":[today]}
    filters_spec = dict(filters_spec)
    filters_spec["filters"] = (filters_spec.get("filters") or []) + [extra]
    return filters_spec

def requires_totals(question: str) -> bool:
    q=_norm(question)
    return any(k in q for k in ["total","suma","sumar","monto","factur","neto","bruto","iva","ingreso"])

def append_totals_row(display_df: pd.DataFrame) -> pd.DataFrame:
    if display_df.empty: return display_df
    total_row={c:"" for c in display_df.columns}
    label_col=next((c for c in display_df.columns if not pd.api.types.is_numeric_dtype(display_df[c])), display_df.columns[0])
    total_row[label_col]="TOTAL"
    for col in TOTAL_CANDIDATES:
        if col in display_df.columns:
            total_row[col]=pd.to_numeric(display_df[col], errors="coerce").sum(skipna=True)
    return pd.concat([display_df, pd.DataFrame([total_row], columns=display_df.columns)], ignore_index=True)

def llm_answer(question: str, base_df: pd.DataFrame, top_k: int = 6) -> Tuple[str, pd.DataFrame, Dict[str,Any], List[str]]:
    """
    Orquesta el flujo de dos pasos:
      1) get_filters_from_query → filtros estrictos
      2) apply_filters en pandas → subconjunto
      3) RAG sobre el subconjunto → docs top-k
      4) LLM para redacción final
    Devuelve: (respuesta, subset, filtros_aplicados, docs_usados)
    """
    # Paso 1: LLM traductor → filtros
    spec = get_filters_from_query(question, base_df)
    spec = enforce_temporal_guardrail(spec, base_df)  # fuerza >= hoy si 'future'

    # Aplicar filtros
    filtered_df, log = apply_filters(base_df, spec.get("filters", []))

    # Si quedó vacío, intentamos un fallback leve: quitamos 'contains/not_contains' y reintentamos
    if filtered_df.empty and spec.get("filters"):
        soft = [f for f in spec["filters"] if f.get("op") not in ("contains","not_contains")]
        if soft != spec["filters"]:
            filtered_df, log = apply_filters(base_df, soft)
            spec = dict(spec); spec["filters"] = soft

    # Sin datos → respuesta determinista
    if filtered_df.empty:
        return "No se encontraron datos que coincidan con la búsqueda.", filtered_df, spec, []

    # Paso 2a: RAG SOLO sobre el subconjunto
    frags, ids = make_fragments(filtered_df)
    ensure_index(frags, ids)
    docs, row_ids = retrieve_top_subset(question, k=top_k)

    # Si por alguna razón docs vacío (subset muy pequeño), usa hasta 10 filas del subset
    if not docs:
        sample = filtered_df.head(min(10, len(filtered_df))).apply(lambda r: row_to_fragment(r, [c for c in CANONICAL_FIELDS if c in filtered_df.columns]), axis=1).tolist()
        docs = sample

    # Paso 2b: Respuesta final
    client=get_openai_client()
    messages=[{"role":"system","content":system_prompt()},
              {"role":"user","content":build_user_prompt(question, docs)}]
    resp=client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.2)
    answer = resp.choices[0].message.content

    return answer, filtered_df, spec, docs

# =========================
#  UI / NAVEGACIÓN
# =========================
with st.sidebar:
    safe_image("Nexa_logo.png")
    st.markdown("---")
    nav = st.radio("Navegación", ["Consulta IA", "Análisis de negocio", "Configuración", "Diagnóstico y uso", "Soporte"], index=0)
    st.markdown("---")
    if st.button("🔄 Actualizar datos"):
        get_data_from_gsheet.clear()
        for k in ["subset_index_hash","subset_collection","subset_embs","subset_ids","subset_docs","subset_backend"]:
            st.session_state.pop(k, None)
        st.experimental_rerun()
    st.markdown("---")
    if st.session_state.authed:
        if st.button("Cerrar sesión"):
            st.session_state.authed=False; st.rerun()

# Login gate
if not st.session_state.authed:
    login_view()
    st.markdown('<div class="nexa-footer">Desarrollado por Nexa Corp. todos los derechos reservados.</div>', unsafe_allow_html=True)
    st.stop()

# Carga datos
raw_df = get_data_from_gsheet()
if raw_df.empty:
    st.error("La hoja MODELO_BOT está vacía o no se pudo leer."); st.stop()
df = normalize_df(raw_df)

# Top bar
c1, c2 = st.columns([5,1])
with c1:
    st.title("Consulta Nexa IA")
    st.caption("Asistente de análisis para Fénix Automotriz (RAG de dos pasos + filtros exactos)")
with c2:
    st.markdown('<div class="nexa-topbar">', unsafe_allow_html=True)
    safe_image("Fenix_isotipo.png", width=72)
    st.markdown('</div>', unsafe_allow_html=True)

USER_AVATAR = (asset("Fenix_isotipo.png") or "🛠️")
BOT_AVATAR  = (asset("Isotipo_Nexa.png") or "🤖")

def estimate_tokens(*texts) -> int:
    total_chars = sum(len(t) for t in texts if t)
    return max(1, total_chars // 4)

# =========================
#  Páginas
# =========================
def page_chat():
    with st.expander("⚙️ Parámetros de consulta", expanded=False):
        top_k = st.slider("Top-K fragmentos para contexto", 3, 15, 6, 1, key="top_k_chat")
        show_diag = st.checkbox("🔎 Mostrar diagnóstico", value=True, key="diag_chat")
        force_index = st.button("🔁 Reindexar subconjunto")

    if "messages" not in st.session_state: st.session_state.messages=[]
    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar=(USER_AVATAR if m["role"]=="user" else BOT_AVATAR)):
            st.markdown(m["content"])

    question=st.chat_input("Escribe tu consulta…")
    if not question:
        return

    # Orquestación de dos pasos
    if force_index:
        for k in ["subset_index_hash","subset_collection","subset_embs","subset_ids","subset_docs","subset_backend"]:
            st.session_state.pop(k, None)

    answer, subset, spec, docs = llm_answer(question, df.copy(), top_k=top_k)

    st.session_state.stats["queries"] += 1
    st.session_state.stats["tokens_est"] += estimate_tokens(question, *docs, answer)
    st.session_state.messages += [{"role":"user","content":question},
                                  {"role":"assistant","content":answer}]

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        if show_diag:
            with st.expander("🧭 Diagnóstico (Paso 1 → Paso 2)", expanded=False):
                st.write("**Esquema**:", infer_schema_for_llm(df))
                st.write("**Filtros generados (Paso 1)**:", spec)
                st.write("**Filas en subconjunto**:", len(subset))
                st.write("**Docs usados (muestra)**:", docs[:2])
        st.markdown(answer)

        if not subset.empty:
            show_cols=[c for c in CANONICAL_FIELDS if c in subset.columns]
            to_show = subset[show_cols] if show_cols else subset.copy()
            if requires_totals(question):
                to_show = append_totals_row(to_show)
            st.markdown("**Subconjunto filtrado (auditoría):**")
            st.dataframe(to_show, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Descargar subconjunto (CSV)",
                               data=to_show.to_csv(index=False).encode("utf-8"),
                               file_name="subconjunto_filtrado.csv", mime="text/csv")

def page_analytics():
    st.subheader("Análisis de negocio (rápido)")
    if "FECHA DE FACTURACION" in df.columns and "MONTO PRINCIPAL NETO" in df.columns and "TIPO CLIENTE" in df.columns:
        month = st.selectbox("Mes", list(SPANISH_MONTHS.keys()), index=2)  # marzo por defecto
        year = st.number_input("Año", value=dt.date.today().year, step=1)
        m = SPANISH_MONTHS[month]
        s = df["FECHA DE FACTURACION"]
        s = s if pd.api.types.is_datetime64_any_dtype(s) else pd.to_datetime(s, errors="coerce", dayfirst=True)
        sub = df[(s.dt.year==int(year)) & (s.dt.month==m)].copy()
        st.dataframe(sub, use_container_width=True, hide_index=True)
        g = (sub.groupby("TIPO CLIENTE", dropna=False)["MONTO PRINCIPAL NETO"]
               .sum(min_count=1).fillna(0.0).sort_values(ascending=False))
        st.markdown("#### Facturación por TIPO CLIENTE")
        st.bar_chart(g)
    else:
        st.info("Faltan columnas para este análisis (FECHA DE FACTURACION, TIPO CLIENTE, MONTO PRINCIPAL NETO).")

def page_settings():
    st.subheader("Configuración de apariencia")
    theme = st.selectbox("Tema de color", list(PALETTE.keys()), index=list(PALETTE.keys()).index(st.session_state.theme_name))
    if theme != st.session_state.theme_name:
        st.session_state.theme_name = theme
        apply_theme(theme)
        st.success("Tema aplicado.")
    st.caption("Estos cambios afectan la capa visual sin modificar la lógica del bot.")

def page_diagnostics():
    st.subheader("Diagnóstico y uso")
    c1, c2 = st.columns(2)
    c1.metric("Consultas en esta sesión", st.session_state.stats["queries"])
    c2.metric("Tokens estimados", f"{st.session_state.stats['tokens_est']:,}".replace(",", "."))
    st.markdown("#### Esquema detectado")
    st.json(infer_schema_for_llm(df))
    st.markdown("#### Dimensión de datos")
    st.write(f"Filas: {len(df):,} — Columnas: {len(df.columns)}".replace(",", "."))

def page_support():
    st.subheader("Soporte")
    st.markdown("- **Correo:** soporte@nexa.cl")
    st.markdown("- **Web:** www.nexa.cl")
    safe_image("Nexa_logo.png", width=180)

# Router
with st.sidebar:
    pass
if "nav" not in st.session_state:
    pass

with st.sidebar:
    pass

with st.sidebar:
    pass

nav_choice = nav
if nav_choice == "Consulta IA":
    page_chat()
elif nav_choice == "Análisis de negocio":
    page_analytics()
elif nav_choice == "Configuración":
    page_settings()
elif nav_choice == "Diagnóstico y uso":
    page_diagnostics()
elif nav_choice == "Soporte":
    page_support()

# Footer
st.markdown('<div class="nexa-footer">Desarrollado por Nexa Corp. todos los derechos reservados.</div>', unsafe_allow_html=True)
