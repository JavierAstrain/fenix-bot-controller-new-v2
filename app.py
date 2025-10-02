# -*- coding: utf-8 -*-
"""
Consulta Nexa IA — Fénix Automotriz
Corrección definitiva:
- Carga segura: SOLO columnas listadas en DATE_COLUMNS se convierten a datetime64[ns] (Timestamp)
- Reglas de negocio reforzadas: "facturas a pagar (futuro)" => FACTURADO == 'SI' + FECHA DE PAGO FACTURA >= HOY
- Comparaciones de fecha SIEMPRE con pandas.Timestamp
"""

import os, re, json, hashlib
import datetime as dt
from calendar import monthrange
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from unidecode import unidecode
from openai import OpenAI

# =========================
#  ASSETS
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
    page_icon=safe_page_icon("Isotipo_Nexa.png"),
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

# === SOLO ESTAS se convierten a fecha (evita corromper el DF) ===
DATE_COLUMNS = [
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

# =========================
#  UTILS
# =========================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", unidecode(str(s)).strip().lower())

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

def _to_datetime(x) -> Optional[pd.Timestamp]:
    try:
        return pd.to_datetime(x, dayfirst=True, errors="coerce")
    except Exception:
        return None

def _series_as_ts(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

# =========================
#  TIEMPO CONSISTENTE
# =========================
TODAY_TS = pd.Timestamp(dt.date.today())

def month_range_boundaries(year: int, month: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=int(year), month=int(month), day=1)
    end_day = monthrange(int(year), int(month))[1]
    end = pd.Timestamp(year=int(year), month=int(month), day=end_day)
    return start, end

def parse_month_year_from_text(q: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    qn=_norm(q)
    if re.search(r"\beste mes\b", qn):
        return TODAY_TS.month, TODAY_TS.year, "este mes"
    if re.search(r"\bmes pasado\b", qn):
        prev = (TODAY_TS - pd.DateOffset(months=1))
        return prev.month, prev.year, "mes pasado"
    if re.search(r"\bproximo mes\b|\bpróximo mes\b", qn):
        nxt = (TODAY_TS + pd.DateOffset(months=1))
        return nxt.month, nxt.year, "próximo mes"
    for name,num in SPANISH_MONTHS.items():
        if re.search(rf"\b{name}\b", qn):
            y = re.search(r"(19|20)\d{2}", qn)
            return num, (int(y.group(0)) if y else TODAY_TS.year), name
    for abbr,num in SPANISH_MONTHS_ABBR.items():
        if re.search(rf"\b{abbr}\b\.?", qn):
            y = re.search(r"(19|20)\d{2}", qn)
            return num, (int(y.group(0)) if y else TODAY_TS.year), abbr
    return None, None, None

def parse_next_days_from_text(q: str) -> Optional[int]:
    qn=_norm(q)
    m=re.search(r"(proxim(?:os)?|siguientes)\s+(\d{1,3})\s+dias", qn)
    if m:
        try:
            return int(m.group(2))
        except Exception:
            return None
    return None

def detect_future_intent_text(q: str) -> bool:
    qn=_norm(q)
    keys=["proxim","próxim","pronto","futuro","en adelante","desde hoy","a partir de hoy",
          "hoy en adelante","pendiente de pago","pendientes de pago","por pagar","a pagar",
          "vencen","vencimiento","por vencer"]
    return any(k in qn for k in keys)

def detect_past_intent_text(q: str) -> bool:
    qn=_norm(q)
    return any(k in qn for k in ["pasado","anteriores","historial","histórico"])

def detect_unpaid_intent(q: str) -> bool:
    qn=_norm(q)
    return any(k in qn for k in ["por pagar","no pagad","pendiente de pago","pendientes de pago","a pagar","por vencer"])

def choose_date_column_by_context(question: str, df: pd.DataFrame) -> Optional[str]:
    q=_norm(question); pref=[]
    if "pago" in q or "pagar" in q or "venc" in q: pref+=["FECHA DE PAGO FACTURA"]
    if "factur" in q or "factura" in q:           pref+=["FECHA DE FACTURACION"]
    if "entreg" in q:                             pref+=["FECHA ENTREGA"]
    if "ingres" in q:                             pref+=["FECHA INGRESO PLANTA"]
    if "recep" in q:                              pref+=["FECHA RECEPCION"]
    pref += ["FECHA DE PAGO FACTURA","FECHA DE FACTURACION","FECHA ENTREGA","FECHA INGRESO PLANTA","FECHA RECEPCION"]
    for c in pref:
        if c in df.columns: return c
    return None

# =========================
#  CARGA GOOGLE SHEETS
# =========================
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
    """
    Normaliza nombres, numéricos, y convierte SOLO DATE_COLUMNS a Timestamp.
    Evita convertir columnas erróneas como PATENTE o MONTO a fechas.
    """
    if raw.empty: return raw.copy()

    # Mapear encabezados a canónicos
    mapping={}
    for c in raw.columns:
        key=_norm(c)
        for can in CANONICAL_FIELDS:
            if _norm(can)==key: mapping[c]=can
        if key in COL_ALIASES: mapping[c]=COL_ALIASES[key]
    df=raw.rename(columns=mapping).copy()

    # === SOLO estas columnas se convierten a fecha (Timestamp) ===
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    # Números (los conocidos)
    num_fields = [c for c in NUM_FIELDS if c in df.columns]
    for col in num_fields:
        df[col]=(df[col].astype(str)
                    .str.replace(r"[ $]","",regex=True)
                    .str.replace(".","",regex=False)
                    .str.replace(",",".",regex=False))
        df[col]=pd.to_numeric(df[col], errors="coerce")

    # Facturado normalizado
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
#  Paso 1: LLM → filtros estrictos (con regla reforzada)
# =========================
def infer_schema_for_llm(df: pd.DataFrame) -> Dict[str,Any]:
    """
    NO intentar inferir fecha con pd.to_datetime en columnas arbitrarias (corrompe).
    Solo marcar como 'date' si está en DATE_COLUMNS; luego number/text por dtype.
    """
    schema={}
    for c in df.columns:
        if c in DATE_COLUMNS:
            schema[c]="date"
        elif pd.api.types.is_numeric_dtype(df[c]):
            schema[c]="number"
        else:
            schema[c]="text"
    return schema

def _contains_factura_pago_futuro(question: str) -> bool:
    q=_norm(question)
    return (("factur" in q or "factura" in q) and
            any(k in q for k in ["pagar","pago","venc","por pagar","pendiente","futuro","próxim","proxim"]))

def enforce_business_rules_on_filters(question: str, spec: Dict[str,Any], df: pd.DataFrame) -> Dict[str,Any]:
    """
    Si la intención es "facturas a pagar (futuro)":
      - FACTURADO == 'SI'
      - FECHA DE PAGO FACTURA >= HOY
    Agregar si faltan o corregir si vinieron mal.
    """
    out = dict(spec)
    flt = list(out.get("filters") or [])
    cols = list(df.columns)

    def add_or_replace(column: str, op: str, value: List[str]):
        nonlocal flt
        mapped = _map_col(column, cols)
        if not mapped: return
        # Eliminar entradas previas del mismo col que choquen
        flt = [f for f in flt if _map_col(f.get("column",""), cols) != mapped]
        flt.append({"column": mapped, "op": op, "value": value})

    if _contains_factura_pago_futuro(question):
        # 1) FACTURADO == 'SI'
        add_or_replace("FACTURADO", "eq", ["SI"])
        # 2) FECHA DE PAGO FACTURA >= HOY (gte_today)
        if "FECHA DE PAGO FACTURA" in cols:
            add_or_replace("FECHA DE PAGO FACTURA", "gte_today", [TODAY_TS.isoformat()])

        # Además marca la intención temporal si no viene
        if not out.get("temporal_intent"):
            out["temporal_intent"] = "future"
        if not out.get("preferred_date_column"):
            out["preferred_date_column"] = "FECHA DE PAGO FACTURA"

    out["filters"] = flt
    return out

def get_filters_from_query(question: str, df: pd.DataFrame) -> Dict[str,Any]:
    client=get_openai_client()
    schema = json.dumps(infer_schema_for_llm(df), ensure_ascii=False)

    system = (
        "Eres un TRADUCTOR DE NEGOCIOS. Convierte una pregunta en filtros tabulares EXACTOS para el DataFrame de Fénix. "
        "Usa SOLO nombres de columnas existentes. Devuelve SOLO JSON."
    )
    rules = """
REGLAS DE NEGOCIO CLAVE:
- 'lista/terminado/entregado/finalizado' → ESTADO SERVICIO IN ['ENTREGADO','FINALIZADO','TERMINADO'].
- 'sin aprobar' → ESTADO PRESUPUESTO != 'APROBADO'.
- 'sin factura/no facturado' → FACTURADO != 'SI' OR NUMERO DE FACTURA empty OR FECHA DE FACTURACION empty.
- Meses en español (enero..dic / ene..dic). Si piden 'de marzo', preferir FECHA DE FACTURACION.
- Temporalidad:
  * 'próximos días', 'en adelante', 'por pagar', 'pendientes de pago', 'vencen' → temporal_intent='future'
  * 'anteriores', 'pasado' → temporal_intent='past'
- Columna fecha preferida: pago→FECHA DE PAGO FACTURA; factura→FECHA DE FACTURACION; entrega→FECHA ENTREGA.

REGLA REFORZADA (FACTURAS A PAGAR - FUTURO):
- Si la pregunta menciona facturas y una condición temporal futura (por pagar, vencen, próximos días/mes):
  * Añade: {"column":"FACTURADO","op":"eq","value":["SI"]}
  * Añade: {"column":"FECHA DE PAGO FACTURA","op":"gte_today","value":[<hoy ISO>]}
"""
    user = f"Esquema (columna:tipo):\n{schema}\n\nReglas:\n{rules}\n\nPregunta:\n{question}\n\nDevuelve SOLO la llamada a función."

    tools=[{
        "type":"function",
        "function":{
            "name":"emitir_filtros",
            "description":"Devuelve filtros exactos (JSON).",
            "parameters":{
                "type":"object",
                "properties":{
                    "filters":{"type":"array","items":{"type":"object","properties":{
                        "column":{"type":"string"},
                        "op":{"type":"string","enum":["eq","neq","gt","gte","lt","lte","contains","not_contains","in","not_in","between_dates","month_eq","empty","not_empty","future","gte_today","past","lt_today"]},
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

    # Refuerzo post-LLM para "facturas a pagar (futuro)"
    out = enforce_business_rules_on_filters(question, out, df)
    return out

# =========================
#  ENFORCER DE TIEMPO (agrega filtros de fecha cuando falten)
# =========================
def month_range_boundaries_dates(year: int, month: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=int(year), month=int(month), day=1)
    end_day = monthrange(int(year), int(month))[1]
    end   = pd.Timestamp(year=int(year), month=int(month), day=end_day)
    return start, end

def time_enforcer_attach(question: str, spec: Dict[str,Any], df: pd.DataFrame) -> Tuple[Dict[str,Any], Dict[str,Any]]:
    diag = {"temporal": None, "month": None, "year": None, "date_col": None, "added": []}
    spec = dict(spec); spec_filters = list(spec.get("filters") or [])
    date_col = spec.get("preferred_date_column") or choose_date_column_by_context(question, df)
    if date_col and date_col not in df.columns:
        date_col = None

    ndays = parse_next_days_from_text(question)
    if ndays and not date_col:
        date_col = choose_date_column_by_context(question, df)
    if ndays and date_col:
        start = TODAY_TS
        end   = TODAY_TS + pd.Timedelta(days=int(ndays))
        spec_filters = [f for f in spec_filters if _map_col(f.get("column",""), list(df.columns)) != date_col]
        spec_filters.append({"column": date_col, "op":"between_dates",
                             "value":[start.isoformat(), end.isoformat()]})
        diag.update({"temporal":"next_days","date_col":date_col,"added":diag["added"]+[{"type":"next_days","start":str(start.date()),"end":str(end.date())}]})

    # Mes explícito
    m, y, token = parse_month_year_from_text(question)
    if m is not None:
        if y is None: y = TODAY_TS.year
        if not date_col:
            date_col = choose_date_column_by_context(question, df)
        if date_col:
            start, end = month_range_boundaries_dates(y, m)
            spec_filters = [f for f in spec_filters if _map_col(f.get("column",""), list(df.columns)) != date_col]
            spec_filters.append({"column": date_col, "op": "between_dates",
                                 "value": [start.isoformat(), end.isoformat()]})
            diag.update({"month": m, "year": y, "date_col": date_col, "temporal": "month_between",
                         "added": diag["added"] + [{"type":"month_between", "col":date_col, "start":str(start.date()), "end":str(end.date())}]})

    # FUTURO / PASADO
    ti = spec.get("temporal_intent")
    if ti is None:
        if detect_future_intent_text(question): ti="future"
        elif detect_past_intent_text(question): ti="past"
    if ti and date_col and not any(_map_col(f.get("column",""), list(df.columns))==date_col and f.get("op")=="between_dates" for f in spec_filters):
        today_str = TODAY_TS.isoformat()
        spec_filters = [f for f in spec_filters if _map_col(f.get("column",""), list(df.columns)) != date_col]
        if ti == "future":
            spec_filters.append({"column": date_col, "op":"gte_today", "value":[today_str]})
        elif ti == "past":
            spec_filters.append({"column": date_col, "op":"lt_today", "value":[today_str]})
        diag.update({"temporal": ti, "date_col": date_col,
                     "added": diag["added"] + [{"type":ti, "col":date_col, "value":str(TODAY_TS.date())}]})

    spec["filters"] = spec_filters
    return spec, diag

# =========================
#  Filtros exactos en pandas (Timestamp)
# =========================
def _ensure_list(x):
    if isinstance(x,list): return x
    if x is None: return []
    return [x]

def _month_token_to_int(tok: str) -> Optional[int]:
    if tok is None: return None
    t=_norm(tok)
    if t.isdigit():
        try:
            n=int(t)
            return n if 1 <= n <= 12 else None
        except Exception:
            return None
    if t in SPANISH_MONTHS: return SPANISH_MONTHS[t]
    if t in SPANISH_MONTHS_ABBR: return SPANISH_MONTHS_ABBR[t]
    return None

def apply_filters(df: pd.DataFrame, filters: List[Dict[str,Any]]) -> Tuple[pd.DataFrame, List[Dict[str,Any]]]:
    """
    Comparaciones de fecha SIEMPRE con Timestamp:
        - gt/gte/lt/lte (fecha)
        - between_dates
        - month_eq
        - future/gte_today, past/lt_today
    """
    if df.empty or not filters: 
        return df.copy(), []

    df2 = df.copy()
    cols=list(df2.columns)
    today_ts = TODAY_TS
    mask = pd.Series(True, index=df2.index)
    log: List[Dict[str,Any]] = []

    def _to_num_like(x):
        try:
            if pd.isna(x): return None
            if isinstance(x,(int,float)): return float(x)
            s=str(x).strip().replace(" ","").replace("$","").replace(".","").replace(",",".")
            v=pd.to_numeric(s, errors="coerce")
            return None if pd.isna(v) else float(v)
        except Exception:
            return None

    for f in filters:
        col=_map_col(str(f.get("column","")), cols)
        op=str(f.get("op","")).strip().lower()
        vals=_ensure_list(f.get("value", []))
        if not col or col not in cols or not op:
            continue

        s=df2[col]
        m=pd.Series(True, index=df2.index)

        try:
            is_date = (col in DATE_COLUMNS)  # ESTRICTO: solo las declaradas
            s_ts = _series_as_ts(s) if is_date else None

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
                # NUMÉRICO
                if pd.api.types.is_numeric_dtype(s):
                    sn=pd.to_numeric(s, errors="coerce")
                    val=_to_num_like(vals[0]) if vals else None
                    if val is None: m=pd.Series(False,index=df2.index)
                    else:
                        if op=="gt":  m=sn>val
                        if op=="gte": m=sn>=val
                        if op=="lt":  m=sn<val
                        if op=="lte": m=sn<=val

                # FECHA (Timestamp)
                elif is_date:
                    val_ts=None
                    if vals:
                        token=str(vals[0]).strip().upper()
                        if token in ("HOY","TODAY","NOW"): val_ts=today_ts
                        else:
                            td=_to_datetime(vals[0])
                            val_ts = td if (td is not None and pd.notna(td)) else None
                    if val_ts is None and op in ("gt","gte"):
                        val_ts=today_ts
                    has_date = s_ts.notna()
                    if val_ts is None:
                        m = pd.Series(False, index=df2.index)
                    else:
                        if op=="gt":  m = has_date & (s_ts >  val_ts)
                        if op=="gte": m = has_date & (s_ts >= val_ts)
                        if op=="lt":  m = has_date & (s_ts <  val_ts)
                        if op=="lte": m = has_date & (s_ts <= val_ts)

                else:  # TEXTO fallback
                    sv=s.astype(str)
                    v=str(vals[0]) if vals else ""
                    if op=="gt":  m=sv>v
                    if op=="gte": m=sv>=v
                    if op=="lt":  m=sv<v
                    if op=="lte": m=sv<=v

            elif op in ["empty","not_empty"]:
                empty_mask = (s.isna() | (s.astype(str).str.strip()=="") |
                              (s.astype(str).str.upper().isin(["NAN","NONE","NULL","-"])))
                m = empty_mask if op=="empty" else ~empty_mask

            elif op=="between_dates":
                if is_date:
                    sd=_to_datetime(vals[0]) if len(vals)>=1 and vals[0] else None
                    ed=_to_datetime(vals[1]) if len(vals)>=2 and vals[1] else None
                    has_date = s_ts.notna()
                    m = has_date
                    if sd is not None and pd.notna(sd): m = m & (s_ts >= sd)
                    if ed is not None and pd.notna(ed): m = m & (s_ts <= ed)
                else:
                    m = pd.Series(True, index=df2.index)

            elif op=="month_eq":
                if is_date:
                    month_token = vals[0] if vals else None
                    target = None
                    if month_token is not None:
                        t=_norm(str(month_token))
                        target = int(t) if t.isdigit() else (SPANISH_MONTHS.get(t) or SPANISH_MONTHS_ABBR.get(t))
                    if target is None:
                        m = pd.Series(False, index=df2.index)
                    else:
                        m = (s_ts.dt.month == int(target)).fillna(False)
                else:
                    m = pd.Series(False, index=df2.index)

            elif op in ["future","gte_today",">hoy"]:
                if is_date:
                    has_date = s_ts.notna()
                    m = has_date & (s_ts >= today_ts)
                else:
                    m = pd.Series(False, index=df2.index)

            elif op in ["past","lt_today","<hoy"]:
                if is_date:
                    has_date = s_ts.notna()
                    m = has_date & (s_ts < today_ts)
                else:
                    m = pd.Series(False, index=df2.index)

            else:
                m=pd.Series(True,index=df2.index)

        except Exception:
            m=pd.Series(True,index=df2.index)

        mask &= m
        log.append({"column":col,"op":op,"value":vals,"remaining":int(mask.sum())})

    return df2[mask].copy(), log

# =========================
#  DURACIÓN
# =========================
def detect_duration_intent(question: str) -> bool:
    q=_norm(question)
    keys = [
        "mas tiempo","más tiempo","lleva mas","lleva más","dias en taller","días en taller",
        "tiempo en taller","antiguedad en taller","antigüedad en taller","desde ingreso","desde la recepcion"
    ]
    return any(k in q for k in keys)

def compute_duration_table(df: pd.DataFrame, top_n: int = 10) -> Tuple[str, pd.DataFrame]:
    if df.empty:
        return "No hay datos para calcular duración.", df
    base = df.copy()
    col = "FECHA INGRESO PLANTA" if "FECHA INGRESO PLANTA" in base.columns else ("FECHA RECEPCION" if "FECHA RECEPCION" in base.columns else None)
    if not col:
        return "No existe columna de ingreso/recepción para calcular duración.", df
    s_ts = _series_as_ts(base[col])
    base["DIAS_EN_TALLER_CALC"] = (TODAY_TS - s_ts).dt.days
    out = base.sort_values("DIAS_EN_TALLER_CALC", ascending=False).head(top_n)
    cols = [c for c in ["OT","PATENTE","NOMBRE CLIENTE","ESTADO SERVICIO","FECHA INGRESO PLANTA","DIAS_EN_TALLER_CALC"] if c in out.columns]
    return f"Top {len(out)} vehículos con mayor tiempo en taller (al {TODAY_TS.date()}):", out[cols] if cols else out

# =========================
#  RAG (embeddings, índice, retrieval) + formateo robusto
# =========================
CHROMA_AVAILABLE=True
try:
    import chromadb
except Exception:
    CHROMA_AVAILABLE=False

def _fmt_value(v) -> str:
    try:
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            return ""
    except Exception:
        pass
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() in {"nan", "none", "null", "nat", "-"}:
            return ""
        return s
    if isinstance(v, (pd.Timestamp, dt.datetime, dt.date)):
        try:
            return pd.Timestamp(v).strftime("%Y-%m-%d")
        except Exception:
            return str(v)
    if isinstance(v, (int, float)):
        try:
            if pd.isna(v):
                return ""
        except Exception:
            pass
        return f"{float(v):.2f}".rstrip("0").rstrip(".")
    return str(v)

def _display_value_for_fragment(col: str, val) -> str:
    try:
        is_null = (val is None) or (isinstance(val, str) and val.strip()=="")
        if not is_null:
            try:
                if not isinstance(val, str) and pd.isna(val):
                    is_null = True
            except Exception:
                pass
        if is_null:
            return "Sin asignar" if col == "OT" else "N/A"
    except Exception:
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
1) Usa SOLO el contexto filtrado. 2) Interpreta intención y responde conciso. 3) Si falta info, dilo.
Contexto de Negocio:
{BUSINESS_CONTEXT}
""".strip()

def build_user_prompt(question: str, context_docs: List[str]) -> str:
    ctx = "\n\n-----\n\n".join(context_docs) if context_docs else "(sin contexto)"
    instruction = ("Analiza el contexto y responde profesionalmente. No inventes datos.")
    return f"{instruction}\n\nContexto proporcionado:\n{ctx}\n\nPregunta del usuario: {question}"

# =========================
#  Orquestador (Dos Pasos + Tiempo)
# =========================
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

def llm_answer(question: str, base_df: pd.DataFrame, top_k: int = 6):
    # Duración determinista
    if detect_duration_intent(question):
        txt, tbl = compute_duration_table(base_df)
        return txt, tbl, {"filters":[]}, [], {"duration": True}

    spec = get_filters_from_query(question, base_df)
    spec, time_diag = time_enforcer_attach(question, spec, base_df)
    filtered_df, log = apply_filters(base_df, spec.get("filters", []))

    # Post-filtro "por pagar / no pagadas": fecha de pago NaN o >= HOY (Timestamp)
    if detect_unpaid_intent(question) and isinstance(filtered_df, pd.DataFrame) and not filtered_df.empty:
        if "FECHA DE PAGO FACTURA" in filtered_df.columns:
            s_ts = _series_as_ts(filtered_df["FECHA DE PAGO FACTURA"])
            filtered_df = filtered_df[(s_ts.isna()) | (s_ts >= TODAY_TS)]

    if (not isinstance(filtered_df, pd.DataFrame)) or filtered_df.empty:
        soft = [f for f in spec.get("filters", []) if f.get("op") not in ("contains","not_contains")]
        if soft != spec.get("filters", []):
            filtered_df, log = apply_filters(base_df, soft)
            spec = dict(spec); spec["filters"] = soft

    if isinstance(filtered_df, pd.DataFrame) and filtered_df.empty:
        return "No se encontraron datos que coincidan con la búsqueda.", filtered_df, spec, [], time_diag

    frags, ids = make_fragments(filtered_df if isinstance(filtered_df, pd.DataFrame) else base_df)
    ensure_index(frags, ids)
    docs, row_ids = retrieve_top_subset(question, k=top_k)
    if not docs and isinstance(filtered_df, pd.DataFrame) and not filtered_df.empty:
        sample = filtered_df.head(min(10, len(filtered_df))).apply(lambda r: row_to_fragment(r, [c for c in CANONICAL_FIELDS if c in filtered_df.columns]), axis=1).tolist()
        docs = sample

    client=get_openai_client()
    messages=[{"role":"system","content":system_prompt()},
              {"role":"user","content":build_user_prompt(question, docs)}]
    resp=client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.2)
    answer = resp.choices[0].message.content

    return answer, filtered_df, spec, docs, time_diag

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
    st.caption("Asistente de análisis para Fénix Automotriz (RAG de dos pasos + fechas consistentes)")
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
def requires_totals(question: str) -> bool:
    q=_norm(question)
    return any(k in q for k in ["total","suma","sumar","monto","factur","neto","bruto","iva","ingreso"])

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

    if force_index:
        for k in ["subset_index_hash","subset_collection","subset_embs","subset_ids","subset_docs","subset_backend"]:
            st.session_state.pop(k, None)

    answer, subset, spec, docs, time_diag = llm_answer(question, df.copy(), top_k=top_k)

    st.session_state.stats["queries"] += 1
    st.session_state.stats["tokens_est"] += estimate_tokens(question, *(docs or []), answer)
    st.session_state.messages += [{"role":"user","content":question},
                                  {"role":"assistant","content":answer}]

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        if show_diag:
            with st.expander("🧭 Diagnóstico (Tiempo + Filtros)", expanded=False):
                st.write("**Hoy (Timestamp)**:", str(TODAY_TS))
                st.write("**Esquema**:", infer_schema_for_llm(df))
                st.write("**Filtros (Paso 1 final)**:", spec)
                if isinstance(subset, pd.DataFrame):
                    st.write("**Filas en subconjunto**:", len(subset))
                st.write("**Docs usados (muestra)**:", (docs or [])[:2])
        st.markdown(answer)

        if isinstance(subset, pd.DataFrame) and not subset.empty:
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
        month_names = list(SPANISH_MONTHS.keys())
        default_idx = TODAY_TS.month - 1
        month = st.selectbox("Mes", month_names, index=default_idx)
        year = st.number_input("Año", value=int(TODAY_TS.year), step=1)
        m = SPANISH_MONTHS[month]
        s_dt = _series_as_ts(df["FECHA DE FACTURACION"])
        start, end = month_range_boundaries(int(year), int(m))
        sub = df[(s_dt >= pd.Timestamp(start)) & (s_dt <= pd.Timestamp(end))].copy()
        st.dataframe(sub, use_container_width=True, hide_index=True)
        g = (sub.groupby("TIPO CLIENTE", dropna=False)["MONTO PRINCIPAL NETO"]
               .sum(min_count=1).fillna(0.0).sort_values(ascending=False))
        st.markdown(f"#### Facturación por TIPO CLIENTE — {int(year)}-{int(m):02d}")
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
    st.markdown("#### Hoy (Timestamp Pandas)")
    st.write(str(TODAY_TS))
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

if not st.session_state.authed:
    login_view()
else:
    if nav == "Consulta IA":
        page_chat()
    elif nav == "Análisis de negocio":
        page_analytics()
    elif nav == "Configuración":
        page_settings()
    elif nav == "Diagnóstico y uso":
        page_diagnostics()
    elif nav == "Soporte":
        page_support()

# Footer
st.markdown('<div class="nexa-footer">Desarrollado por Nexa Corp. todos los derechos reservados.</div>', unsafe_allow_html=True)
