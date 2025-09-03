import os
import io
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

# ---------- Config & Env ----------
st.set_page_config(page_title="KPI Vendas • Instituto Plenum", layout="wide")
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")

# Lazy import para evitar falhas se a lib não estiver instalada ainda
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    openai_client = None

HISTORY_PATH = "kpi_history.json"

# ---------- Helpers ----------
def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(entry):
    history = load_history()
    history.append(entry)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def read_any_file(uploaded):
    frames = []
    if not uploaded:
        return pd.DataFrame()
    if not isinstance(uploaded, list):
        uploaded = [uploaded]
    for up in uploaded:
        name = up.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(up, dtype=str, encoding="utf-8", sep=None, engine="python")
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            df = pd.read_excel(up, dtype=str)
        else:
            st.warning(f"Formato não suportado: {up.name}")
            continue
        df["__source_file"] = up.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def normalize_cols(df):
    colmap = {}
    for c in df.columns:
        norm = (
            c.strip().lower()
            .replace("ã", "a").replace("õ", "o").replace("ç", "c")
            .replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u")
        )
        colmap[norm] = c
    return colmap

def coerce_date(s):
    if pd.isna(s):
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(str(s), fmt).date()
        except Exception:
            continue
    try:
        return pd.to_datetime(s, errors="coerce").date()
    except Exception:
        return None

def to_numeric_safe(x):
    try:
        if isinstance(x, str):
            x = x.replace(".", "").replace(" ", "").replace("R$", "").replace(",", ".")
        return float(x)
    except Exception:
        return np.nan

def month_le(date_obj):
    if date_obj is None or pd.isna(date_obj):
        return np.nan
    return date_obj.month

# ---------- OpenAI (com fallback) ----------
def openai_insights(prompt_text: str, max_tokens: int = 500):
    if not openai_client or not OPENAI_API_KEY:
        return ""  # sem erro aqui; deixamos o local cobrir
    try:
        # Usando Responses API moderna
        resp = openai_client.responses.create(
            model=OPENAI_MODEL,
            input=prompt_text,
            max_output_tokens=max_tokens,
        )
        return (getattr(resp, "output_text", "") or "").strip()
    except Exception as e:
        # Se der erro de modelo/chave, apenas retorna vazio e seguimos com insights locais
        return ""

# ---------- Insights locais (regras) ----------
def pct(x):
    try:
        return f"{x*100:.1f}%"
    except Exception:
        return "-"

def format_currency(v):
    try:
        return f"R$ {float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "R$ 0,00"

def build_local_insights(df_filtered):
    """
    Gera um bloco de texto com bullets de insights:
    - Top cidades/estados por venda
    - Concentração
    - 'Cidades que mais valem a pena investir' com base em momentum (crescimento últimos 3 meses vs. 3 anteriores)
    - Observações de sazonalidade até julho
    """
    lines = []

    # --- Top cidades e estados ---
    by_city = (
        df_filtered.groupby(["_cidade", "_estado"], dropna=True, as_index=False)["_valor"].sum()
        .sort_values("_valor", ascending=False)
    )
    by_state = (
        df_filtered.groupby(["_estado"], dropna=True, as_index=False)["_valor"].sum()
        .sort_values("_valor", ascending=False)
    )
    total = float(np.nansum(df_filtered["_valor"]))

    top_cities_list = by_city.head(5).to_dict(orient="records")
    top_states_list = by_state.head(5).to_dict(orient="records")

    if len(top_cities_list) > 0:
        nomes = [f"{r['_cidade']} ({r['_estado']})" for r in top_cities_list]
        share_top5 = np.nansum([r["_valor"] for r in top_cities_list]) / total if total > 0 else 0
        lines.append(f"**As cidades que mais tiveram vendas foram**: {', '.join(nomes)}.")
        lines.append(f"Essas 5 cidades concentram **{pct(share_top5)}** do total no recorte atual.")
    else:
        lines.append("Não há cidades com vendas no recorte atual.")

    if len(top_states_list) > 0:
        nomes = [f"{r['_estado']}" for r in top_states_list]
        share_uf5 = np.nansum([r["_valor"] for r in top_states_list]) / total if total > 0 else 0
        lines.append(f"**Os estados com maiores vendas**: {', '.join(nomes)}.")
        lines.append(f"Top 5 UFs respondem por **{pct(share_uf5)}** do total.")
    else:
        lines.append("Não há estados com vendas no recorte atual.")

    # --- Momentum por cidade (últimos 3 meses vs 3 anteriores) ---
    df_d = df_filtered.dropna(subset=["_data"]).copy()
    if not df_d.empty:
        df_d["yyyymm"] = df_d["_data"].apply(lambda d: d.year * 100 + d.month)
        last_ym = int(df_d["yyyymm"].max())
        recent_range = [last_ym - i for i in range(0, 3)]
        prev_range = [last_ym - i for i in range(3, 6)]

        recent = df_d[df_d["yyyymm"].isin(recent_range)].groupby(["_cidade", "_estado"], as_index=False)["_valor"].sum()
        recent.rename(columns={"_valor": "recent"}, inplace=True)
        prev = df_d[df_d["yyyymm"].isin(prev_range)].groupby(["_cidade", "_estado"], as_index=False)["_valor"].sum()
        prev.rename(columns={"_valor": "prev"}, inplace=True)

        mom = pd.merge(recent, prev, on=["_cidade", "_estado"], how="left")
        mom["prev"] = mom["prev"].fillna(0.0)
        mom["delta"] = mom["recent"] - mom["prev"]
        mom["growth"] = np.where(mom["prev"] > 0, mom["delta"] / mom["prev"], np.where(mom["recent"] > 0, 1.0, 0.0))
        mom = mom.sort_values(["growth", "recent"], ascending=[False, False])

        # Filtra “oportunidades” (crescimento > 30% e não necessariamente no top total)
        opp = mom[(mom["growth"] >= 0.30) & (mom["recent"] >= 1)]
        opp = opp.head(5).to_dict(orient="records")

        if opp:
            frases = [f"{o['_cidade']} ({o['_estado']}, {pct(o['growth'])} vs. 3 meses anteriores)" for o in opp]
            lines.append("**As cidades que mais valem a pena investir (momentum recente)**: " + ", ".join(frases) + ".")
        else:
            lines.append("Não foram identificadas cidades com **crescimento relevante** nos últimos 3 meses em relação aos 3 anteriores.")
    else:
        lines.append("Sem datas válidas para avaliar **momentum** por cidade.")

    # --- Observação 'até julho' ---
    # Se o usuário filtrou “até julho”, reforçamos a leitura
    meses = df_filtered["_mes"].dropna().unique().tolist()
    if len(meses) > 0 and max(meses) <= 7:
        lines.append("A análise considera **compras até julho**; atenção à **sazonalidade** do período.")

    # --- Recomendações táticas (curtas) ---
    if len(top_cities_list) > 0:
        foco = top_cities_list[0]
        lines.append(f"Sugestão: priorize **{foco['_cidade']} ({foco['_estado']})** com ações de cross-sell/upsell (ticket atual: {format_currency(foco['_valor'])} no recorte).")

    return "### 📌 Insights quantitativos\n\n- " + "\n- ".join(lines)

# ---------- UI helpers ----------
def kpi_card(label, value, help_text=None):
    st.metric(label, value)
    if help_text:
        st.caption(help_text)

def top_n(df, group_cols, value_col, n=10):
    grp = df.groupby(group_cols, dropna=False, as_index=False)[value_col].sum()
    grp = grp.sort_values(value_col, ascending=False).head(n)
    return grp

# ---------- Sidebar ----------
st.sidebar.title("⚙️ Configurações")
uploaded = st.sidebar.file_uploader("Carregar planilhas (CSV/XLSX) — pode selecionar várias", type=["csv", "xlsx", "xls"], accept_multiple_files=True)

if uploaded:
    df_raw = read_any_file(uploaded)
else:
    df_raw = pd.DataFrame()

if df_raw.empty:
    st.title("KPI de Vendas com Insights (OpenAI + Streamlit)")
    st.info("Carregue suas planilhas na barra lateral para começar.")
    st.divider()
    # histórico mesmo sem dados
    st.subheader("🗂️ Histórico de KPIs (conversa)")
    hist = load_history()
    if not hist:
        st.caption("Ainda não há histórico salvo.")
    else:
        for i, h in enumerate(reversed(hist), 1):
            with st.expander(f"{h.get('timestamp')} • {h.get('title','KPI salvo')}"):
                for msg in h.get("chat", []):
                    with st.chat_message(msg.get("role", "assistant")):
                        st.markdown(msg.get("content", ""))
    st.stop()

# Mapeamento de colunas
st.sidebar.subheader("Mapeamento de Colunas")
colmap = normalize_cols(df_raw)

def pick_col(label, candidates):
    opts = [colmap[c] for c in colmap if any(k in c for k in candidates)]
    default = opts[0] if opts else None
    return st.sidebar.selectbox(label, options=[None] + list(df_raw.columns), index=(0 if default is None else (list([None]+list(df_raw.columns))).index(default)))

col_estado = pick_col("Coluna de Estado (UF)", ["estado", "uf", "sigla"])
col_cidade = pick_col("Coluna de Cidade", ["cidade", "municipio"])
col_data   = pick_col("Coluna de Data", ["data", "emissao", "pedido", "nota", "competencia"])
col_valor  = pick_col("Coluna de Valor (R$)", ["valor", "total", "receita", "faturamento", "montante"])

if not all([col_estado, col_cidade, col_data, col_valor]):
    st.error("Mapeie as **quatro** colunas (Estado, Cidade, Data, Valor) na barra lateral.")
    st.stop()

# ---------- Preparação dos dados ----------
df = df_raw.copy()

# Normalizações
df["_estado"] = df[col_estado].astype(str).str.strip().str.upper()
df["_cidade"] = df[col_cidade].astype(str).str.strip().str.title()

df["_data"] = df[col_data].apply(coerce_date)
df["_mes"] = df["_data"].apply(month_le).astype("Int64")

df["_valor"] = df[col_valor].apply(to_numeric_safe)

# Filtros
st.sidebar.subheader("Filtros")
ufs = sorted([x for x in df["_estado"].dropna().unique().tolist() if x])
uf_sel = st.sidebar.multiselect("Estados (UF)", options=ufs, default=ufs)

# “até julho” = mês <= 7; também permito um ano opcional
anos = sorted(list({d.year for d in df["_data"].dropna()}))
ano_sel = st.sidebar.selectbox("Ano (opcional)", options=[None]+anos, index=0)
ate_julho = st.sidebar.checkbox("Considerar somente 'até julho' (mês ≤ 7)", value=True)

# Filtro por cidades (dependente de UF)
df_f = df[df["_estado"].isin(uf_sel)] if uf_sel else df.copy()
if ano_sel:
    df_f = df_f[df_f["_data"].apply(lambda d: (not pd.isna(d)) and d.year == ano_sel)]
if ate_julho:
    df_f = df_f[df_f["_mes"].apply(lambda m: (not pd.isna(m)) and int(m) <= 7)]

cidades_opts = sorted(df_f["_cidade"].dropna().unique().tolist())
cidades_sel = st.sidebar.multiselect("Cidades (opcional)", options=cidades_opts)

if cidades_sel:
    df_f = df_f[df_f["_cidade"].isin(cidades_sel)]

# ---------- Título ----------
st.title("📊 KPI de Vendas com Insights (OpenAI)")

# ---------- KPIs Globais ----------
total_valor = float(np.nansum(df_f["_valor"]))
qtd_reg = int(len(df_f))
qtd_cidades = df_f["_cidade"].nunique()
qtd_estados = df_f["_estado"].nunique()

c1, c2, c3, c4 = st.columns(4)
with c1: kpi_card("💰 Total (R$)", f"{total_valor:,.2f}".replace(",", "X").replace(".", ",").replace("X","."), "Soma do período filtrado")
with c2: kpi_card("🧾 Registros", f"{qtd_reg:,}".replace(",", "."))
with c3: kpi_card("🏙️ Cidades", f"{qtd_cidades:,}".replace(",", "."))
with c4: kpi_card("🗺️ Estados", f"{qtd_estados:,}".replace(",", "."))

st.divider()

# ---------- Evolução (linha por mês) ----------
df_evo = df_f.dropna(subset=["_data"]).copy()
if not df_evo.empty:
    df_evo["Ano-Mês"] = df_evo["_data"].apply(lambda d: f"{d.year}-{d.month:02d}")
    serie = df_evo.groupby("Ano-Mês", as_index=False)["_valor"].sum().sort_values("Ano-Mês")
    fig_evo = px.line(serie, x="Ano-Mês", y="_valor", title="Evolução mensal do valor (R$)")
    fig_evo.update_yaxes(tickformat=",")
    st.plotly_chart(fig_evo, use_container_width=True)
else:
    st.info("Sem datas válidas para plotar a evolução.")

st.divider()

# ---------- Rankings ----------
colA, colB = st.columns(2)

with colA:
    st.subheader("🏆 Top Cidades (geral)")
    top_cid = top_n(df_f, ["_cidade", "_estado"], "_valor", n=20)
    top_cid.rename(columns={"_cidade": "Cidade", "_estado": "UF", "_valor": "Valor (R$)"}, inplace=True)
    st.dataframe(top_cid, use_container_width=True, hide_index=True)
    fig_cid = px.bar(top_cid.head(10), x="Cidade", y="Valor (R$)", color="UF", title="Top 10 Cidades")
    fig_cid.update_yaxes(tickformat=",")
    st.plotly_chart(fig_cid, use_container_width=True)

with colB:
    st.subheader("🌎 Top Estados")
    top_uf = top_n(df_f, ["_estado"], "_valor", n=27)
    top_uf.rename(columns={"_estado": "UF", "_valor": "Valor (R$)"}, inplace=True)
    st.dataframe(top_uf, use_container_width=True, hide_index=True)
    fig_uf = px.bar(top_uf.head(10), x="UF", y="Valor (R$)", title="Top 10 Estados")
    fig_uf.update_yaxes(tickformat=",")
    st.plotly_chart(fig_uf, use_container_width=True)

st.divider()

# ---------- “Cidades que compraram até julho” ----------
st.subheader("📅 Cidades que compraram até julho")
df_jul = df_f.copy()
if not ate_julho:
    df_jul = df_jul[df_jul["_mes"].apply(lambda m: (not pd.isna(m)) and int(m) <= 7)]
compras_jul = (
    df_jul.groupby(["_cidade", "_estado"], as_index=False)["_valor"].sum()
    .rename(columns={"_cidade": "Cidade", "_estado": "UF", "_valor": "Valor (R$)"})
    .sort_values("Valor (R$)", ascending=False)
)
st.dataframe(compras_jul, use_container_width=True, hide_index=True)

st.divider()

# ---------- 200 cidades de MG ----------
st.subheader("🏅 Top 200 Cidades de Minas Gerais (MG)")
df_mg = df_f[df_f["_estado"] == "MG"].copy()
top_mg = (
    df_mg.groupby("_cidade", as_index=False)["_valor"].sum()
    .rename(columns={"_cidade": "Cidade", "_valor": "Valor (R$)"})
    .sort_values("Valor (R$)", ascending=False)
    .head(200)
)
st.dataframe(top_mg, use_container_width=True, hide_index=True)
if not top_mg.empty:
    fig_mg = px.bar(top_mg.head(25), x="Cidade", y="Valor (R$)", title="Top 25 MG")
    fig_mg.update_yaxes(tickformat=",")
    st.plotly_chart(fig_mg, use_container_width=True)
else:
    st.info("Não há dados de MG no recorte atual.")

st.divider()

# ---------- Insights gerados (locais + OpenAI opcional) ----------
st.subheader("🧠 Insights automáticos")

# Resumo para IA
def safe_top_rows(df, cols, n=15):
    if df is None or df.empty:
        return []
    return df[cols].head(n).to_dict(orient="records")

summary_payload = {
    "filtros": {
        "ufs": uf_sel,
        "cidades": cidades_sel,
        "ano": ano_sel,
        "ate_julho": bool(ate_julho),
    },
    "kpis": {
        "total_valor": total_valor,
        "qtd_registros": qtd_reg,
        "qtd_cidades": int(qtd_cidades),
        "qtd_estados": int(qtd_estados),
    },
    "top_cidades": safe_top_rows(top_cid, ["Cidade", "UF", "Valor (R$)"], n=15) if 'top_cid' in locals() else [],
    "top_estados": safe_top_rows(top_uf, ["UF", "Valor (R$)"], n=15) if 'top_uf' in locals() else [],
    "top_mg": safe_top_rows(top_mg, ["Cidade", "Valor (R$)"], n=30) if 'top_mg' in locals() else [],
}

# Insights locais obrigatórios (contêm as frases que você pediu)
local_text = build_local_insights(df_f)

# Tenta complementar com OpenAI (se disponível)
prompt = f"""
Você é um analista de dados. Com base no resumo JSON abaixo, escreva insights concisos e práticos
sobre performance de vendas, concentração geográfica, sazonalidade até julho, anomalias e
oportunidades de ação. Use bullet points curtos. Evite jargões e cite cidades/UFs quando relevante.

JSON:
{json.dumps(summary_payload, ensure_ascii=False)}
"""
with st.spinner("Gerando insights..."):
    ai_text = openai_insights(prompt)

# Mostra os locais e, se houver, os adicionais da IA
st.markdown(local_text)
if ai_text:
    st.markdown("\n\n### 🤖 Complemento do modelo\n" + ai_text)

st.divider()

# ---------- Salvar como “conversa” ----------
st.subheader("💾 Salvar KPI no histórico (conversa)")
title_default = f"KPI {datetime.now().strftime('%Y-%m-%d %H:%M')}"
title_txt = st.text_input("Título do snapshot", value=title_default)
if st.button("Salvar snapshot agora"):
    snapshot = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "title": title_txt,
        "filters": summary_payload["filtros"],
        "kpis": summary_payload["kpis"],
        "top_cidades": summary_payload["top_cidades"],
        "top_estados": summary_payload["top_estados"],
        "top_mg": summary_payload["top_mg"],
        "chat": [
            {"role": "user", "content": "Gerar KPI com filtros atuais e insights."},
            {"role": "assistant", "content": f"KPIs gerados. Total {format_currency(total_valor)} e {qtd_cidades} cidades no recorte."},
            {"role": "assistant", "content": local_text + ("\n\n### 🤖 Complemento do modelo\n" + ai_text if ai_text else "")},
        ],
    }
    save_history(snapshot)
    st.success("Snapshot salvo no histórico (formato conversa). Veja abaixo.")

# ---------- Histórico (visual estilo chat) ----------
st.subheader("🗂️ Histórico de KPIs (conversa)")
hist = load_history()
if not hist:
    st.caption("Ainda não há histórico salvo.")
else:
    for i, h in enumerate(reversed(hist), 1):
        with st.expander(f"{h.get('timestamp')} • {h.get('title','KPI salvo')}"):
            for msg in h.get("chat", []):
                with st.chat_message(msg.get("role", "assistant")):
                    st.markdown(msg.get("content", ""))
