import streamlit as st
import pandas as pd
from utils.ai_agent import buscar_info_localizacao
import unicodedata

st.title("🏛️ Busca de Informações de Empresas Públicas")

uploaded_file = st.file_uploader("Envie a planilha com os nomes das empresas (opcionalmente com colunas Cidade e Estado)", type=["csv", "xlsx"])

def _norm(s):
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s

st.markdown("""
**Como funciona**
- Lê as colunas `Empresa` (obrigatória) e, se existirem, `Cidade` e `Estado` (opcionais).
- Se `Cidade` estiver vazia, **preencho com o nome do Estado** dessa linha.
- A busca usa uma consulta combinando: `Empresa, Cidade, Estado` (somente campos não vazios).
- Você pode clicar em **Buscar** quantas vezes quiser; o processo é idempotente.
""")

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Mapear colunas por nome aproximado
    cols_lower = {c.lower().strip(): c for c in df.columns}
    col_empresa = cols_lower.get("empresa") or next((c for c in df.columns if "empresa" in c.lower()), None)
    col_cidade  = cols_lower.get("cidade")  or next((c for c in df.columns if "cidade"  in c.lower()), None)
    col_estado  = cols_lower.get("estado")  or next((c for c in df.columns if "estado"  in c.lower()), None)

    st.subheader("Prévia dos Dados")
    st.dataframe(df.head())

    if col_empresa is None:
        st.error("A planilha precisa conter a coluna: Empresa")
    else:
        if st.button("🔎 Buscar informações (precisão cidade/estado)"):
            cidade_vals = df[col_cidade] if col_cidade else pd.Series([""] * len(df))
            estado_vals = df[col_estado] if col_estado else pd.Series([""] * len(df))

            cidade_vals = cidade_vals.fillna("")
            estado_vals = estado_vals.fillna("")

            # Preenche cidade vazia com o nome do estado da própria linha
            cidade_preenchida = []
            estado_preenchida = []
            for cid, est in zip(cidade_vals, estado_vals):
                cid_s = str(cid).strip()
                est_s = str(est).strip()
                if cid_s == "" and est_s != "":
                    cid_s = est_s
                cidade_preenchida.append(cid_s)
                estado_preenchida.append(est_s)

            cidade_final = pd.Series(cidade_preenchida)
            estado_final = pd.Series(estado_preenchida)

            resultados = []
            for idx, row in df.iterrows():
                emp = str(row[col_empresa]).strip()
                cid = cidade_final.iloc[idx] if idx < len(cidade_final) else ""
                est = estado_final.iloc[idx] if idx < len(estado_final) else ""

                partes = [emp]
                if cid:
                    partes.append(cid)
                if est:
                    partes.append(est)
                consulta = ", ".join([p for p in partes if p])

                info = buscar_info_localizacao(consulta)
                if isinstance(info, dict):
                    info_out = dict(info)
                else:
                    info_out = {"Resultado": info}

                # Guardar o que foi usado para consulta
                info_out["Empresa_Origem"] = emp
                info_out["Cidade_Usada"] = cid
                info_out["Estado_Usado"] = est

                resultados.append(info_out)

            resultados_df = pd.DataFrame(resultados)
            st.subheader("Resultados Encontrados")
            st.dataframe(resultados_df)

            csv = resultados_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Baixar Resultados em CSV", csv, "resultados_empresas.csv", "text/csv")
else:
    st.info("Carregue uma planilha para iniciar.")



# ==============================
# 🔎 Extração de dados a partir de PDFs (NFS-e)
# ==============================
import io
import pdfplumber
import re
import pandas as pd

st.markdown("---")
st.header("📄 Ler vários PDFs (NFS-e) e montar planilha")

pdfs = st.file_uploader("Envie 1 ou mais PDFs da NFS-e", type=["pdf"], accept_multiple_files=True)

UF_LIST = [
    "AC","AL","AM","AP","BA","CE","DF","ES","GO","MA","MT","MS","MG","PA","PB","PR","PE",
    "PI","RJ","RN","RO","RR","RS","SC","SE","SP","TO"
]
UF_RE = r"\\b(" + "|".join(UF_LIST) + r")\\b"

def _to_text(f):
    try:
        with pdfplumber.open(io.BytesIO(f.read())) as pdf:
            pages_text = []
            for p in pdf.pages:
                t = p.extract_text() or ""
                pages_text.append(t)
            return "\\n".join(pages_text)
    except Exception as e:
        return ""

def _find_emitida_em(text):
    # Ex: "Emitida em: 25/07/2025 às 09:13:26"
    m = re.search(r"Emitida\\s*em\\s*:?\\s*(\\d{2}/\\d{2}/\\d{4})", text, flags=re.I)
    return m.group(1) if m else None

def _find_valor_servicos(text):
    # Ex: "Valor dos serviços: R$ 1.390,00"
    m = re.search(r"Valor\\s+dos\\s+servi[cç]os\\s*:?\\s*R\\$\\s*([\\d\\.]+,\\d{2})", text, flags=re.I)
    return m.group(1) if m else None

def _find_cidade_estado(text):
    # Heurística:
    # 1) Pegamos linhas para achar uma linha com UF (duas letras) e cidade à esquerda
    linhas = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for idx, ln in enumerate(linhas):
        m = re.search(UF_RE, ln)
        if m:
            uf = m.group(1)
            # cidade é o conteúdo à esquerda do UF na mesma linha, limpando pontuação residual
            before = ln[:m.start()].strip()
            # remover rótulos como "Email:" "Telefone:" se vierem na mesma linha
            before = re.sub(r"(?i)(email|telefone)\\s*:\\s*.*$", "", before).strip()
            # às vezes vem algo como 'Tiradentes' apenas
            # se vazio, tenta linha anterior
            cidade = before if before else (linhas[idx-1].strip() if idx > 0 else None)
            # limpar lixo comum (CEP, números)
            cidade = re.sub(r"(?i)cep\\s*:\\s*\\d{5}-?\\d{3}", "", cidade or "").strip()
            cidade = re.sub(r"\\b\\d{2,}[\\w\\.-]*\\b", "", cidade).strip(",-; ")
            # não devolver vazio
            if cidade:
                return cidade, uf
    return None, None

if pdfs:
    resultados = []
    for f in pdfs:
        # Precisamos ler duas vezes: uma para o pdfplumber (que consome stream)
        raw = f.read()
        text = ""
        try:
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                pages_text = []
                for p in pdf.pages:
                    t = p.extract_text() or ""
                    pages_text.append(t)
                text = "\\n".join(pages_text)
        except Exception:
            text = ""

        emitida = _find_emitida_em(text)
        valor = _find_valor_servicos(text)
        cidade, uf = _find_cidade_estado(text)

        resultados.append({
            "Arquivo": f.name,
            "Emitida_em": emitida,
            "Valor_dos_servicos": valor,
            "Cidade": cidade,
            "Estado": uf
        })

    df_pdf = pd.DataFrame(resultados)
    st.subheader("Prévia da planilha extraída dos PDFs")
    st.dataframe(df_pdf)

    # Downloads
    csv_bytes = df_pdf.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Baixar CSV (PDFs)", data=csv_bytes, file_name="dados_nfse.csv", mime="text/csv")

    try:
        import io as _io
        bio = _io.BytesIO()
        # Para XLSX, usar engine openpyxl
        df_pdf.to_excel(bio, index=False, engine="openpyxl")
        st.download_button("⬇️ Baixar XLSX (PDFs)", data=bio.getvalue(), file_name="dados_nfse.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.info("Para exportar XLSX, instale 'openpyxl' no requirements.")
