import streamlit as st
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

st.set_page_config(page_title="Agente de KPI e Buscador", layout="wide")

st.title("📊 Agente Inteligente - KPI & Busca de Informações")
st.write("Selecione a página no menu lateral para usar o agente.")
