import os
import openai
import pandas as pd

openai.api_key = os.getenv("OPENAI_API_KEY")

def gerar_insights(df: pd.DataFrame) -> str:
    prompt = f"""Você é um analista de dados. Baseado no seguinte dataframe de vendas, gere insights de compras,
    cidades que mais compram, estados relevantes e possíveis oportunidades.
    Dados:
    {df.head(50).to_string()}
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-5.0-nano",
            messages=[{"role": "system", "content": "Você é um analista de dados."},
                      {"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"Erro ao gerar insights: {e}"



import requests
import pandas as pd

def buscar_info_localizacao(empresa: str) -> dict:
    """
    Busca informações de localização (cidade, estado, mesorregião, microrregião)
    a partir do nome da empresa (normalmente prefeitura ou câmara municipal).
    """
    try:
        # Extrair o nome do município do texto
        palavras = empresa.split()
        municipio = None
        for i, p in enumerate(palavras):
            if p.upper() in ["DE", "DO", "DA", "DOS", "DAS"] and i+1 < len(palavras):
                municipio = " ".join(palavras[i+1:]).title()
                break

        if not municipio:
            return {"Empresa": empresa, "Cidade": None, "Estado": None, "Mesorregião": None, "Microrregião": None}

        # Buscar no IBGE
        url = f"https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
        resp = requests.get(url)
        dados = resp.json()

        info = next((d for d in dados if d["nome"].lower() == municipio.lower()), None)

        if not info:
            return {"Empresa": empresa, "Cidade": municipio, "Estado": None, "Mesorregião": None, "Microrregião": None}

        cidade = info["nome"]
        estado = info["microrregiao"]["mesorregiao"]["UF"]["nome"]
        meso = info["microrregiao"]["mesorregiao"]["nome"]
        micro = info["microrregiao"]["nome"]

        return {
            "Empresa": empresa,
            "Cidade": cidade,
            "Estado": estado,
            "Mesorregião": meso,
            "Microrregião": micro
        }

    except Exception as e:
        return {"Empresa": empresa, "Cidade": None, "Estado": None, "Mesorregião": None, "Microrregião": None, "Erro": str(e)}
