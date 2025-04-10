import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from streamlit_lottie import st_lottie
import json

# --- Layout e tema ---
st.set_page_config(page_title="✈️ Aviator Bot", layout="wide")

# Função para carregar animação Lottie
def load_lottie(path):
    with open(path, "r") as f:
        return json.load(f)

# Carregar animação do avião
lottie = load_lottie("aviator_lottie.json")

# --- Título e animação ---
st.markdown("<h1 style='color:#FFD700;'>✈️ Aviator Bot – Previsão de Rodadas</h1>", unsafe_allow_html=True)
st_lottie(lottie, height=150, key="aviator")

# --- Upload de dados ---
st.subheader("📂 Envie seu arquivo CSV com rodadas:")
arquivo = st.file_uploader("Escolha um arquivo .csv", type="csv")

historico = []

if arquivo:
    df = pd.read_csv(arquivo)

    # Exibir dados
    st.markdown("### 🧾 Visualização dos dados")
    st.dataframe(df)

    # --- Treinamento do modelo ---
    df = df.dropna()
    X = df[["Multiplicador", "Valor Apostado"]]
    y = df["Resultado"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = RandomForestClassifier()
    model.fit(X, y_encoded)

    # --- Formulário de nova previsão ---
    st.markdown("### 🔮 Simule uma nova rodada")

    col1, col2 = st.columns(2)
    with col1:
        mult = st.slider("Multiplicador previsto", 1.0, 10.0, 2.0)
    with col2:
        aposta = st.number_input("Valor apostado", value=10.0)

    if st.button("💥 Prever resultado"):
        pred = model.predict([[mult, aposta]])
        resultado = le.inverse_transform(pred)[0]

        st.success(f"✈️ Resultado previsto: **{resultado}**")

        # Mensagens divertidas
        if resultado == "Não Caído":
            st.balloons()
            st.info("O avião alçou voo! ✈️💰")
        else:
            st.warning("O avião caiu... tente outra rodada! ")

        # Guardar no histórico
        historico.append({"Multiplicador": mult, "Aposta": aposta, "Resultado": resultado})

    # --- Histórico de previsões ---
    if historico:
        st.markdown("### 📜 Histórico de Previsões")
        st.dataframe(pd.DataFrame(historico))

    # --- Gráfico ao vivo ---
    st.markdown("### 📊 Distribuição dos multiplicadores")
    fig = px.histogram(df, x="Multiplicador", nbins=20)
    st.plotly_chart(fig)
  
