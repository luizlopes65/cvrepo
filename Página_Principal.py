import streamlit as st
import numpy as np
import pandas as pd
import os
from PIL import Image
from datetime import datetime

idade = int(datetime.now().year) - 2004
path = 'c:/Users/Admin/OneDrive/Desktop/ProjetosPython/PycharmProjects/Personal/ScriptsPy/Streamlit/CV/'

image = Image.open(os.path.join(path, 'foto.jpeg'))


st.markdown("<h1 style='text-align: center;'>Currículo/Portfólio</h1>", unsafe_allow_html=True)
st.subheader(f'Luiz Eduardo Foschiera Couto Lopes, {idade} anos')


col1, col2 = st.columns([1, 2])


with col1:
    st.image(image, width=180)

with col2:
    st.write('- Olá, meu nome é Luiz Eduardo Lopes. Sou estudade de Matemática Aplicada e Computacional na Unicamp e tenho paixão por desenvolvimento de algoritmos, machine learning, ciência de dados e modelagem matemática.')
    st.write('- Possuo experiência em programação em python, elaboração, teste e otimização de modelos de aprendizado de máquina, ciência de dados e construção de algoritmos.')
    st.write('- Sou fluente em inglês (certificado ECCE e FCE) e possuo francês funcional.')
    st.write('- Além disso, possuo experiência em pesquisa no CNPEM -LNBR (Centro Nacional de Pesquisa em Energia e Materiais - Laboratório Nacional de Biorrenováveis), onde explorei a utilização de métodos de aprendizado de máquina para a predição da energia de ativação de reações químicas de lignina. Construí familiaridade, também, com cargos de liderança e gestão, como gestão do grupo de estudos em IA e Data Science Iris Unicamp e vice-presidência da Bateria PercUrsão Unicamp.')
    st.write('- Criei esse site no intuito de apresentar meu currículo e portifólio de forma customizada e interativa. Buscando desafiar minhas habilidades em Python e explorar novos caminhos com o Streamlit.') 

st.markdown("<h2 style='text-align: center;'>Para navegar, basta checar a barra lateral desta página e escolher para onde quer ir.</h2>", unsafe_allow_html=True)

st.write('\n\n',datetime.now().date().strftime("%d/%m/%Y"))
