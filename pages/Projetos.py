import streamlit as st
import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = 'c:/Users/Admin/OneDrive/Desktop/ProjetosPython/PycharmProjects/Personal/ScriptsPy/Streamlit/CV/'


st.markdown("<h1 style='text-align: center;'>Projetos</h1>", unsafe_allow_html=True)

with st.expander('MNIST-Classifier'):
    st.write('O dataset MNIST consistem em imagens de dígitos manuscritos de 0 a 9. O objetivo consiste em criar um modelo que consiga identificar o dígito a partir da imagem. Cada imagem é formada por 728 pixels (28x28) e possui seu respectivo label.')
    st.write('A minha abordagem do problema foi criar um modelo de rede neural convolucional 2D, utilizando Tensorflow com aceleração da GPU, obtendo até 0.98 de acurácia;')
    st.write('Abaixo você pode testar e observar a performace do modelo:')
    model = tf.keras.models.load_model(os.path.join('pages','MNIST_model.keras'))
    numberchosen = st.select_slider('Número a prever:', [0,1,2,3,4,5,6,7,8,9])
    train_df = pd.read_csv(os.path.join('pages','mnist_test.csv'))
    labels = train_df['label']
    labels = labels[labels == numberchosen]
    escolha = np.random.choice(labels.index)
    train_df = train_df.drop(columns=['label'])
    train_df = train_df.apply(lambda x: x/255)
    st.image(np.array(train_df.iloc[escolha]).reshape(28,28), caption='Imagem do número escolhido', use_column_width=True)
    dados_treino = []
    for i in range(len(train_df)):
        dados_treino.append(list(train_df.iloc[i]))
    dados_treino = np.expand_dims(dados_treino, axis=1)
    dados_treino_im = []
    for i in range(len(dados_treino)):
        dados_treino_im.append(dados_treino[i].reshape((28,28)))

    dados_treino_im = np.array(dados_treino_im).reshape(-1, 28, 28, 1)

    y_pred = model.predict(dados_treino_im)

    st.write('O número predito pelo modelo foi', np.argmax(y_pred[escolha]))


with st.expander('YourSetlist'):
    st.write('Este projeto foge um pouco da linha da qual estou acostumado a desenvolver. A ideia era criar um aplicativo voltado para bandas/showmen, onde você e seus amigos (banda/equipe) podem construir o setlist (lista de músicas a serem tocadas) de seu show juntos, e terem acesso a ele de forma prática.')
    st.write('O principal desafio era me adequar aos modos usuais de construção de aplicativos, isto é, programação em javascript, C++, HTML ou React, entre outras. Apesar de ter vontade de aprendê-las no futuro, gostaria de aplicar a ideia o maids rápido possível.')
    st.write('Nesse espírito conheci o ADALO, plataforma de desenvolvimento de apps sem a necessidade de programar, somente utilizando lógica e relações entre bases de dados. E foi nele que desenvolvi meu aplicativo. Depois de quase 3 meses de trabalho, soltei para alguns amigos e familiares, para ter contato com necessidade , ideias e bugs. No momento, o app ainda está em construção, mas deixo abaixo o link para a versão atual (Note: a versão abaixo corresponde a versão do dia 19/02/2024):')
    st.markdown('<a href="https://luiz-eduardo-foschiera-couto-lopess-team.adalo.com/yoursetlist" target="_blank">YourSetlist</a>', unsafe_allow_html=True)
