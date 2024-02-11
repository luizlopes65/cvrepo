import streamlit as st
import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


st.markdown("<h1 style='text-align: center;'>Projetos</h1>", unsafe_allow_html=True)

with st.expander('MNIST-Classifier'):
    st.write('O dataset MNIST consistem em imagens de dígitos manuscritos de 0 a 9. O objetivo consiste em criar um modelo que consiga identificar o dígito a partir da imagem. Cada imagem é formada por 728 pixels (28x28) e possui seu respectivo label.')
    st.write('A minha abordagem do problema foi criar um modelo de rede neural convolucional 2D, utilizando Tensorflow com aceleração da GPU, obtendo até 0.98 de acurácia;')
    st.write('Abaixo você pode testar e observar a performace do modelo:')
    model = tf.keras.models.load_model(os.path.join('MNIST_model.keras'))
    numberchosen = st.select_slider('Número a prever:', [0,1,2,3,4,5,6,7,8,9])
    train_df = pd.read_csv(os.path.join('mnist_test.csv'))
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

    print(y_pred[escolha])

    st.write('O número predito pelo modelo foi', np.argmax(y_pred[escolha]))



with st.expander('Who survives in Titanic?'):
    st.write('projeto')

with st.expander('YourSetlist'):
    st.write('projeto')

with st.expander('Simulação de Ambiente Celular'):
    st.write('projeto')
