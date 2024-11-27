import streamlit as st
import joblib
import os 
from PIL import Image
from torchvision import transforms, models
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

classes = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'shark', 'starfish', 'stingray']

# Carregar modelos
model_dt = joblib.load(os.path.join('models', "decisionTree.pkl"))
model_rf = joblib.load(os.path.join('models', "randomForest.pkl"))
model_lr = joblib.load(os.path.join('models', "logisticRegression.pkl"))
model_knn = joblib.load(os.path.join('models', "KNN.pkl"))
model_svm = joblib.load(os.path.join('models', "SVM.pkl"))
modelos = {"Decision Tree": model_dt, "Random Forest": model_rf, "Logistic Regression": model_lr, "K-Nearest Neighbour": model_knn, "Support Vector Machine": model_svm}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

resnet.to(device)

# Título e menu de navegação
st.title("Classificação de espécies marinhas a partir de embeddings")
st.sidebar.title("Navegação")
selecao = st.sidebar.radio("Escolha a seção", ["Inferência", "Dataset", "Performance dos modelos"])

# Seção de Inferência
if selecao == "Inferência":
    
    st.header("Inferência")
    st.write("Escolha um modelo de classificação, visualize gráficos interativos e veja estatísticas descritivas das espécies.")
    
    # Classificação em lote com upload de CSV
    st.subheader("Classificação de espécies marinhas a partir de uma imagem")
    uploaded_file = st.file_uploader("Faça upload de um arquivo PNG, JPG ou JPEG com o animal a ser classificado", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        inference_image = Image.open(uploaded_file)
        st.image(inference_image, caption="Uploaded Image")
        
        # Fazer a previsão para cada linha do arquivo
        modelo_selecionado = st.selectbox("Escolha o modelo para classificação em lote", list(modelos.keys()))
        model = modelos[modelo_selecionado]
        st.write(f"Model: {model}")
        
        inference_image = inference_image.convert('RGB')
        inference_image = transform(inference_image)

        inference_image = inference_image.unsqueeze(0)
        inference_image = inference_image.to(device)
        
        embedding = resnet(inference_image)
        embedding = embedding.view(embedding.size(0), -1)
        embedding = embedding.cpu().detach().numpy()
        pred = model.predict(embedding)
        proba = model.predict_proba(embedding)
        
        st.write(f"Probabilidade das classes: {proba}")
        st.write(f"Classe prevista: {classes[pred[0]]}")

elif selecao == "Dataset":
    st.header("Dataset")
    st.markdown("O conjunto de dados utlizado foi o [Underwater Object Detection Dataset](https://www.kaggle.com/datasets/slavkoprytula/aquarium-data-cots/data), \
                disponível no Kaggle através do link ")
    st.write("A primeira etapa envolve a criação de um parser em Python para processar o arquivo YAML do dataset. O arquivo YAML contém as anotações sobre os bounding \
             boxes de cada objeto presente nas imagens, ou seja, as regiões de interesse (ROIs) que delimitam as espécies marinhas nas fotos subaquáticas. Esse parser \
             permite acessar essas anotações, facilitando o acesso às informações sobre os objetos presentes nas imagens.")
    st.write("Em seguida é realizada a extração dessas regiões de interesse. Para cada bounding box, uma subimagem é recortada e salva em uma pasta específica, onde o \
             nome da pasta corresponde ao rótulo da espécie. Dessa forma, as imagens das espécies são organizadas de forma a facilitar o treinamento posterior dos modelos.")
    st.write("Após a organização dos dados, um modelo pré-treinado de rede neural convolucional é carregado utilizando a biblioteca PyTorch. Modelos como ResNet50, VGG16, \
             EfficientNetB7 ou MobileNet são opções viáveis, devido ao seu alto desempenho em tarefas de visão computacional. Para a comparação dos modelos de classificação, \
             foi utilizado a ResNet50.")
    st.write("Para a tarefa de extração de características das ROIs, a camada de classificação do modelo pré-treinado é congelada, ou seja, não será mais treinada. \
             Em vez disso, utiliza-se apenas as camadas convolucionais para gerar as características (embeddings) que representam cada região de interesse. \
             Essas características são então extraídas e armazenadas.")
    st.write("Com as características extraídas, a próxima etapa envolve a criação da estrutura para treinar um novo modelo de aprendizado de máquina. As variáveis X e y \
             são definidas, sendo X o conjunto de características extraídas das ROIs e y os rótulos associados a cada ROI. Essas variáveis são utilizadas para treinar um \
             modelo de classificação.")
    st.write("Para o treinamento do modelo, são utilizados algoritmos clássicos de aprendizado de máquina, como Árvores de Decisão, Random Forest, Regressão Logística, \
             KNN e Support Vector Machines (SVM). O conjunto de dados é dividido, sendo 70\% para treinamento e 30\% para teste, garantindo uma avaliação equilibrada \
             do desempenho dos modelos.")
    
    class_counts = {
        'Fish': 2644,
        'Jellyfish': 693,
        'Penguin': 515,
        'Puffin': 283,
        'Shark': 344,
        'Starfish': 115,
        'Stingray': 183
    }

    fig, ax = plt.subplots()

    ax.bar(class_counts.keys(), class_counts.values(), color='skyblue')

    ax.set_title('Quantidade de amostras por classe')
    ax.set_xlabel('Classe')
    ax.set_ylabel('Número de amostras')

    st.pyplot(fig)
    st.write("Através do gráfico de distribuição das classes apresentado abaixo é possível observar o grande desbalanceamento do dataset à favor da classe Fish")

elif selecao == "Performance dos modelos":
    st.header("Performance dos modelos")
    index = [0, 1, 2, 3, 4, 5, 6, "accuracy", "macro avg", "weighted avg"]

    st.subheader("Árvore de Decisão")
    st.write("As Árvores de Decisão são algoritmos simples e intuitivos que dividem os dados com base em regras de decisão até \
             chegarem a uma classe ou valor predito. Elas são rápidas para treinar e executar, com complexidade de treinamento \
             de O(n log n) e de inferência de O(d), onde d é a profundidade da árvore. No entanto, podem sofrer com overfitting \
             e não se saem bem em dados complexos ou desbalanceados.")
    dt_report = {
        "precision": [0.78, 0.53, 0.49, 0.34, 0.44, 0.59, 0.51, None, 0.53, 0.65],
        "recall": [0.67, 0.80, 0.54, 0.46, 0.41, 0.59, 0.35, None, 0.54, 0.62],
        "f1-score": [0.72, 0.64, 0.51, 0.39, 0.42, 0.59, 0.41, 0.62, 0.53, 0.63],
        "support": [804, 205, 167, 80, 91, 37, 69, 1453, 1453, 1453]
    }    
    dt_table = pd.DataFrame(dt_report, index=index)
    st.table(dt_table)
    cm_dt = Image.open(os.path.join("cm", "dt.png"))
    st.image(cm_dt, caption="Matriz de confusão - Árvore de Decisão")

    st.subheader("Random Forest")
    st.write("O Random Forest é um algoritmo de aprendizado supervisionado que combina múltiplas árvores de decisão para melhorar \
             a robustez e reduzir o overfitting. Ele utiliza amostras aleatórias do conjunto de dados e subconjuntos de características \
             em cada divisão, criando diversidade entre as árvores e aumentando sua capacidade de generalização. Apesar de mais lento que \
             uma única árvore, seu tempo de treinamento é eficiente devido à construção paralela das árvores, com complexidade de \
             O(m * n * log n), onde m é o número de árvores e n o número de amostras.")
    rf_report = {
        "precision": [0.99, 0.51, 0.42, 0.35, 0.32, 0.35, 0.34, None, 0.47, 0.86],
        "recall": [0.61, 0.99, 0.94, 0.83, 1.00, 1.00, 1.00, None, 0.91, 0.69],
        "f1-score": [0.76, 0.67, 0.58, 0.49, 0.48, 0.52, 0.51, 0.69, 0.57, 0.72],
        "support": [1110, 158, 83, 46, 27, 13, 16, 1453, 1453, 1453]
    }
    rf_table = pd.DataFrame(rf_report, index=index)
    st.table(rf_table)
    cm_rf = Image.open(os.path.join("cm", "rf.png"))
    st.image(cm_rf, caption="Matriz de confusão - Random Forest")

    st.subheader("Regressor Logístico")
    st.write("A Regressão Logística é um modelo supervisionado usado para classificação, que calcula probabilidades com base em \
             uma combinação linear das características de entrada transformada pela função logística ou softmax. É eficiente para \
             conjuntos de dados moderados, com uma complexidade de treinamento de O(n * p) e inferência rápida, proporcional ao número \
             de características.")
    lr_report = {
        "precision": [0.90, 0.80, 0.89, 0.72, 0.73, 0.73, 0.87, None, 0.81, 0.85],
        "recall": [0.89, 0.95, 0.82, 0.72, 0.63, 0.75, 0.71, None, 0.78, 0.85],
        "f1-score": [0.89, 0.87, 0.85, 0.72, 0.68, 0.74, 0.78, 0.85, 0.79, 0.85],
        "support": [691, 260, 202, 108, 98, 36, 58, 1453, 1453, 1453]
    }
    lr_table = pd.DataFrame(lr_report, index=index)
    st.table(lr_table)
    cm_lr = Image.open(os.path.join("cm", "lr.png"))
    st.image(cm_lr, caption="Matriz de confusão - Regressor Logístico")

    st.subheader("K-Nearest Neighbors")
    st.write("O algoritmo K-Nearest Neighbors (KNN) classifica amostras com base na proximidade entre pontos no espaço das características, \
             considerando os k vizinhos mais próximos no conjunto de treinamento e atribuindo a classe mais comum. Apesar de não exigir \
             treinamento, o tempo de inferência do KNN pode ser elevado, com complexidade de O(n * d), devido ao cálculo de distâncias para \
             todas as amostras do conjunto de dados.")
    knn_report = {
        "precision": [0.96, 0.70, 0.75, 0.59, 0.62, 0.51, 0.72, None, 0.69, 0.84],
        "recall": [0.77, 0.99, 0.79, 0.75, 0.85, 0.95, 0.83, None, 0.85, 0.81],
        "f1-score": [0.85, 0.82, 0.77, 0.66, 0.72, 0.67, 0.77, 0.81, 0.75, 0.82],
        "support": [851, 220, 174, 85, 62, 20, 41, 1453, 1453, 1453]
    }
    knn_table = pd.DataFrame(knn_report, index=index)
    st.table(knn_table)
    cm_knn = Image.open(os.path.join("cm", "knn.png"))
    st.image(cm_knn, caption="Matriz de confusão - KNN")

    st.subheader("SVM")
    st.write("A Máquina de Vetores de Suporte (SVM) é um modelo supervisionado que busca separar classes no espaço de \
             características por meio de um hiperplano que maximiza a margem entre os exemplos mais próximos \
             (vetores de suporte). Para dados não linearmente separáveis, utiliza funções kernel, como RBF, que \
             projetam os dados em um espaço de maior dimensionalidade. Apesar de eficiente na inferência, com complexidade \
             linear em relação aos vetores de suporte, o treinamento pode ser custoso, com complexidade de O(n² * d).")
    svm_report = {
        "precision": [0.81, 0.81, 0.92, 0.75, 0.86, 0.78, 0.87, None, 0.83, 0.83],
        "recall": [0.92, 0.94, 0.74, 0.66, 0.52, 0.85, 0.69, None, 0.76, 0.82],
        "f1-score": [0.86, 0.87, 0.82, 0.70, 0.65, 0.82, 0.77, 0.82, 0.78, 0.82],
        "support": [602, 267, 229, 122, 140, 34, 59, 1453, 1453, 1453]
    }
    svm_table = pd.DataFrame(svm_report, index=index)
    st.table(svm_table)
    cm_svm = Image.open(os.path.join("cm", "svm.png"))
    st.image(cm_svm, caption="Matriz de confusão - SVM")