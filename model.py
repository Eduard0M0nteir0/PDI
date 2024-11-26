import streamlit as st
import joblib
import os 
from PIL import Image
from torchvision import transforms, models
import torch
import torch.nn as nn


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
st.title("Dashboard de Classificação de Iris")
st.sidebar.title("Navegação")
selecao = st.sidebar.radio("Escolha a seção", ["Visão Geral", "Gráficos Interativos", "Estatísticas Descritivas"])

# Seção de Visão Geral
if selecao == "Visão Geral":
    st.header("Visão Geral")
    st.write("Este dashboard permite explorar diferentes aspectos do conjunto de dados Iris.")
    st.write("Escolha um modelo de classificação, visualize gráficos interativos e veja estatísticas descritivas das espécies.")
    
    # Classificação em lote com upload de CSV
    st.subheader("Classificação em Lote com Arquivo CSV")
    uploaded_file = st.file_uploader("Faça upload de um arquivo PNG, JPG ou JPEG com o animal a ser classificado", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Fazer a previsão para cada linha do arquivo
        modelo_selecionado = st.selectbox("Escolha o modelo para classificação em lote", list(modelos.keys()))
        model = modelos[modelo_selecionado]
        
        image = image.convert('RGB')
        image = transform(image)

        image = image.unsqueeze(0)
        image = image.to(device)
        
        embedding = model(image)
        embedding = embedding.view(embedding.size(0), -1)
        embedding = embedding.cpu().detach().numpy()
        pred = model_dt.predict(embedding)

        st.write(f"Resultados das previsões: {classes[pred]}")

# # Seção de Gráficos Interativos
# elif selecao == "Gráficos Interativos":
#     st.header("Gráficos Interativos")
#     plot_tipo = st.selectbox("Escolha o tipo de gráfico", ["Scatter Plot", "Gráfico de Barras", "Histograma", "Boxplot"])

#     if plot_tipo == "Scatter Plot":
#         # Scatter plot entre comprimento e largura das pétalas
#         st.write("Scatter Plot: Comprimento vs. Largura da Pétala")
#         fig, ax = plt.subplots()
#         sns.scatterplot(data=iris_df, x="petal length (cm)", y="petal width (cm)", hue="species", ax=ax)
#         st.pyplot(fig)

#     elif plot_tipo == "Gráfico de Barras":
#         # Gráfico de barras da quantidade de cada espécie
#         st.write("Gráfico de Barras: Quantidade de cada Espécie")
#         fig, ax = plt.subplots()
#         iris_df['species'].value_counts().plot(kind="bar", color=['blue', 'green', 'red'], ax=ax)
#         ax.set_ylabel("Quantidade")
#         ax.set_title("Distribuição das Espécies")
#         st.pyplot(fig)
    
#     elif plot_tipo == "Histograma":
#         # Histograma do comprimento da sépala
#         st.write("Histograma: Comprimento da Sépala")
#         fig, ax = plt.subplots()
#         sns.histplot(iris_df['sepal length (cm)'], kde=True, ax=ax, color='purple')
#         ax.set_title("Distribuição do Comprimento da Sépala")
#         st.pyplot(fig)
    
#     elif plot_tipo == "Boxplot":
#         # Boxplot do comprimento da sépala por espécie
#         st.write("Boxplot: Comprimento da Sépala por Espécie")
#         fig, ax = plt.subplots()
#         sns.boxplot(data=iris_df, x="species", y="sepal length (cm)", palette="Set2", ax=ax)
#         ax.set_title("Comprimento da Sépala por Espécie")
#         st.pyplot(fig)

# # Seção de Estatísticas Descritivas
# elif selecao == "Estatísticas Descritivas":
#     st.header("Estatísticas Descritivas")
#     especie_selecionada = st.selectbox("Escolha a espécie para visualização", ["setosa", "versicolor", "virginica"])

#     # Filtrar o dataset para a espécie selecionada
#     especie_df = iris_df[iris_df['species'] == especie_selecionada]

#     # Cálculo de estatísticas descritivas
#     media_sepal_length = especie_df['sepal length (cm)'].mean()
#     media_petal_length = especie_df['petal length (cm)'].mean()
#     desvio_sepal_length = especie_df['sepal length (cm)'].std()
#     desvio_petal_length = especie_df['petal length (cm)'].std()

#     st.subheader(f"Métricas da Espécie: {especie_selecionada.capitalize()}")
#     st.write(f"Média do Comprimento da Sépala: {media_sepal_length:.2f} cm")
#     st.write(f"Desvio Padrão do Comprimento da Sépala: {desvio_sepal_length:.2f} cm")
#     st.write(f"Média do Comprimento da Pétala: {media_petal_length:.2f} cm")
#     st.write(f"Desvio Padrão do Comprimento da Pétala: {desvio_petal_length:.2f} cm")

#     st.write("Espécies disponíveis: Setosa, Versicolor, Virginica.")
#     st.write("A navegação acima permite comparar e explorar diferentes espécies de maneira interativa.")