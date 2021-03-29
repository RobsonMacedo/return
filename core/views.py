from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import Empregado, Cpf, Departamento, Telefone
from django_seed import Seed
from faker import Faker
from .models import Empregado, Cpf, Departamento, Telefone
import random
from .forms import AcoesForm
from django.http import HttpResponseRedirect

# bibliotecas para a conexão com o yahoofinance

import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime 
from datetime import timedelta
from datetime import date
from time import mktime
import ast

# bibliotecas do Django
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import AcoesForm
from .models import Acoes

# bibliotecas de machine learning
import re
import pickle
from datetime import date
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas_datareader as pdr


# trabalhar numa forma de fazer seed 
# Esse códi faz fake de nome e sobrenome
""" fake = Faker()
nomes = list()
sobrenomes = list()
for _ in range(3):
    nomes.append(fake.first_name())
    sobrenomes.append(fake.last_name())
print(nomes)
print(sobrenomes) """

def empregados(request, Cliente):
    return HttpResponse('lista de clientes')

def empregado_Robson(Cliente):
    empregado = Cliente
    empregado.nome = 'Robson'
    empregado.sobrenome = 'Macedo'
    empregado.save()
    return HttpResponse('empregado salvo com sucesso')

def empregado_detalhe(request, id):
    empregado = Empregado.objects.get(id=id)
    return HttpResponse('empregado: ' + empregado.nome)    

def empregado_por_nome(request, nome):
    return HttpResponse('empregado: ' + str(nome)) 

def resultados(request):
    acoesForm = AcoesForm(request.POST or None)
    data_passada = datetime.strptime(acoesForm['field'].data, '%Y-%m-%d').date()
    # pegando o histórico da ação
    start = acoesForm['field'].data
    algo = acoesForm['algo'].data
    end = datetime.now().strftime('%Y-%m-%d')
    ticker = str.upper(acoesForm['acao'].data) + '.SA'
    try:
        dados_acao = pdr.DataReader(ticker, 'yahoo', start, end)

        #tratando os dados
        
        ## criando coluna variacao
        lista = []
        for i in range(0, dados_acao.shape[0]):
            lista.append(dados_acao['Close'][i] - dados_acao['Open'][i])
        dados_acao['Variation'] = lista
        lista.clear()

        ## criando coluna situacao - Realidade
        for i in range(0, dados_acao.shape[0]):
            if (dados_acao['Variation'][i]>0):
                lista.append('compra')
            else:
                lista.append('venda')
        dados_acao['situation'] = lista
        lista.clear()

        ## médias móveis (mm9, mm21)
        ini = 0
        fim = 9
        for i in range(0, 9):
            lista.append(dados_acao.iloc[i, 3])
        for i in range(9, len(dados_acao.Close)):
            lista.append(dados_acao.iloc[ini:fim, 3].mean())
            ini += 1
            fim += 1
        dados_acao['mm9'] = lista
        lista.clear()
        
        ini = 0
        fim = 21
        for i in range(0, 21):
            lista.append(dados_acao.iloc[i, 3])
        for i in range(21, len(dados_acao.Close)):
            lista.append(dados_acao.iloc[ini:fim, 3].mean())
            ini += 1
            fim += 1
        dados_acao['mm21'] = lista
        lista.clear()

        ##bandas de bolingner (superior e inferior)
        lista = []
        lista.clear()
        ini = 0
        fim = 21
        for i in range(0, 21):
            lista.append(dados_acao.iloc[i, 3])
        for i in range(21, len(dados_acao.Close)):
            lista.append(dados_acao.iloc[ini:fim, 3].mean() +
                        2*(dados_acao.iloc[ini:fim, 3].std()))
            ini += 1
            fim += 1
        dados_acao['bb_sup'] = lista

        lista = []
        lista.clear()
        ini = 0
        fim = 21
        for i in range(0, 21):
            lista.append(dados_acao.iloc[i, 3])
        for i in range(21, len(dados_acao.Close)):
            lista.append(dados_acao.iloc[ini:fim, 3].mean() -
                        2*(dados_acao.iloc[ini:fim, 3].std()))
            ini += 1
            fim += 1
        dados_acao['bb_inf'] = lista
        lista.clear()

        ## Gerando padrão para cálculo do algortimo
        for i in range(0, len(dados_acao.Close)):
            if ((dados_acao.Close[i] > dados_acao.mm9[i]) and (dados_acao.Close[i] > dados_acao.mm21[i])):
                lista.append('compra')
            else:
                lista.append('venda')
        dados_acao['previsao inicial'] = lista

        ## calculando as previsoes
        ## preparando dados de treino e teste
        X = dados_acao.iloc[:, [8, 9, 10, 11]].values
        y = dados_acao.iloc[:, 12].values

        encoder = LabelEncoder()
        X[:, 0] = encoder.fit_transform(X[:, 0])
        X[:, 1] = encoder.fit_transform(X[:, 1])
        X[:, 2] = encoder.fit_transform(X[:, 2])
        X[:, 3] = encoder.fit_transform(X[:, 3])
        y = encoder.fit_transform(y)
        X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2)

        ##Pegando os valores para predição
        p1 = dados_acao.iloc[-9:, 3].mean()
        p2 = dados_acao.iloc[-21:, 3].mean()
        p3 = p2 + 2*(dados_acao.iloc[-21:, 3].std())
        p4 = p2 - 2*(dados_acao.iloc[-21:, 3].std())


        ## Treinando usando naive bayes
    
        for i in range(10):
            kfold = StratifiedKFold(
            n_splits=20, shuffle=True, random_state=i)
        resultados_naive = []
        matrizes = []
        for id_treino, id_teste in kfold.split(X, np.zeros(shape=(X.shape[0], 1))):
            classificador_naive = GaussianNB()
            classificador_naive.fit(X[id_treino], y[id_treino])
            predicao = classificador_naive.predict(X[id_teste])
            resultados_naive.append(
                accuracy_score(y[id_teste], predicao))

        media_naive = np.array(resultados_naive).mean()
        desvio_naive = np.array(resultados_naive).std()
        class_naive = classificador_naive.predict_proba([[p1, p2, p3, p4]])
        if class_naive[0][0] > class_naive[0][1]:
            class_naive_verbose = 'compra'
        else:
            class_naive_verbose = 'venda'
        

        ## Treinando usando arvores de decisao
    
        for i in range(10):
            kfold = StratifiedKFold(
                n_splits=20, shuffle=True, random_state=i)
            resultados_arvores = []
            matrizes = []
            for id_treino, id_teste in kfold.split(X, np.zeros(shape=(X.shape[0], 1))):
                classificador_arvores = DecisionTreeClassifier()
                classificador_arvores.fit(X[id_treino], y[id_treino])
                predicao = classificador_arvores.predict(X[id_teste])
                resultados_arvores.append(
                    accuracy_score(y[id_teste], predicao))

        media_arvores = np.array(resultados_arvores).mean()
        desvio_arvores = np.array(resultados_arvores).std()
        class_arvores = classificador_arvores.predict_proba([[p1, p2, p3, p4]])

        if class_arvores[0][0] > class_arvores[0][1]:
            class_arvores_verbose = 'compra'
        else:
            class_arvores_verbose = 'venda'

        ## Treinando usando random forest
        for i in range(10):
            kfold = StratifiedKFold(
                n_splits=20, shuffle=True, random_state=i)
            resultados_RFC = []
            matrizes = []
            for id_treino, id_teste in kfold.split(X, np.zeros(shape=(X.shape[0], 1))):
                classificador_RFC = RandomForestClassifier(n_estimators=5)
                classificador_RFC.fit(X[id_treino], y[id_treino])
                predicao = classificador_RFC.predict(X[id_teste])
                resultados_RFC.append(
                    accuracy_score(y[id_teste], predicao))

        media_random = np.array(resultados_RFC).mean()
        desvio_random = np.array(resultados_RFC).std()
        class_random = classificador_RFC.predict_proba([[p1, p2, p3, p4]])

        if class_random[0][0] > class_random[0][1]:
            class_random_verbose = 'compra'
        else:
            class_random_verbose = 'venda'

        # KNN
        for i in range(10):

            kfold = StratifiedKFold(
                n_splits=20, shuffle=True, random_state=i)
            resultados_KNN_classifier = []
            matrizes = []
            for id_treino, id_teste in kfold.split(X, np.zeros(shape=(X.shape[0], 1))):
                classificador_KNN = KNeighborsClassifier()
                classificador_KNN.fit(X[id_treino], y[id_treino])
                predicao = classificador_KNN.predict(X[id_teste])
                resultados_KNN_classifier.append(
                    accuracy_score(y[id_teste], predicao))

        media_knn = np.array(resultados_KNN_classifier).mean()
        desvio_knn = np.array(resultados_KNN_classifier).std()
        class_knn = classificador_KNN.predict_proba([[p1, p2, p3, p4]])

        if class_knn[0][0] > class_knn[0][1]:
            class_knn_verbose = 'compra'
        else:
            class_knn_verbose = 'venda'

        # SVC
        for i in range(5):

            kfold = StratifiedKFold(
                n_splits=20, shuffle=True, random_state=i)
            resultados_SVC = []
            matrizes = []
            for id_treino, id_teste in kfold.split(X, np.zeros(shape=(X.shape[0], 1))):
                classificador_SVC = SVC(gamma='auto', probability=True)
                classificador_SVC.fit(X[id_treino], y[id_treino])
                predicao = classificador_SVC.predict(X[id_teste])
                resultados_SVC.append(
                    accuracy_score(y[id_teste], predicao))

        media_svc = np.array(resultados_SVC).mean()
        desvio_svc = np.array(resultados_SVC).std()  
        class_svc = classificador_SVC.predict_proba([[p1, p2, p3, p4]]) 
    
        if class_svc[0][0] > class_svc[0][1]:
            class_svc_verbose = 'compra'
        else:
            class_svc_verbose = 'venda'



        context = { 'acoesForm':acoesForm, 
                    'start':start[0:4], 
                    'dados_acao': dados_acao, 
                    'ticker': ticker, 
                    'class_naive_verbose': class_naive_verbose if class_naive_verbose else 0,  
                    'class_arvores_verbose': class_arvores_verbose if class_naive_verbose else 0,
                    'class_random_verbose': class_random_verbose if class_random_verbose else 0,
                    'class_knn_verbose': class_knn_verbose if class_knn_verbose else 0,
                    'class_svc_verbose': class_svc_verbose if class_svc_verbose else 0,
                    'algo': algo}
        
        if acoesForm.is_valid():
            return render(request, 'resultados.html', context)
    except:
        
        return render(request, 'erro_na_busca.html', {'ticker': ticker})

def resultado_detalhe(request, algo, ticker):
    context = {'algo': algo, 'ticker': ticker}
    return render(request, 'resultado_detalhe.html',context)



