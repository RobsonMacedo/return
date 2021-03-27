from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import Empregado, Cpf, Departamento, Telefone
from django_seed import Seed
from faker import Faker
from .models import Empregado, Cpf, Departamento, Telefone
import random
from .forms import AcoesForm

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
    end = datetime.now().strftime('%Y-%m-%d')
    ticker = str.upper(acoesForm['acao'].data) + '.SA'
    dados_acao = pdr.DataReader(ticker, 'yahoo', start, end)

    #tratando os dados
    
    ## criando coluna variacao
    lista = []
    for i in range(0, dados_acao.shape[0]):
        lista.append(dados_acao['Close'][i] - dados_acao['Open'][i])
    dados_acao['Variation'] = lista
    lista.clear()

    ## criando coluna situacao - Realidade
    lista = []
    for i in range(0, dados_acao.shape[0]):
        if (dados_acao['Variation'][i]>0):
            lista.append('compra')
        else:
            lista.append('venda')
    dados_acao['situation'] = lista
    lista.clear()

    ## médias móveis (mm9, mm21)

    lista = []
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
    
    lista = []
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

    

    print(dados_acao.shape[0])
    print(dados_acao.Variation)
    if acoesForm.is_valid():
        return render(request, 'resultados.html', {'acoesForm':acoesForm, 'dados_acao': dados_acao, 'ticker': ticker})

# Métodos para pegar o histórico da ação

