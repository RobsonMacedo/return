from django.forms import ModelForm
from .models import Acoes, Simulador, ConsultaBase, ConsultaResultado, FaleComigo
from django import forms
from datetime import date
from datetime import datetime

class AcoesForm(ModelForm):
    CHOICES_FIELD = ('2021-01-01', '2021'),('2020-01-01', '2020'), ('2019-01-01', '2019'), ('2018-01-01', '2018'),('2017-01-01', '2017')
    CHOICES_ALGO = ('all', 'TODOS'), ('Naive Bayes', 'Naive Bayes'),('arvores', 'Árvores de Decisão'), ('random', 'Random Forest'), ('knn', 'KNN'), ('svc', 'SVC')
    
    field = forms.ChoiceField(choices=CHOICES_FIELD, label='Entre com a data inicial para a previsão: ')
    algo = forms.ChoiceField(choices=CHOICES_ALGO, label='Selecione o algoritmo: ')
    

    
    class Meta:
        model = Acoes
        fields = ['acao']
        labels = {'acao': 'Entre com o papel:'}

        widgets = {
            'acao': forms.TextInput(attrs={ 'placeholder': 'Ex: PETR4'})
            
         
        }