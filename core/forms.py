from django.forms import ModelForm
from .models import Acoes, Simulador, ConsultaBase, ConsultaResultado, FaleComigo
from django import forms
from datetime import date
from datetime import datetime

class AcoesForm(ModelForm):
    CHOICES = ('2021-01-01', '2021'),('2020-01-01', '2020'), ('2019-01-01', '2019'), ('2018-01-01', '2018'),('2017-01-01', '2017')
    field = forms.ChoiceField(choices=CHOICES, label='Entre com a data inicial para a previsão: ')
    
    class Meta:
        model = Acoes
        fields = ['acao']
        labels = {'acao': 'Entre com o papel:'}

        widgets = {
            'acao': forms.TextInput(attrs={ 'placeholder': 'Ex: PETR4'})
            
         
        }