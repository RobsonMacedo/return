from django.http import HttpResponse
from django.shortcuts import render
from core.models import Acoes, Algos
from core.forms import AcoesForm

def home(request):
    form = AcoesForm()
    return render(request, 'index.html', {'form':form})

   
