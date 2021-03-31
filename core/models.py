from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django import forms

# models do return

class Acoes(models.Model):
    
    acao = models.CharField(max_length=100)
    previsao_data_ini = models.DateField()
    previsao_data_fim = models.DateField()
    

class Algos(models.Model):
    algo = models.CharField(max_length=100)   
    acao = models.ForeignKey(Acoes, on_delete=models.CASCADE)

    def __str__(self):
        return self.algo



