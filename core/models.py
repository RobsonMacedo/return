from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django import forms

# models do return

class Acoes(models.Model):
    
    acao = models.CharField(max_length=10000)
    previsao_data_ini = models.DateField() # código acertado / select colocado

    
class Simulador(models.Model):
    acao = models.CharField(max_length=10)
    data_inicio = models.DateField(auto_now=False, auto_now_add=False)
    data_final = models.DateField(auto_now=False, auto_now_add=False)    
    valor = models.FloatField()
    

class Resultados(models.Model):
    data = models.DateField(auto_now=False, auto_now_add=False, blank=True, null=True)
    nome = models.CharField(max_length=50, blank=True, null=True)
    media_naive = models.FloatField()
    desvio_naive = models.FloatField()
    class_naive = models.CharField(max_length=50, blank=True, null=True)
    predict_class_naive = models.CharField(max_length=50, blank=True, null=True)
    media_arvores = models.FloatField()
    desvio_arvores = models.FloatField()
    class_arvores = models.CharField(max_length=50, blank=True, null=True)
    predict_class_arvores = models.CharField(max_length=50, blank=True, null=True)
    media_random = models.FloatField()
    desvio_random = models.FloatField()
    class_random = models.CharField(max_length=50, blank=True, null=True)
    predict_class_random = models.CharField(max_length=50, blank=True, null=True)
    media_knn = models.FloatField()
    desvio_knn = models.FloatField()
    class_knn = models.CharField(max_length=50, blank=True, null=True)
    predict_class_knn = models.CharField(max_length=50, blank=True, null=True)
    media_svc = models.FloatField()
    desvio_svc = models.FloatField()
    class_svc = models.CharField(max_length=50, blank=True, null=True)
    predict_class_svc = models.CharField(max_length=50, blank=True, null=True)
    ordered_ranking = models.CharField(max_length=500)
    ordered_ranking2 = models.CharField(max_length=500)
    data_base = models.DecimalField(max_digits=5, decimal_places=0)


class Base(models.Model):
    data = models.DateField(auto_now=False, auto_now_add=False, blank=True, null=True)
    nome = models.CharField(max_length=50, blank=True, null=True)
    abertura = models.FloatField()
    close = models.FloatField()
    low = models.FloatField()
    high = models.FloatField()
    adj_close = models.FloatField()
    volume = models.CharField(max_length=20, blank=True, null=True)
    variacao = models.CharField(max_length=20, blank=True, null=True)
    previsao_normal = models.CharField(max_length=20, blank=True, null=True)
    mm9 = models.FloatField()
    mm21 = models.FloatField()
    bb_sup = models.FloatField()
    bb_inf = models.FloatField()
    previsao_return = models.CharField(max_length=20, blank=True, null=True)
    situacao_pont = models.CharField(max_length=20)
    nota = models.FloatField()
    oscilacao = models.FloatField()
    data_base = models.DecimalField(max_digits=5, decimal_places=0)

class ConsultaBase(models.Model):
    acao = models.CharField(max_length=20, blank=True, null=True)
    

class ConsultaResultado(models.Model):
    acao = models.CharField(max_length=20, blank=True, null=True)
    
class FaleComigo(models.Model):
    nome = models.CharField(max_length=100, blank=True, null=True)
    email = models.EmailField(max_length=254, blank=True, null=True) 
    colaboracao = models.TextField(blank=True, null=True) 



# models do treinamento

class Departamento(models.Model):
    nome = models.CharField(max_length=100)

    def __str__(self):
        return self.nome
    
class Cpf(models.Model):
    numero = models.CharField(max_length=11, primary_key=True)
    data_exp = models.DateField(auto_now=False)

    def __str__(self):
        return self.numero

    class Meta:
        verbose_name = 'CPF'  

class Empregado(models.Model):
    id = models.AutoField(primary_key=True, null=False)
    nome = models.CharField(max_length=20, null=False)
    sobrenome = models.CharField(max_length=50, null=False)
    salario = models.DecimalField(max_digits=10, decimal_places=2)
    idade = models.PositiveIntegerField(validators=[MinValueValidator(0), 
        MaxValueValidator(120)])
    email = models.EmailField()
    cpf = models.OneToOneField(Cpf, on_delete=models.CASCADE, blank=True, null=True)
    departamentos = models.ManyToManyField(Departamento, blank=True, null=True)

    class Meta:
        ordering = ['nome']
        verbose_name = 'Cadastro de Empregado'

    def __str__(self):
        return self.nome # this method pu the name into admin console line

class Telefone(models.Model):
    id = models.AutoField(primary_key=True, null=False)
    numero = models.CharField(max_length=10)
    descricao = models.CharField(max_length=80)
    empregado = models.ForeignKey(Empregado, on_delete=models.CASCADE)

    class Meta:
        verbose_name= 'Cadastro de Telefone'

    def __str__(self):
        return self.descricao



    

