from django.contrib import admin
from .models import Empregado, Telefone, Cpf, Departamento, Acoes, Simulador

class EmpregadoAdmin(admin.ModelAdmin):
    list_display = ('nome', 'idade')
    list_filter = (['departamentos'])
    search_fields = (['nome'])


admin.site.register(Departamento)
admin.site.register(Simulador)
admin.site.register(Acoes)
admin.site.register(Telefone)
admin.site.register(Cpf)
admin.site.register(Empregado,  EmpregadoAdmin)