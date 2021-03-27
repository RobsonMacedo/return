"""return URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
import re
from .views import home
from core.views import empregados, empregado_detalhe, empregado_Robson, empregado_por_nome, resultados


urlpatterns = [
    path('admin/', admin.site.urls),
    path('empregados/', empregados),
    path('empregado_robson/', empregado_Robson),
    path('empregado/<int:id>/', empregado_detalhe),
    path('empregado/<str:nome>/', empregado_por_nome),
    path('resultados/', resultados, name='resultados'),
    path('', home, name='home'),
    
]
