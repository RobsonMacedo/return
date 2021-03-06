# Generated by Django 3.1.7 on 2021-03-30 22:41

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Acoes',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('acao', models.CharField(max_length=100)),
                ('previsao_data_ini', models.DateField()),
                ('previsao_data_fim', models.DateField()),
            ],
        ),
        migrations.CreateModel(
            name='Algos',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('algo', models.CharField(max_length=100)),
                ('acao', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='core.acoes')),
            ],
        ),
    ]
