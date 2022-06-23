# Generated by Django 2.2.12 on 2022-06-22 14:56

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='TopicModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('projectName', models.CharField(default='', max_length=100)),
                ('inpType', models.CharField(default='', max_length=100)),
                ('ndoc', models.CharField(default='', max_length=100)),
                ('tsize', models.CharField(default='', max_length=100)),
                ('doc_type', models.CharField(default='', max_length=100)),
                ('document', models.CharField(default='', max_length=100)),
                ('Centre', models.CharField(default='', max_length=100)),
                ('lang', models.CharField(default='', max_length=100)),
            ],
        ),
    ]
