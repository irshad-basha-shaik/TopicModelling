from django.db import models

# Create your models here.
class TopicModel(models.Model):
    projectName = models.CharField(max_length=100, default='')
    inpType = models.CharField(max_length=100, default='')
    ndoc = models.CharField(max_length=100,default='')
    tsize = models.CharField(max_length=100,default='')
    doc_type = models.CharField(max_length=100,default='')
    document = models.CharField(max_length=100, default='')
    Centre = models.CharField(max_length=100, default='')
    lang = models.CharField(max_length=100, default='')


