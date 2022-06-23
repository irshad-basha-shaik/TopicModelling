from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('home', views.home, name='home'),
    path('GenerateLDAModel_html', views.GenerateLDAModel_html, name='GenerateLDAModel_html'),
    path('color_green', views.color_green, name='color_green'),
    path('make_bold', views.make_bold, name='make_bold'),
    path('clean_text', views.clean_text, name='clean_text'),
    path('lemmatizer', views.lemmatizer, name='lemmatizer'),
    path('doc_rules', views.doc_rules, name='doc_rules'),
    path('generate_concept', views.generate_concept, name='generate_concept'),
    path('generate_thesarus', views.generate_thesarus, name='generate_thesarus'),
    path('generate_results', views.generate_results, name='generate_results'),
    path('project_list', views.project_list, name='project_list')

]