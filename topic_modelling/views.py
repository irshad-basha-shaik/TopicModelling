from django.shortcuts import render

# Create your views here.
def home(request):
    return render(request, "topic_modelling/home.html")
def doc_rules(request):
    return render(request, "topic_modelling/doc_rules.html")
def generate_concept(request):
    return render(request, "topic_modelling/generate_concept.html")
def generate_thesarus(request):
    return render(request, "topic_modelling/generate_thesarus.html")
def generate_results(request):
    return render(request, "topic_modelling/generate_results.html")
def project_list(request):
    return render(request, "topic_modelling/project_list.html")
