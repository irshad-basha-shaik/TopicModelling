from django import forms
from .models import TopicModel

INPUTTYPE = [
    ('Select', 'Select'),
    ('IDI', 'IDI'),
    ('FGD SECONDARY RESEARCH', 'FGD SECONDARY RESEARCH'),
    ('ONLINE SCRAPING', 'ONLINE SCRAPING')
]
TEXTSIZE = [
    ('Select', 'Select'),
    ('Long', 'Long'),
    ('Short', 'Short')
]
DOCTYPE = [
    ('Select', 'Select'),
    ('word', 'word'),
    ('pdf', 'pdf'),
    ('excel', 'excel'),
    ('csv', 'csv')
]
CENTER = [
    ('Select', 'Select'),
    ('Centre1', 'Centre1'),
    ('Centre2', 'Centre2')
]
LANGUAGE = [
    ('Select', 'Select'),
    ('English', 'English'),
    ('Hindi', 'Hindi')
]
class TopicForm(forms.ModelForm):
    projectName = forms.CharField(max_length=100,widget=forms.TextInput(attrs={'class': 'form-control'}),required=False)
    inpType = forms.ChoiceField(choices=INPUTTYPE,widget=forms.Select(attrs={'class': 'form-control'}))
    ndoc = forms.CharField(max_length=100,widget=forms.TextInput(attrs={'class': 'form-control'}))
    tsize = forms.ChoiceField(choices=TEXTSIZE,widget=forms.Select(attrs={'class': 'form-control'}))
    doc_type = forms.ChoiceField(choices=DOCTYPE,widget=forms.Select(attrs={'class': 'form-control'}))
    document = forms.ChoiceField(choices=DOCTYPE,widget=forms.Select(attrs={'class': 'form-control'}))
    Centre = forms.ChoiceField(choices=CENTER,widget=forms.Select(attrs={'class': 'form-control'}))
    lang = forms.ChoiceField(choices=LANGUAGE,widget=forms.Select(attrs={'class': 'form-control'}))

    def clean(self):
        cleaned_data = super(TopicForm, self).clean()
        self.instance.field = 'value'
        return cleaned_data

    class Meta:
        model = TopicModel
        fields = ['projectName','inpType','ndoc','tsize','doc_type','document','Centre','lang']
