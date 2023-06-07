from django import forms
from .models import Blog


class ExerciseForm(forms.Form):
    name = forms.CharField(label='Name')
    age = forms.IntegerField(label='Age')
    weight = forms.FloatField(label='Weight')
    height = forms.FloatField(label='Height')
    bmi = forms.FloatField(label='BMI')

class BlogForm(forms.ModelForm):
    
    class Meta:
        model = Blog
        fields = ['title', 'image', 'description']