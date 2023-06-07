from django.apps import AppConfig
from django import forms


class AuthappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'authapp'

class ExerciseForm(forms.Form):
    name = forms.CharField(label='Name')
    age = forms.IntegerField(label='Age')
    weight = forms.FloatField(label='Weight')
    height = forms.FloatField(label='Height')
    bmi = forms.FloatField(label='BMI')