# Generated by Django 4.2.1 on 2023-06-03 13:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('authapp', '0009_delete_counter'),
    ]

    operations = [
        migrations.AddField(
            model_name='enrollment',
            name='biceps_count',
            field=models.IntegerField(default=0),
        ),
    ]