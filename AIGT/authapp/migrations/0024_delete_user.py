# Generated by Django 4.2.1 on 2023-06-06 07:47

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('authapp', '0023_remove_enrollment_biceps_count_and_more'),
    ]

    operations = [
        migrations.DeleteModel(
            name='User',
        ),
    ]
