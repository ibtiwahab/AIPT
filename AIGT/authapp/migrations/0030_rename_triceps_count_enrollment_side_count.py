# Generated by Django 4.2.1 on 2023-06-06 08:35

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('authapp', '0029_alter_enrollment_price'),
    ]

    operations = [
        migrations.RenameField(
            model_name='enrollment',
            old_name='triceps_count',
            new_name='side_count',
        ),
    ]
