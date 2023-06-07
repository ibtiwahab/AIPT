from distutils.command.upload import upload
from django.db import models
from django.contrib.auth.models import User
# Create your models here.
# this file we creates a database tables

class Contact(models.Model):
    name=models.CharField(max_length=25)
    email=models.EmailField()
    phonenumber=models.CharField(max_length=12)
    description=models.TextField()

    def __str__(self):
        return self.email

class Enrollment(models.Model):        
    biceps_count = models.IntegerField(default=0)
    shoulder_count = models.IntegerField(default=0)
    squat_count = models.IntegerField(default=0)
    side_count = models.IntegerField(default=0)
    pushup_count = models.IntegerField(default=0)
    FullName = models.CharField(max_length=25)
    Email = models.EmailField()
    Gender = models.CharField(max_length=25)
    PhoneNumber = models.CharField(max_length=12, unique=True)
    DOB = models.CharField(max_length=50)
    SelectMembershipplan = models.CharField(max_length=200)
    SelectTrainer = models.CharField(max_length=55)
    Reference = models.CharField(max_length=55)
    Address = models.TextField()
    paymentStatus = models.CharField(max_length=55, blank=True, null=True)
    Price = models.IntegerField(blank=True, null=True)
    DueDate = models.DateTimeField(blank=True, null=True)
    timeStamp = models.DateTimeField(auto_now_add=True, blank=True)

    def __str__(self):
        return self.FullName


class BicepsHistory(models.Model):
    enrollment = models.ForeignKey(Enrollment, on_delete=models.CASCADE)
    biceps_count = models.IntegerField(default=0)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.enrollment.FullName} - {self.enrollment.PhoneNumber} - {self.biceps_count} - {self.timestamp}"
class ShoulderpressHistory(models.Model):
    enrollment = models.ForeignKey(Enrollment, on_delete=models.CASCADE)
    shoulder_count = models.IntegerField(default=0)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.enrollment.FullName} - {self.enrollment.PhoneNumber} - {self.shoulder_count} - {self.timestamp}"
class SquatHistory(models.Model):
    enrollment = models.ForeignKey(Enrollment, on_delete=models.CASCADE)
    squat_count = models.IntegerField(default=0)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.enrollment.FullName} - {self.enrollment.PhoneNumber} - {self.squat_count} - {self.timestamp}"
class TricepsHistory(models.Model):
    enrollment = models.ForeignKey(Enrollment, on_delete=models.CASCADE)
    side_count = models.IntegerField(default=0)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.enrollment.FullName} - {self.enrollment.PhoneNumber} - {self.side_count} - {self.timestamp}"
    
class PushupHistory(models.Model):
    enrollment = models.ForeignKey(Enrollment, on_delete=models.CASCADE)
    pushup_count = models.IntegerField(default=0)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.enrollment.FullName} - {self.enrollment.PhoneNumber} - {self.pushup_count} - {self.timestamp}"
class Trainer(models.Model):
    name=models.CharField(max_length=55)
    gender=models.CharField(max_length=25)
    phone=models.CharField(max_length=25)
    salary=models.IntegerField(max_length=25)
    timeStamp=models.DateTimeField(auto_now_add=True,blank=True)
    def __str__(self):
        return self.name

class MembershipPlan(models.Model):
    plan=models.CharField(max_length=185)
    price=models.IntegerField(max_length=55)

    def __int__(self):
        return self.id


class Gallery(models.Model):
    title=models.CharField(max_length=100)
    img=models.ImageField(upload_to='gallery')
    timeStamp=models.DateTimeField(auto_now_add=True,blank=True)
    def __int__(self):
        return self.id


class Attendance(models.Model):
    Selectdate=models.DateTimeField(auto_now_add=True)
    phonenumber=models.CharField(max_length=15)
    Login=models.CharField(max_length=200)
    Logout=models.CharField(max_length=200)
    SelectWorkout=models.CharField(max_length=200)
    TrainedBy=models.CharField(max_length=200)
    def __int__(self):
        return self.id

# fitness_recommendation/models.py

from django.db import models

class User(models.Model):
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    height = models.FloatField()
    weight = models.FloatField()
    bmi = models.FloatField()  # Body mass index

class Exercise(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()

class Blog(models.Model):
    title = models.CharField(max_length=255)
    image = models.ImageField(upload_to='blog_images')
    description = models.TextField()
   
    published_at = models.DateTimeField(auto_now_add=True)

    def _str_(self):
        return self.title
    


