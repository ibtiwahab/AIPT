from django.urls import path
from authapp import views
from authapp.views import *

urlpatterns = [
    path('',views.Home,name="Home"),
    path('signup',views.signup,name="signup"),
    path('login',views.handlelogin,name="handlelogin"),
    path('logout',views.handleLogout,name="handleLogout"),
    path('contact',views.contact,name="contact"),
    path('join',views.enroll,name="enroll"),
    path('profile',views.profile,name="profile"),
    path('gallery',views.gallery,name="gallery"),
    path('attendance',views.attendance,name="attendance"),
    path('', index, name='index'),
    path('video_feed/', video_feed, name='video_feed'),
    #path('squat_counter/',squat counter,name='squat_counter'),
    path('squat_feed/',squat_feed,name='squat_feed'),
    path('shoulder_feed/',shoulder_feed,name='shoulder_feed'),
    path('biceps_feed/',biceps_feed,name='biceps_feed'),
    path('triceps_feed/',triceps_feed,name='triceps_feed'),
    path('exercise/', views.exercise_suggestion, name='exercise_suggestion'),
    path('create_blog/', views.create_blog, name='create_blog'),
    path('blog_list/', views.blog_list, name='blog_list'),
    path('biceps_inst/', biceps_inst, name='biceps_inst'),
    path('triceps_inst/', triceps_inst, name='triceps_inst'),
    
    path('pushups_inst/', pushups_inst, name='pushups_inst'),
    path('squats_inst/', squats_inst, name='squats_inst'),
    path('shoulderpresss_inst/', shoulderpresss_inst, name='shoulderpresss_inst'),
    path('biceps_table/', views.biceps_table, name='biceps_table'),
    path('shoulder_table/', views.shoulder_table, name='shoulder_table'),
    path('squat_table/', views.squat_table, name='squat_table'),
    path('triceps_table/', views.triceps_table, name='triceps_table'),
    path('pushup_table/', views.pushup_table, name='pushup_table'),
      
]
