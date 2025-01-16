"""
URL configuration for djangoProject1 project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from webtests import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("webtests/",include("webtests.url")),
    path('', views.login, name='login'),
    path('signup/', views.signup,name='signup'),
    path('preference/', views.preference,name='preference'),
    path('movie_recommend/', views.movie_recommend,name='movie'),
    path('movie_all/', views.movie_all, name='movie_all'),
    path('movie_all/<int:movie_id>/', views.movie_detail,name='movie_detail'),
    path('search_results/', views.search_results, name='search_results'),
    path('user_homepage/', views.user_homepage, name='user_homepage'),
]
