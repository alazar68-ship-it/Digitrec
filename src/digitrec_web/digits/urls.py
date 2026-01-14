from __future__ import annotations

from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("health", views.health, name="health"),
    path("api/predict", views.api_predict, name="api_predict"),
    path("ui/predict", views.ui_predict, name="ui_predict"),
]
