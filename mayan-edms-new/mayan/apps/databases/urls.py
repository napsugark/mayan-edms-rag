from django.urls import re_path

from .views import ModelPropertyListView, PropertyModelListView

urlpatterns = [
    re_path(
        route=r'^models/$', name='property_model_list',
        view=PropertyModelListView.as_view()
    ),
    re_path(
        route=r'^apps/(?P<app_label>[-\w]+)/models/(?P<model_name>[-\w]+)/properties/$',
        name='model_property_list', view=ModelPropertyListView.as_view()
    ),


]
