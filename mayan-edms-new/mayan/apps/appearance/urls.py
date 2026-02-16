from django.urls import re_path

from mayan.apps.views.generics import SimpleView

urlpatterns_error_pages = [
    re_path(
        route=r'^errors/403/$', name='error_403', view=SimpleView.as_view(
            template_name='403.html'
        )
    ),
    re_path(
        route=r'^errors/404/$', name='error_404', view=SimpleView.as_view(
            template_name='404.html'
        )
    ),
    re_path(
        route=r'^errors/500/$', name='error_500', view=SimpleView.as_view(
            template_name='500.html'
        )
    )
]

urlpatterns = []
urlpatterns.extend(urlpatterns_error_pages)
