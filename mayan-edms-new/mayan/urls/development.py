from django.conf import settings
from django.urls import include, re_path

from .base import *  # NOQA

if 'debug_toolbar' in settings.INSTALLED_APPS:
    try:
        import debug_toolbar
    except ImportError:
        pass
    else:
        urlpatterns += [  # NOQA
            re_path(
                route=r'^__debug__/', view=include(debug_toolbar.urls)
            )
        ]

if 'silk' in settings.INSTALLED_APPS:
    try:
        import silk
    except ImportError:
        pass
    else:
        urlpatterns += [  # NOQA
            re_path(
                route=r'^silk/', view=include('silk.urls', namespace='silk')
            )
        ]
