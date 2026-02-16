"""
This module should be called settings.py but is named conf.py to avoid a
clash with the mayan/settings/* module
"""
from django.utils.translation import gettext_lazy as _

from mayan.apps.smart_settings.settings import setting_cluster

setting_namespace = setting_cluster.do_namespace_add(
    name='mayan', label=_(message='Mayan')
)

setting_celery_class = setting_namespace.do_setting_add(
    default='celery.Celery', help_text=_(
        message='The class used to instantiate the main Celery app.'
    ), global_name='MAYAN_CELERY_CLASS'
)
