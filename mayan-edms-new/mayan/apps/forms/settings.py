from django.utils.translation import gettext_lazy as _

from mayan.apps.smart_settings.settings import setting_cluster

from .literals import DEFAULT_FORMS_SHOW_DROPZONE_SUBMIT_BUTTON

setting_namespace = setting_cluster.do_namespace_add(
    label=_(message='Forms'), name='forms'
)

setting_show_dropzone_submit_button = setting_namespace.do_setting_add(
    default=DEFAULT_FORMS_SHOW_DROPZONE_SUBMIT_BUTTON,
    global_name='FORMS_SHOW_DROPZONE_SUBMIT_BUTTON', help_text=_(
        message='Display a submit button and wait for the submit button to be '
        'pressed before processing up the upload.'
    )
)
