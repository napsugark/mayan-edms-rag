from django.utils.translation import gettext_lazy as _

from mayan.apps.dependencies.classes import JavaScriptDependency

JavaScriptDependency(
    label=_(message='JavaScript image cropper'), module=__name__,
    name='cropperjs', version_string='=1.6.2'
)
JavaScriptDependency(
    module=__name__, name='jquery-cropper', version_string='=1.0.2'
)
