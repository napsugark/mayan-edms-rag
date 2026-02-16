from django.utils.translation import gettext_lazy as _

from mayan.apps.dependencies.classes import JavaScriptDependency

JavaScriptDependency(
    label=_(message='FontAwesome'), module=__name__,
    name='@fortawesome/fontawesome-free', version_string='=6.7.2'
)
