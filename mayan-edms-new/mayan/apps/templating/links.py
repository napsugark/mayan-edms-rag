from django.utils.translation import gettext_lazy as _

from mayan.apps.navigation.links import Link
from mayan.apps.navigation.utils import get_content_type_kwargs_factory

from .icons import icon_template_sandbox
from .permissions import permission_template_sandbox

link_object_template_sandbox = Link(
    icon=icon_template_sandbox, kwargs=get_content_type_kwargs_factory(
        variable_name='resolved_object'
    ), permission=permission_template_sandbox, text=_(message='Sandbox'),
    view='templating:object_template_sandbox'
)
