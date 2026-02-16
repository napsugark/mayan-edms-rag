import hashlib

from django.template.response import TemplateResponse
from django.utils.translation import gettext_lazy as _
from django.urls import reverse

from mayan.apps.acls.classes import ModelPermission
from mayan.apps.common.menus import menu_list_facet

from .links import link_object_template_sandbox
from .permissions import permission_template_sandbox
from .template_backends import Template


class AJAXTemplate:
    _registry = {}

    @classmethod
    def all(cls, rendered=False, request=None):
        if not rendered:
            return cls._registry.values()
        else:
            result = []
            for template in cls._registry.values():
                result.append(
                    template.render(request=request)
                )
            return result

    @classmethod
    def get(cls, name):
        return cls._registry[name]

    def __init__(self, name, template_name, context=None):
        self.context = context or None
        self.name = name
        self.template_name = template_name
        self.__class__._registry[name] = self

    def get_absolute_url(self):
        return reverse(
            kwargs={'name': self.name}, viewname='rest_api:template-detail'
        )

    def render(self, request):
        result = TemplateResponse(
            context=self.context, request=request,
            template=self.template_name
        ).render()

        # Calculate the hash of the bytes version but return the unicode
        # version.
        self.html = result.rendered_content.replace('\n', '')
        self.hex_hash = hashlib.sha256(string=result.content).hexdigest()
        return self


class ModelTemplating:
    _registry = {}

    @classmethod
    def do_render(cls, obj, template_string):
        model = obj._meta.model

        model_templating = cls.get_for_model(model=model)

        template = Template(template_string=template_string)
        result = template.render(
            context={
                model_templating.variable_name: obj
            }
        )

        return result

    @classmethod
    def get_for_model(cls, model):
        try:
            return cls._registry[model]
        except KeyError:
            raise KeyError(
                _(
                    message='Model `%s` is not available for template '
                    'sandbox.'
                ) % model._meta.verbose_name
            )

    def __init__(self, model, variable_name):
        self.model = model
        self.variable_name = variable_name

        ModelPermission.register(
            model=self.model, permissions=(permission_template_sandbox,)
        )

        menu_list_facet.bind_links(
            links=(link_object_template_sandbox,), sources=(self.model,)
        )

        self.__class__._registry[self.model] = self
