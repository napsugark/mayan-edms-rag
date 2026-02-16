from django.template import TemplateSyntaxError

from rest_framework.exceptions import ValidationError
from rest_framework.permissions import IsAuthenticated
from rest_framework.settings import api_settings

from mayan.apps.rest_api import generics
from mayan.apps.rest_api.api_view_mixins import ContentTypeAPIViewMixin

from .classes import AJAXTemplate, ModelTemplating
from .permissions import permission_template_sandbox
from .serializers import (
    AJAXTemplateSerializer, ObjectTemplateSandboxSerializer
)


class APITemplateDetailView(generics.RetrieveAPIView):
    """
    Returns the selected partial template details.
    get: Retrieve the details of the partial template.
    """
    permission_classes = (IsAuthenticated,)
    serializer_class = AJAXTemplateSerializer

    def get_object(self):
        return AJAXTemplate.get(
            name=self.kwargs['name']
        ).render(
            request=self.request
        )


class APITemplateListView(generics.ListAPIView):
    """
    Returns a list of all the available templates.
    """
    permission_classes = (IsAuthenticated,)
    serializer_class = AJAXTemplateSerializer

    def get_source_queryset(self):
        return AJAXTemplate.all(
            rendered=True, request=self.request
        )


class APIObjectTemplateSandboxActionView(
    ContentTypeAPIViewMixin, generics.ObjectActionAPIView
):
    """
    post: Interactive inspection of object properties.
    """
    mayan_object_permission_map = {
        'GET': permission_template_sandbox,
        'POST': permission_template_sandbox
    }
    lookup_url_kwarg = 'object_id'
    serializer_class = ObjectTemplateSandboxSerializer

    def object_action(self, obj, request, serializer):
        template_string = serializer.validated_data['template']

        try:
            result = ModelTemplating.do_render(
                obj=obj, template_string=template_string
            )
        except KeyError as exception:
            raise ValidationError(
                {
                    api_settings.NON_FIELD_ERRORS_KEY: [
                        str(exception)
                    ]
                }, code='invalid'
            )
        except TemplateSyntaxError as exception:
            raise ValidationError(
                {
                    'template': [
                        str(exception)
                    ]
                }, code='invalid'
            )
        else:
            return {'result': result}

    def get_source_queryset(self):
        content_type = self.get_content_type()
        return content_type.get_all_objects_for_this_type()
