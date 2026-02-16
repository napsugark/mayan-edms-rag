from django.contrib.contenttypes.models import ContentType

from mayan.apps.databases.tests.mixins import ContentTypeTestCaseMixin

from ..classes import Template

from .literals import TEST_TEMPLATE


class ObjectTemplateSandboxActionApiViewTestMixin(ContentTypeTestCaseMixin):
    def _request_object_template_sandbox_get_api_view(self):
        return self.get(
            data={'template': TEST_TEMPLATE}, kwargs={
                'app_label': self._test_object_content_type.app_label,
                'model_name': self._test_object_content_type.model,
                'object_id': self._test_object.pk
            }, viewname='rest_api:object-template-sandbox'
        )

    def _request_object_template_sandbox_post_api_view(self):
        return self.post(
            data={'template': TEST_TEMPLATE}, kwargs={
                'app_label': self._test_object_content_type.app_label,
                'model_name': self._test_object_content_type.model,
                'object_id': self._test_object.pk
            }, viewname='rest_api:object-template-sandbox'
        )

    def _request_object_template_sandbox_post_api_view_invalid_model(self):
        content_type = ContentType.objects.get_for_model(model=ContentType)

        return self.post(
            data={'template': TEST_TEMPLATE}, kwargs={
                'app_label': content_type.app_label,
                'model_name': content_type.model,
                'object_id': ContentType.objects.first().pk
            }, viewname='rest_api:object-template-sandbox'
        )


class ObjectTemplateSandboxViewTestMixin(ContentTypeTestCaseMixin):
    def _request_object_template_sandbox_get_view(self):
        return self.get(
            kwargs=self._test_object_view_kwargs,
            viewname='templating:object_template_sandbox'
        )

    def _request_object_template_sandbox_post_view(self):
        return self.post(
            data={'template_template': TEST_TEMPLATE},
            kwargs=self._test_object_view_kwargs,
            viewname='templating:object_template_sandbox'
        )


class TemplateTestMixin:
    def _render_test_template(self, template_string, context=None):
        template = Template(template_string=template_string)
        return template.render(context=context)
