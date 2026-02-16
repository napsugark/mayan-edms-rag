from django.urls import reverse

from mayan.apps.converter.links import (
    link_transformation_delete_single, link_transformation_edit,
    link_transformation_select
)
from mayan.apps.converter.tests.mixins import TransformationViewTestMixin
from mayan.apps.documents.tests.base import GenericDocumentViewTestCase

from ..layers import layer_redactions
from ..links import link_redaction_list
from ..permissions import (
    permission_redaction_create, permission_redaction_delete,
    permission_redaction_edit, permission_redaction_view
)
from ..transformations import TransformationRedactionPercent

from .literals import TEST_REDACTION_ARGUMENT


class RedactionViewTestCase(
    TransformationViewTestMixin, GenericDocumentViewTestCase
):
    _test_layer = layer_redactions
    _test_transformation_argument = TEST_REDACTION_ARGUMENT
    auto_create_test_layer = False
    auto_create_test_transformation_class = False
    TestTransformationClass = TransformationRedactionPercent

    def test_redaction_delete_single_link_no_permissions(self):
        self._create_test_transformation()

        self.add_test_view(test_object=self._test_transformation)

        context = self.get_test_view()
        context['layer_name'] = self._test_layer.name
        context['content_object'] = self._test_transformation_object
        resolved_link = link_transformation_delete_single.resolve(
            context=context
        )

        self.assertEqual(resolved_link, None)

    def test_redaction_delete_single_link_with_access(self):
        self._create_test_transformation()

        self.grant_access(
            obj=self._test_transformation_object_parent,
            permission=permission_redaction_delete
        )

        self.add_test_view(test_object=self._test_transformation)

        context = self.get_test_view()
        context['layer_name'] = self._test_layer.name
        context['content_object'] = self._test_transformation_object
        resolved_link = link_transformation_delete_single.resolve(
            context=context
        )

        self.assertNotEqual(resolved_link, None)

        self.assertEqual(
            resolved_link.url, reverse(
                viewname=link_transformation_delete_single.view, kwargs={
                    'app_label': self._test_transformation_object_content_type.app_label,
                    'model_name': self._test_transformation_object_content_type.model,
                    'object_id': self._test_transformation_object.pk,
                    'layer_name': self._test_layer.name,
                    'transformation_id': self._test_transformation.pk
                }
            )
        )

    def test_redaction_edit_link_no_permission(self):
        self._create_test_transformation()

        self.add_test_view(test_object=self._test_transformation)

        context = self.get_test_view()
        context['layer_name'] = self._test_layer.name
        context['content_object'] = self._test_transformation_object
        resolved_link = link_transformation_edit.resolve(context=context)
        self.assertEqual(resolved_link, None)

    def test_redaction_edit_link_with_access(self):
        self._create_test_transformation()

        self.grant_access(
            obj=self._test_transformation_object_parent,
            permission=permission_redaction_edit
        )

        self.add_test_view(test_object=self._test_transformation)

        context = self.get_test_view()
        context['layer_name'] = self._test_layer.name
        context['content_object'] = self._test_transformation_object
        resolved_link = link_transformation_edit.resolve(context=context)
        self.assertNotEqual(resolved_link, None)

        self.assertEqual(
            resolved_link.url, reverse(
                viewname=link_transformation_edit.view, kwargs={
                    'app_label': self._test_transformation_object_content_type.app_label,
                    'model_name': self._test_transformation_object_content_type.model,
                    'object_id': self._test_transformation_object.pk,
                    'layer_name': self._test_layer.name,
                    'transformation_id': self._test_transformation.pk
                }
            )
        )

    def test_redaction_select_link_no_permission(self):
        self._create_test_transformation()

        self.add_test_view(test_object=self._test_transformation)

        context = self.get_test_view()
        context['layer_name'] = self._test_layer.name
        context['content_object'] = self._test_transformation_object
        resolved_link = link_transformation_select.resolve(context=context)

        self.assertEqual(resolved_link, None)

    def test_redaction_select_link_with_access(self):
        self._create_test_transformation()

        self.grant_access(
            obj=self._test_transformation_object_parent,
            permission=permission_redaction_create
        )

        self.add_test_view(test_object=self._test_transformation)

        context = self.get_test_view()
        context['layer_name'] = self._test_layer.name
        context['content_object'] = self._test_transformation_object
        resolved_link = link_transformation_select.resolve(context=context)
        self.assertNotEqual(resolved_link, None)

        self.assertEqual(
            resolved_link.url, reverse(
                viewname=link_transformation_select.view, kwargs={
                    'app_label': self._test_transformation_object_content_type.app_label,
                    'model_name': self._test_transformation_object_content_type.model,
                    'object_id': self._test_transformation_object.pk,
                    'layer_name': self._test_layer.name
                }
            )
        )

    def test_redaction_list_link_no_permission(self):
        self._create_test_transformation()

        self.add_test_view(test_object=self._test_transformation)

        context = self.get_test_view()
        context['layer_name'] = self._test_layer.name
        context['content_object'] = self._test_transformation_object
        resolved_link = link_redaction_list.resolve(context=context)

        self.assertEqual(resolved_link, None)

    def test_redaction_list_link_with_access(self):
        self._create_test_transformation()

        self.grant_access(
            obj=self._test_transformation_object_parent,
            permission=permission_redaction_view
        )

        self.add_test_view(test_object=self._test_transformation)

        context = self.get_test_view()
        context['layer_name'] = self._test_layer.name
        context['content_object'] = self._test_transformation_object
        resolved_link = link_redaction_list.resolve(context=context)
        self.assertNotEqual(resolved_link, None)

        self.assertEqual(
            resolved_link.url, reverse(
                viewname=link_redaction_list.view, kwargs={
                    'app_label': self._test_transformation_object_content_type.app_label,
                    'model_name': self._test_transformation_object_content_type.model,
                    'object_id': self._test_transformation_object.pk,
                    'layer_name': self._test_layer.name
                }
            )
        )


class RedactionLinkDisplayTestCase(
    TransformationViewTestMixin, GenericDocumentViewTestCase
):
    _test_layer = layer_redactions
    _test_transformation_argument = TEST_REDACTION_ARGUMENT
    auto_create_test_layer = False
    auto_create_test_transformation_class = False
    TestTransformationClass = TransformationRedactionPercent

    def test_redaction_delete_link_view_with_view_access(self):
        self._create_test_transformation()

        self.grant_access(
            obj=self._test_transformation_object_parent,
            permission=permission_redaction_view
        )

        self._clear_events()

        response = self._request_transformation_list_view()
        self.assertContains(
            response=response, text=str(self._test_transformation_object),
            status_code=200
        )
        self.assertContains(
            response=response,
            text=self._test_transformation.get_transformation_class().label,
            status_code=200
        )
        self.assertNotContains(
            response=response, text=link_transformation_delete_single.text,
            status_code=200
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_redaction_delete_link_view_with_all_access(self):
        self._create_test_transformation()

        self.grant_access(
            obj=self._test_transformation_object_parent,
            permission=permission_redaction_delete
        )
        self.grant_access(
            obj=self._test_transformation_object_parent,
            permission=permission_redaction_view
        )

        self._clear_events()

        response = self._request_transformation_list_view()
        self.assertContains(
            response=response, text=str(self._test_transformation_object),
            status_code=200
        )
        self.assertContains(
            response=response,
            text=self._test_transformation.get_transformation_class().label,
            status_code=200
        )
        self.assertContains(
            response=response, text=link_transformation_delete_single.text,
            status_code=200
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_redaction_edit_link_view_with_view_access(self):
        self._create_test_transformation()

        self.grant_access(
            obj=self._test_transformation_object_parent,
            permission=permission_redaction_view
        )

        self._clear_events()

        response = self._request_transformation_list_view()
        self.assertContains(
            response=response, text=str(self._test_transformation_object),
            status_code=200
        )
        self.assertContains(
            response=response,
            text=self._test_transformation.get_transformation_class().label,
            status_code=200
        )
        self.assertNotContains(
            response=response,
            text=link_transformation_edit.text,
            status_code=200
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_redaction_edit_link_view_with_all_access(self):
        self._create_test_transformation()

        self.grant_access(
            obj=self._test_transformation_object_parent,
            permission=permission_redaction_edit
        )
        self.grant_access(
            obj=self._test_transformation_object_parent,
            permission=permission_redaction_view
        )

        self._clear_events()

        response = self._request_transformation_list_view()
        self.assertContains(
            response=response, text=str(self._test_transformation_object),
            status_code=200
        )
        self.assertContains(
            response=response,
            text=self._test_transformation.get_transformation_class().label,
            status_code=200
        )
        self.assertContains(
            response=response,
            text=link_transformation_edit.text,
            status_code=200
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)
