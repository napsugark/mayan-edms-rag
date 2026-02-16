from mayan.apps.documents.tests.base import GenericDocumentViewTestCase

from ..links import link_transformation_delete_single

from .mixins import TransformationViewTestMixin


class TransformationLinkConditionTestCase(
    TransformationViewTestMixin, GenericDocumentViewTestCase
):
    auto_login_super_user = True
    create_test_case_super_user = True
    create_test_case_user = False

    def test_transformation_condition_link_view_with_super_user(self):
        self._create_test_transformation()

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

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class TransformationLinkDisplayTestCase(
    TransformationViewTestMixin, GenericDocumentViewTestCase
):
    def test_transformation_delete_link_view_with_view_access(self):
        self._create_test_transformation()

        self.grant_access(
            obj=self._test_transformation_object_parent,
            permission=self._test_layer_permission_view
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

    def test_transformation_delete_link_view_with_all_access(self):
        self._create_test_transformation()

        self.grant_access(
            obj=self._test_transformation_object_parent,
            permission=self._test_layer_permission_delete
        )
        self.grant_access(
            obj=self._test_transformation_object_parent,
            permission=self._test_layer_permission_view
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
