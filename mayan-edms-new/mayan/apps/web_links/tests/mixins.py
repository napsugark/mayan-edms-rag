from mayan.apps.testing.tests.mixins import TestMixinObjectCreationTrack

from ..models import WebLink

from .literals import (
    TEST_WEB_LINK_LABEL, TEST_WEB_LINK_LABEL_EDITED, TEST_WEB_LINK_TEMPLATE
)


class WebLinkTestMixin(TestMixinObjectCreationTrack):
    _test_object_model = WebLink
    _test_object_name = '_test_web_link'

    def _create_test_web_link(self, add_test_document_type=False):
        self._test_web_link = WebLink.objects.create(
            label=TEST_WEB_LINK_LABEL, template=TEST_WEB_LINK_TEMPLATE,
        )
        if add_test_document_type:
            self._test_web_link.document_types.add(self._test_document_type)


class DocumentTypeAddRemoveWebLinkViewTestMixin(WebLinkTestMixin):
    def _request_test_document_type_web_link_add_remove_get_view(self):
        return self.get(
            viewname='web_links:document_type_web_links', kwargs={
                'document_type_id': self._test_document_type.pk
            }
        )

    def _request_test_document_type_web_link_add_view(self):
        return self.post(
            viewname='web_links:document_type_web_links', kwargs={
                'document_type_id': self._test_document_type.pk
            }, data={
                'available-submit': 'true',
                'available-selection': self._test_web_link.pk
            }
        )

    def _request_test_document_type_web_link_remove_view(self):
        return self.post(
            viewname='web_links:document_type_web_links', kwargs={
                'document_type_id': self._test_document_type.pk
            }, data={
                'added-submit': 'true',
                'added-selection': self._test_web_link.pk
            }
        )


class ResolvedWebLinkAPIViewTestMixin(WebLinkTestMixin):
    def _request_resolved_web_link_detail_api_view(self):
        return self.get(
            viewname='rest_api:resolved_web_link-detail',
            kwargs={
                'document_id': self._test_document.pk,
                'resolved_web_link_id': self._test_web_link.pk
            }
        )
        self._create_test_web_link(add_test_document_type=True)

    def _request_resolved_web_link_list_api_view(self):
        return self.get(
            viewname='rest_api:resolved_web_link-list', kwargs={
                'document_id': self._test_document.pk
            }
        )

    def _request_resolved_web_link_navigate_api_view(self):
        return self.get(
            viewname='rest_api:resolved_web_link-navigate',
            kwargs={
                'document_id': self._test_document.pk,
                'resolved_web_link_id': self._test_web_link.pk
            }
        )


class WebLinkAPIViewTestMixin(WebLinkTestMixin):
    def _request_test_web_link_create_api_view(self):
        self._test_object_track()

        response = self.post(
            viewname='rest_api:web_link-list', data={
                'label': TEST_WEB_LINK_LABEL,
                'template': TEST_WEB_LINK_TEMPLATE
            }
        )

        self._test_object_set()

        return response

    def _request_test_web_link_delete_api_view(self):
        return self.delete(
            viewname='rest_api:web_link-detail', kwargs={
                'web_link_id': self._test_web_link.pk
            }
        )

    def _request_test_web_link_detail_api_view(self):
        return self.get(
            viewname='rest_api:web_link-detail', kwargs={
                'web_link_id': self._test_web_link.pk
            }
        )

    def _request_test_web_link_edit_via_patch_api_view(self):
        return self.patch(
            viewname='rest_api:web_link-detail',
            kwargs={'web_link_id': self._test_web_link.pk}, data={
                'label': TEST_WEB_LINK_LABEL_EDITED
            }
        )

    def _request_test_web_link_edit_via_put_api_view(self):
        return self.put(
            viewname='rest_api:web_link-detail',
            kwargs={'web_link_id': self._test_web_link.pk}, data={
                'label': TEST_WEB_LINK_LABEL_EDITED,
                'template': TEST_WEB_LINK_TEMPLATE
            }
        )


class WebLinkDocumentTypeAPIViewMixin(WebLinkTestMixin):
    def _request_test_web_link_document_type_add_api_view(self):
        return self.post(
            viewname='rest_api:web_link-document_type-add',
            kwargs={'web_link_id': self._test_web_link.pk}, data={
                'document_type': self._test_document_type.pk
            }
        )

    def _request_test_web_link_document_type_list_api_view(self):
        return self.get(
            viewname='rest_api:web_link-document_type-list', kwargs={
                'web_link_id': self._test_web_link.pk
            }
        )

    def _request_test_web_link_document_type_remove_api_view(self):
        return self.post(
            viewname='rest_api:web_link-document_type-remove',
            kwargs={'web_link_id': self._test_web_link.pk}, data={
                'document_type': self._test_document_type.pk
            }
        )


class WebLinkDocumentTypeViewTestMixin(WebLinkTestMixin):
    def _request_test_web_link_document_type_add_remove_get_view(self):
        return self.get(
            viewname='web_links:web_link_document_types', kwargs={
                'web_link_id': self._test_web_link.pk
            }
        )

    def _request_test_web_link_document_type_add_view(self):
        return self.post(
            viewname='web_links:web_link_document_types', kwargs={
                'web_link_id': self._test_web_link.pk
            }, data={
                'available-selection': self._test_document_type.pk,
                'available-submit': 'true'
            }
        )

    def _request_test_web_link_document_type_remove_view(self):
        return self.post(
            viewname='web_links:web_link_document_types', kwargs={
                'web_link_id': self._test_web_link.pk
            }, data={
                'added-selection': self._test_document_type.pk,
                'added-submit': 'true'
            }
        )


class WebLinkViewTestMixin(WebLinkTestMixin):
    def _request_test_document_web_link_instance_view(self):
        return self.post(
            viewname='web_links:web_link_instance_view', kwargs={
                'document_id': self._test_document.pk,
                'web_link_id': self._test_web_link.pk
            }
        )

    def _request_test_document_web_link_list_view(self):
        return self.get(
            viewname='web_links:document_web_link_list', kwargs={
                'document_id': self._test_document.pk
            }
        )

    def _request_test_web_link_create_view(self):
        self._test_object_track()

        response = self.post(
            viewname='web_links:web_link_create', data={
                'label': TEST_WEB_LINK_LABEL,
                'template_template': TEST_WEB_LINK_TEMPLATE
            }
        )

        self._test_object_set()

        return response

    def _request_test_web_link_delete_view(self):
        return self.post(
            viewname='web_links:web_link_delete', kwargs={
                'web_link_id': self._test_web_link.pk
            }
        )

    def _request_test_web_link_edit_view(self):
        return self.post(
            viewname='web_links:web_link_edit', kwargs={
                'web_link_id': self._test_web_link.pk
            }, data={
                'label': TEST_WEB_LINK_LABEL_EDITED,
                'template_template': TEST_WEB_LINK_TEMPLATE
            }
        )

    def _request_test_web_link_list_view(self):
        return self.get(viewname='web_links:web_link_list')
