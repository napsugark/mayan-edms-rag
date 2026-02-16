from mayan.apps.source_web_forms.tests.mixins import WebFormSourceTestMixin
from mayan.apps.sources.tests.mixins.source_view_mixins import (
    SourceActionViewTestMixin
)
from mayan.apps.testing.tests.mixins import TestMixinObjectCreationTrack

from ..models import Tag

from .literals import (
    TEST_TAG_COLOR, TEST_TAG_COLOR_EDITED, TEST_TAG_LABEL,
    TEST_TAG_LABEL_EDITED
)


class TagTestMixin(TestMixinObjectCreationTrack):
    _test_object_model = Tag
    _test_object_name = '_test_tag'
    _test_tag_add_test_document = False
    auto_create_test_tag = False

    def setUp(self):
        super().setUp()
        self._test_tag_list = []

        if self.auto_create_test_tag:
            self._create_test_tag(
                add_test_document=self._test_tag_add_test_document
            )

    def _create_test_tag(self, add_test_document=False):
        total_test_label_count = len(self._test_tag_list)
        label = '{}_{}'.format(TEST_TAG_LABEL, total_test_label_count)

        self._test_tag = Tag.objects.create(
            color=TEST_TAG_COLOR, label=label
        )

        self._test_tag_list.append(self._test_tag)

        if add_test_document:
            self._test_tag.attach_to(
                document=self._test_document, user=self._test_case_user
            )


class DocumentTagViewTestMixin(TagTestMixin):
    def _request_test_document_tag_attach_view(self):
        return self.post(
            viewname='tags:tag_attach', kwargs={
                'document_id': self._test_document.pk
            }, data={
                'tags': self._test_tag.pk,
                'user': self._test_case_user.pk
            }
        )

    def _request_test_document_multiple_tag_multiple_attach_view(self):
        return self.post(
            viewname='tags:multiple_documents_tag_attach', data={
                'id_list': self._test_document.pk, 'tags': self._test_tag.pk,
                'user': self._test_case_user.pk
            }
        )

    def _request_test_document_tag_multiple_remove_view(self):
        return self.post(
            viewname='tags:single_document_multiple_tag_remove', kwargs={
                'document_id': self._test_document.pk
            }, data={
                'tags': self._test_tag.pk
            }
        )

    def _request_test_document_multiple_tag_remove_view(self):
        return self.post(
            viewname='tags:multiple_documents_selection_tag_remove', data={
                'id_list': self._test_document.pk,
                'tags': self._test_tag.pk
            }
        )

    def _request_test_document_tag_list_view(self):
        return self.get(
            viewname='tags:document_tag_list', kwargs={
                'document_id': self._test_document.pk
            }
        )

    def _request_test_tag_document_list_view(self):
        return self.get(
            viewname='tags:tag_document_list', kwargs={
                'tag_id': self._test_tag.pk
            }
        )


class TagAPIViewTestMixin(TagTestMixin):
    def _request_test_document_tag_attach_api_view(self):
        return self.post(
            viewname='rest_api:document-tag-attach', kwargs={
                'document_id': self._test_document.pk
            }, data={'tag': self._test_tag.pk}
        )

    def _request_test_document_tag_list_api_view(self):
        return self.get(
            viewname='rest_api:document-tag-list', kwargs={
                'document_id': self._test_document.pk
            }
        )

    def _request_test_document_tag_remove_api_view(self):
        return self.post(
            viewname='rest_api:document-tag-remove', kwargs={
                'document_id': self._test_document.pk
            }, data={'tag': self._test_tag.pk}
        )

    def _request_test_tag_create_api_view(self):
        self._test_object_track()

        response = self.post(
            viewname='rest_api:tag-list', data={
                'label': TEST_TAG_LABEL, 'color': TEST_TAG_COLOR
            }
        )

        self._test_object_set()

        return response

    def _request_test_tag_delete_api_view(self):
        return self.delete(
            viewname='rest_api:tag-detail',
            kwargs={'tag_id': self._test_tag.pk}
        )

    def _request_test_tag_detail_api_view(self):
        return self.get(
            viewname='rest_api:tag-detail',
            kwargs={'tag_id': self._test_tag.pk}
        )

    def _request_test_tag_document_list_api_view(self):
        return self.get(
            viewname='rest_api:tag-document-list', kwargs={
                'tag_id': self._test_tag.pk
            }
        )

    def _request_test_tag_edit_api_view(self, extra_data=None, verb='patch'):
        data = {
            'color': TEST_TAG_COLOR_EDITED,
            'label': TEST_TAG_LABEL_EDITED
        }

        if extra_data:
            data.update(extra_data)

        return getattr(self, verb)(
            viewname='rest_api:tag-detail',
            kwargs={'tag_id': self._test_tag.pk},
            data=data
        )

    def _request_test_tag_list_api_view(self):
        return self.get(viewname='rest_api:tag-list')


class TagViewTestMixin(TagTestMixin):
    def _request_test_tag_create_view(self):
        self._test_object_track()

        response = self.post(
            viewname='tags:tag_create', data={
                'label': TEST_TAG_LABEL,
                'color': TEST_TAG_COLOR
            }
        )

        self._test_object_set()

        return response

    def _request_test_tag_delete_view(self):
        return self.post(
            viewname='tags:tag_single_delete', kwargs={
                'tag_id': self._test_tag.pk
            }
        )

    def _request_test_tag_multiple_delete_view(self):
        return self.post(
            viewname='tags:tag_multiple_delete', data={
                'id_list': self._test_tag.pk
            }
        )

    def _request_test_tag_edit_view(self):
        return self.post(
            viewname='tags:tag_edit', kwargs={
                'tag_id': self._test_tag.pk
            }, data={
                'color': TEST_TAG_COLOR_EDITED,
                'label': TEST_TAG_LABEL_EDITED
            }
        )

    def _request_test_tag_list_view(self):
        return self.get(viewname='tags:tag_list')


class TaggedDocumentUploadWizardStepViewTestMixin(
    TagTestMixin, WebFormSourceTestMixin, SourceActionViewTestMixin
):
    def _request_test_source_document_upload_post_view_with_tags(self):
        tag_pk_list = list(
            Tag.objects.values_list('pk', flat=True)
        )

        return self._request_test_source_document_upload_post_view(
            extra_data={'tags': tag_pk_list}
        )
