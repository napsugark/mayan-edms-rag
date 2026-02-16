from mayan.apps.testing.tests.mixins import TestMixinObjectCreationTrack

from ..models import Comment

from .literals import TEST_COMMENT_TEXT, TEST_COMMENT_TEXT_EDITED


class DocumentCommentTestMixin(TestMixinObjectCreationTrack):
    _test_object_model = Comment
    _test_object_name = '_test_document_comment'

    def _create_test_comment(self, text=None, user=None):
        text = text or TEST_COMMENT_TEXT
        user = user or self._test_user

        self._test_document_comment = self._test_document.comments.create(
            text=text, user=user
        )


class DocumentCommentAPIViewTestMixin(DocumentCommentTestMixin):
    def _request_test_comment_create_api_view(self):
        self._test_object_track()

        response = self.post(
            viewname='rest_api:comment-list', kwargs={
                'document_id': self._test_document.pk
            }, data={
                'text': TEST_COMMENT_TEXT
            }
        )

        self._test_object_set()

        return response

    def _request_test_comment_delete_api_view(self):
        return self.delete(
            viewname='rest_api:comment-detail', kwargs={
                'document_id': self._test_document.pk,
                'comment_id': self._test_document_comment.pk
            }
        )

    def _request_test_comment_detail_api_view(self):
        return self.get(
            viewname='rest_api:comment-detail', kwargs={
                'document_id': self._test_document.pk,
                'comment_id': self._test_document_comment.pk
            }
        )

    def _request_test_comment_edit_patch_api_view(self):
        return self.patch(
            viewname='rest_api:comment-detail', kwargs={
                'document_id': self._test_document.pk,
                'comment_id': self._test_document_comment.pk
            }, data={'text': TEST_COMMENT_TEXT_EDITED}
        )

    def _request_test_comment_edit_put_api_view(self):
        return self.put(
            viewname='rest_api:comment-detail', kwargs={
                'document_id': self._test_document.pk,
                'comment_id': self._test_document_comment.pk
            }, data={'text': TEST_COMMENT_TEXT_EDITED}
        )

    def _request_test_comment_list_api_view(self):
        return self.get(
            viewname='rest_api:comment-list', kwargs={
                'document_id': self._test_document.pk
            }
        )


class DocumentCommentViewTestMixin(DocumentCommentTestMixin):
    def _request_test_comment_create_view(self):
        self._test_object_track()

        response = self.post(
            viewname='comments:comment_add', kwargs={
                'document_id': self._test_document.pk
            }, data={'text': TEST_COMMENT_TEXT}
        )

        self._test_object_set()

        return response

    def _request_test_comment_delete_view(self):
        return self.post(
            viewname='comments:comment_delete', kwargs={
                'comment_id': self._test_document_comment.pk
            }
        )

    def _request_test_comment_detail_view(self):
        return self.get(
            viewname='comments:comment_details', kwargs={
                'comment_id': self._test_document_comment.pk
            }
        )

    def _request_test_comment_edit_view(self):
        return self.post(
            viewname='comments:comment_edit', kwargs={
                'comment_id': self._test_document_comment.pk
            }, data={
                'text': TEST_COMMENT_TEXT_EDITED
            }
        )

    def _request_test_comment_list_view(self):
        return self.get(
            viewname='comments:comments_for_document', kwargs={
                'document_id': self._test_document.pk
            }
        )
