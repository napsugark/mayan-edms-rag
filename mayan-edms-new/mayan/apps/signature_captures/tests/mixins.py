from mayan.apps.documents.tests.mixins.document_mixins import (
    DocumentTestMixin
)

from ..models import SignatureCapture

from .literals import (
    TEST_SIGNATURE_CAPTURE_DATA, TEST_SIGNATURE_CAPTURE_INTERNAL_NAME,
    TEST_SIGNATURE_CAPTURE_SVG, TEST_SIGNATURE_CAPTURE_TEXT,
    TEST_SIGNATURE_CAPTURE_TEXT_EDITED
)


class SignatureCaptureTestMixin(DocumentTestMixin):
    _test_object_model = SignatureCapture
    _test_object_name = '_test_signature_capture'
    auto_create_test_signature_capture = False

    def setUp(self):
        super().setUp()
        self._test_signature_capture_list = []
        if self.auto_create_test_signature_capture:
            self._create_test_signature_capture()

    def _create_test_signature_capture(self):
        total_test_signature_capture_count = len(self._test_signature_capture_list)
        text = '{}_{}'.format(
            TEST_SIGNATURE_CAPTURE_TEXT, total_test_signature_capture_count
        )

        self._test_signature_capture = SignatureCapture.objects.create(
            document=self._test_document, data=TEST_SIGNATURE_CAPTURE_DATA,
            internal_name=TEST_SIGNATURE_CAPTURE_INTERNAL_NAME,
            text=text, svg=TEST_SIGNATURE_CAPTURE_SVG,
            user=self._test_case_user
        )

        self._test_signature_capture_list.append(self._test_signature_capture)


class SignatureCaptureAPIViewTestMixin(SignatureCaptureTestMixin):
    def _request_test_signature_capture_create_api_view(self):
        self._test_object_track()

        response = self.post(
            viewname='rest_api:signature_capture-list', kwargs={
                'document_id': self._test_document.pk
            }, data={
                'data': TEST_SIGNATURE_CAPTURE_DATA,
                'internal_name': TEST_SIGNATURE_CAPTURE_INTERNAL_NAME,
                'svg': TEST_SIGNATURE_CAPTURE_SVG,
                'text': TEST_SIGNATURE_CAPTURE_TEXT
            }
        )

        self._test_object_set()

        return response

    def _request_test_signature_capture_delete_api_view(self):
        return self.delete(
            viewname='rest_api:signature_capture-detail',
            kwargs={
                'document_id': self._test_document.pk,
                'signature_capture_id': self._test_signature_capture.pk
            }
        )

    def _request_test_signature_capture_detail_api_view(self):
        return self.get(
            viewname='rest_api:signature_capture-detail',
            kwargs={
                'document_id': self._test_document.pk,
                'signature_capture_id': self._test_signature_capture.pk
            }
        )

    def _request_test_signature_capture_edit_api_view(
        self, extra_data=None, verb='patch'
    ):
        data = {
            'internal_name': self._test_signature_capture.internal_name,
            'text': TEST_SIGNATURE_CAPTURE_TEXT_EDITED
        }

        if extra_data:
            data.update(extra_data)

        verb = getattr(self, verb)
        return verb(
            viewname='rest_api:signature_capture-detail', kwargs={
                'document_id': self._test_document.pk,
                'signature_capture_id': self._test_signature_capture.pk
            }, data=data
        )

    def _request_test_signature_capture_list_api_view(self):
        return self.get(
            viewname='rest_api:signature_capture-list', kwargs={
                'document_id': self._test_document.pk
            }
        )


class SignatureCaptureViewTestMixin(SignatureCaptureTestMixin):
    def _request_test_signature_capture_create_view(self):
        self._test_object_track()

        response = self.post(
            viewname='signature_captures:signature_capture_create', kwargs={
                'document_id': self._test_document.pk
            }, data={
                'data': TEST_SIGNATURE_CAPTURE_DATA,
                'internal_name': TEST_SIGNATURE_CAPTURE_INTERNAL_NAME,
                'svg': TEST_SIGNATURE_CAPTURE_SVG,
                'text': TEST_SIGNATURE_CAPTURE_TEXT
            }
        )

        self._test_object_set()

        return response

    def _request_test_signature_capture_delete_view(self):
        return self.post(
            viewname='signature_captures:signature_capture_delete',
            kwargs={
                'signature_capture_id': self._test_signature_capture.pk
            }
        )

    def _request_test_signature_capture_detail_view(self):
        return self.get(
            viewname='signature_captures:signature_capture_detail',
            kwargs={
                'signature_capture_id': self._test_signature_capture.pk
            }
        )

    def _request_test_signature_capture_edit_view(self):
        return self.post(
            viewname='signature_captures:signature_capture_edit', kwargs={
                'signature_capture_id': self._test_signature_capture.pk
            }, data={
                'data': TEST_SIGNATURE_CAPTURE_DATA,
                'internal_name': TEST_SIGNATURE_CAPTURE_INTERNAL_NAME,
                'svg': TEST_SIGNATURE_CAPTURE_SVG,
                'text': TEST_SIGNATURE_CAPTURE_TEXT_EDITED
            }
        )

    def _request_test_signature_capture_list_view(self):
        return self.get(
            viewname='signature_captures:signature_capture_list', kwargs={
                'document_id': self._test_document.pk
            }
        )
