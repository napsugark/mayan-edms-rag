class DocumentDownloadViewTestMixin:
    def _request_test_document_download_get_view(self):
        return self.get(
            viewname='document_downloads:document_download_single', kwargs={
                'document_id': self._test_document.pk
            }
        )

    def _request_test_document_download_post_view(self):
        data = {}
        for test_document_file, test_document_file_index in enumerate(iterable=self._test_document_file_list):
            data.update(
                {
                    'form-{}-document_file_id'.format(
                        test_document_file_index
                    ): self._test_document_file.pk,
                    'form-{}-include'.format(test_document_file_index): True
                }
            )

        return self.post(
            viewname='document_downloads:document_download_single', kwargs={
                'document_id': self._test_document.pk
            }, data={
                'form-0-document_file_id': self._test_document_file.pk,
                'form-0-include': True,
                'form-TOTAL_FORMS': len(self._test_document_file_list),
                'form-INITIAL_FORMS': '0',
                'form-MAX_NUM_FORMS': ''
            }
        )

    def _request_test_document_multiple_download_get_view(self):
        return self.get(
            viewname='document_downloads:document_download_multiple', query={
                'id_list': ','.join(self._test_document_id_list_string)
            }
        )

    def _request_test_document_multiple_download_post_view(self):
        data = {}
        for test_document_file, test_document_file_index in enumerate(iterable=self._test_document_file_list):
            data.update(
                {
                    'form-{}-document_file_id'.format(
                        test_document_file_index
                    ): self._test_document_file.pk,
                    'form-{}-include'.format(test_document_file_index): True
                }
            )

        return self.post(
            viewname='document_downloads:document_download_multiple', query={
                'id_list': ','.join(self._test_document_id_list_string)
            }, data={
                'form-0-document_file_id': self._test_document_file.pk,
                'form-0-include': True,
                'form-TOTAL_FORMS': len(self._test_document_file_list),
                'form-INITIAL_FORMS': '0',
                'form-MAX_NUM_FORMS': ''
            }
        )


class DocumentFileDownloadAPIViewTestMixin:
    def _request_test_document_file_download_api_view(self):
        return self.get(
            viewname='rest_api:documentfile-download', kwargs={
                'document_id': self._test_document.pk,
                'document_file_id': self._test_document.file_latest.pk
            }
        )


class DocumentFileDownloadViewTestMixin:
    def _request_test_document_file_download_view(self):
        return self.get(
            viewname='document_downloads:document_file_download', kwargs={
                'document_file_id': self._test_document_file.pk
            }
        )
