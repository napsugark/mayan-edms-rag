from django.db.models import Q

from mayan.apps.testing.tests.mixins import TestMixinObjectCreationTrack

from ..models import SmartLink, SmartLinkCondition

from .literals import (
    TEST_SMART_LINK_CONDITION_EXPRESSION,
    TEST_SMART_LINK_CONDITION_EXPRESSION_EDITED,
    TEST_SMART_LINK_CONDITION_FOREIGN_DOCUMENT_DATA,
    TEST_SMART_LINK_CONDITION_INCLUSION, TEST_SMART_LINK_CONDITION_OPERATOR,
    TEST_SMART_LINK_DYNAMIC_LABEL, TEST_SMART_LINK_LABEL,
    TEST_SMART_LINK_LABEL_EDITED
)


class SmartLinkTestMixin(TestMixinObjectCreationTrack):
    _test_object_model = SmartLink
    _test_object_name = '_test_smart_link'

    def _create_test_smart_link(self, add_test_document_type=False):
        self._test_smart_link = SmartLink.objects.create(
            label=TEST_SMART_LINK_LABEL,
            dynamic_label=TEST_SMART_LINK_DYNAMIC_LABEL
        )
        if add_test_document_type:
            self._test_smart_link.document_types.add(self._test_document_type)

    def _create_test_smart_link_condition(self):
        self._test_smart_link_condition = SmartLinkCondition.objects.create(
            smart_link=self._test_smart_link,
            foreign_document_data=TEST_SMART_LINK_CONDITION_FOREIGN_DOCUMENT_DATA,
            expression=TEST_SMART_LINK_CONDITION_EXPRESSION,
            operator=TEST_SMART_LINK_CONDITION_OPERATOR
        )


class DocumentTypeAddRemoveSmartLinkViewTestMixin(SmartLinkTestMixin):
    def _request_test_document_type_smart_link_add_remove_get_view(self):
        return self.get(
            viewname='linking:document_type_smart_links', kwargs={
                'document_type_id': self._test_document_type.pk
            }
        )

    def _request_test_document_type_smart_link_add_view(self):
        return self.post(
            viewname='linking:document_type_smart_links', kwargs={
                'document_type_id': self._test_document_type.pk
            }, data={
                'available-selection': self._test_smart_link.pk,
                'available-submit': 'true'
            }
        )

    def _request_test_document_type_smart_link_remove_view(self):
        return self.post(
            viewname='linking:document_type_smart_links', kwargs={
                'document_type_id': self._test_document_type.pk
            }, data={
                'added-selection': self._test_smart_link.pk,
                'added-submit': 'true'
            }
        )


class ResolvedSmartLinkAPIViewTestMixin(SmartLinkTestMixin):
    def _request_resolved_smart_link_detail_api_view(self):
        return self.get(
            viewname='rest_api:resolvedsmartlink-detail',
            kwargs={
                'document_id': self._test_document_list[0].pk,
                'resolved_smart_link_id': self._test_smart_link.pk
            }
        )
        self._create_test_smart_link(add_test_document_type=True)

    def _request_resolved_smart_link_list_api_view(self):
        return self.get(
            viewname='rest_api:resolvedsmartlink-list', kwargs={
                'document_id': self._test_document_list[0].pk
            }
        )

    def _request_resolved_smart_link_document_list_api_view(self):
        return self.get(
            viewname='rest_api:resolvedsmartlinkdocument-list',
            kwargs={
                'document_id': self._test_document_list[0].pk,
                'resolved_smart_link_id': self._test_smart_link.pk
            }
        )


class SmartLinkAPIViewTestMixin(SmartLinkTestMixin):
    def _request_test_smart_link_create_api_view(self):
        self._test_object_track()

        response = self.post(
            viewname='rest_api:smartlink-list', data={
                'label': TEST_SMART_LINK_LABEL
            }
        )

        self._test_object_set()

        return response

    def _request_test_smart_link_delete_api_view(self):
        return self.delete(
            viewname='rest_api:smartlink-detail', kwargs={
                'smart_link_id': self._test_smart_link.pk
            }
        )

    def _request_test_smart_link_detail_api_view(self):
        return self.get(
            viewname='rest_api:smartlink-detail', kwargs={
                'smart_link_id': self._test_smart_link.pk
            }
        )

    def _request_test_smart_link_edit_patch_api_view(self):
        return self.patch(
            viewname='rest_api:smartlink-detail',
            kwargs={'smart_link_id': self._test_smart_link.pk}, data={
                'label': TEST_SMART_LINK_LABEL_EDITED
            }
        )

    def _request_test_smart_link_edit_put_api_view(self):
        return self.put(
            viewname='rest_api:smartlink-detail',
            kwargs={'smart_link_id': self._test_smart_link.pk}, data={
                'label': TEST_SMART_LINK_LABEL_EDITED
            }
        )

    def _request_test_smart_link_list_api_view(self):
        return self.get(viewname='rest_api:smartlink-list')


class SmartLinkConditionAPIViewTestMixin(SmartLinkTestMixin):
    def _request_smart_link_condition_create_api_view(self):
        pk_list = list(
            SmartLinkCondition.objects.values_list('pk', flat=True)
        )

        response = self.post(
            viewname='rest_api:smartlinkcondition-list',
            kwargs={'smart_link_id': self._test_smart_link.pk}, data={
                'expression': TEST_SMART_LINK_CONDITION_EXPRESSION,
                'foreign_document_data': TEST_SMART_LINK_CONDITION_FOREIGN_DOCUMENT_DATA,
                'operator': TEST_SMART_LINK_CONDITION_OPERATOR
            }
        )

        try:
            self._test_smart_link_condition = SmartLinkCondition.objects.get(
                ~Q(pk__in=pk_list)
            )
        except SmartLinkCondition.DoesNotExist:
            self._test_smart_link_condition = None

        return response

    def _request_smart_link_condition_delete_api_view(self):
        return self.delete(
            viewname='rest_api:smartlinkcondition-detail',
            kwargs={
                'smart_link_condition_id': self._test_smart_link_condition.pk,
                'smart_link_id': self._test_smart_link.pk
            }
        )

    def _request_smart_link_condition_detail_api_view(self):
        return self.get(
            viewname='rest_api:smartlinkcondition-detail',
            kwargs={
                'smart_link_condition_id': self._test_smart_link_condition.pk,
                'smart_link_id': self._test_smart_link.pk
            }
        )

    def _request_smart_link_condition_edit_via_patch_api_view(self):
        return self.patch(
            viewname='rest_api:smartlinkcondition-detail',
            kwargs={
                'smart_link_condition_id': self._test_smart_link_condition.pk,
                'smart_link_id': self._test_smart_link.pk
            }, data={
                'expression': TEST_SMART_LINK_CONDITION_EXPRESSION_EDITED
            }
        )

    def _request_smart_link_condition_edit_via_put_api_view(self):
        return self.put(
            viewname='rest_api:smartlinkcondition-detail',
            kwargs={
                'smart_link_condition_id': self._test_smart_link_condition.pk,
                'smart_link_id': self._test_smart_link.pk
            }, data={
                'expression': TEST_SMART_LINK_CONDITION_EXPRESSION_EDITED,
                'foreign_document_data': TEST_SMART_LINK_CONDITION_FOREIGN_DOCUMENT_DATA,
                'operator': TEST_SMART_LINK_CONDITION_OPERATOR

            }
        )

    def _request_smart_link_condition_list_api_view(self):
        return self.get(
            viewname='rest_api:smartlinkcondition-list', kwargs={
                'smart_link_id': self._test_smart_link.pk
            }
        )


class SmartLinkConditionViewTestMixin(SmartLinkTestMixin):
    def _request_test_smart_link_condition_create_view(self):
        pk_list = list(
            SmartLinkCondition.objects.values_list('pk', flat=True)
        )

        response = self.post(
            viewname='linking:smart_link_condition_create', kwargs={
                'smart_link_id': self._test_smart_link.pk
            }, data={
                'foreign_document_data': TEST_SMART_LINK_CONDITION_FOREIGN_DOCUMENT_DATA,
                'expression': TEST_SMART_LINK_CONDITION_EXPRESSION,
                'inclusion': TEST_SMART_LINK_CONDITION_INCLUSION,
                'operator': TEST_SMART_LINK_CONDITION_OPERATOR
            }
        )

        try:
            self._test_smart_link_condition = SmartLinkCondition.objects.get(
                ~Q(pk__in=pk_list)
            )
        except SmartLinkCondition.DoesNotExist:
            self._test_smart_link_condition = None

        return response

    def _request_test_smart_link_condition_delete_view(self):
        return self.post(
            viewname='linking:smart_link_condition_delete', kwargs={
                'smart_link_condition_id': self._test_smart_link_condition.pk
            }
        )

    def _request_test_smart_link_condition_edit_view(self):
        return self.post(
            viewname='linking:smart_link_condition_edit', kwargs={
                'smart_link_condition_id': self._test_smart_link_condition.pk
            }, data={
                'foreign_document_data': TEST_SMART_LINK_CONDITION_FOREIGN_DOCUMENT_DATA,
                'expression': TEST_SMART_LINK_CONDITION_EXPRESSION_EDITED,
                'inclusion': TEST_SMART_LINK_CONDITION_INCLUSION,
                'operator': TEST_SMART_LINK_CONDITION_OPERATOR
            }
        )

    def _request_test_smart_link_condition_list_view(self):
        return self.get(
            viewname='linking:smart_link_condition_list', kwargs={
                'smart_link_id': self._test_smart_link.pk
            }
        )


class SmartLinkDocumentTypeAPIViewTestMixin(SmartLinkTestMixin):
    def _request_test_smart_link_document_type_add_api_view(self):
        return self.post(
            viewname='rest_api:smartlink-document_type-add',
            kwargs={'smart_link_id': self._test_smart_link.pk}, data={
                'document_type': self._test_document_type.pk
            }
        )

    def _request_test_smart_link_document_type_list_api_view(self):
        return self.get(
            viewname='rest_api:smartlink-document_type-list', kwargs={
                'smart_link_id': self._test_smart_link.pk
            }
        )

    def _request_test_smart_link_document_type_remove_api_view(self):
        return self.post(
            viewname='rest_api:smartlink-document_type-remove',
            kwargs={'smart_link_id': self._test_smart_link.pk}, data={
                'document_type': self._test_document_type.pk
            }
        )


class SmartLinkDocumentTypeViewTestMixin(SmartLinkTestMixin):
    def _request_test_smart_link_document_type_add_remove_get_view(self):
        return self.get(
            viewname='linking:smart_link_document_types', kwargs={
                'smart_link_id': self._test_smart_link.pk
            }
        )

    def _request_test_smart_link_document_type_add_view(self):
        return self.post(
            viewname='linking:smart_link_document_types', kwargs={
                'smart_link_id': self._test_smart_link.pk
            }, data={
                'available-selection': self._test_document_type.pk,
                'available-submit': 'true'
            }
        )

    def _request_test_smart_link_document_type_remove_view(self):
        return self.post(
            viewname='linking:smart_link_document_types', kwargs={
                'smart_link_id': self._test_smart_link.pk
            }, data={
                'added-selection': self._test_document_type.pk,
                'added-submit': 'true'
            }
        )


class ResolvedSmartLinkDocumentViewTestMixin(SmartLinkTestMixin):
    def _request_test_document_resolved_smart_link_document_list_view(self):
        return self.get(
            viewname='linking:smart_link_instance_view', kwargs={
                'document_id': self._test_document_list[0].pk,
                'smart_link_id': self._test_smart_link.pk
            }
        )

    def _request_test_document_resolved_smart_link_list_view(self):
        return self.get(
            viewname='linking:document_smart_link_instance_list', kwargs={
                'document_id': self._test_document_list[0].pk
            }
        )


class SmartLinkViewTestMixin(SmartLinkTestMixin):
    def _request_test_smart_link_create_view(self):
        self._test_object_track()

        response = self.post(
            viewname='linking:smart_link_create', data={
                'label': TEST_SMART_LINK_LABEL
            }
        )

        self._test_object_set()

        return response

    def _request_test_smart_link_delete_view(self):
        return self.post(
            viewname='linking:smart_link_delete', kwargs={
                'smart_link_id': self._test_smart_link.pk
            }
        )

    def _request_test_smart_link_edit_view(self):
        return self.post(
            viewname='linking:smart_link_edit', kwargs={
                'smart_link_id': self._test_smart_link.pk
            }, data={
                'label': TEST_SMART_LINK_LABEL_EDITED
            }
        )

    def _request_test_smart_link_list_view(self):
        return self.get(viewname='linking:smart_link_list')
