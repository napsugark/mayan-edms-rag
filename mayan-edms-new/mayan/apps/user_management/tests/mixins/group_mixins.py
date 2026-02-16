from django.contrib.auth.models import Group

from ...links import link_group_setup

from ..literals import TEST_GROUP_NAME, TEST_GROUP_NAME_EDITED


class GroupAPIViewTestMixin:
    def _request_test_group_create_api_view(self):
        result = self.post(
            viewname='rest_api:group-list', data={
                'name': TEST_GROUP_NAME
            }
        )
        if 'id' in result.json():
            self._test_group = Group.objects.get(
                pk=result.json()['id']
            )

        return result

    def _request_test_group_delete_api_view(self):
        return self.delete(
            viewname='rest_api:group-detail', kwargs={
                'group_id': self._test_group.pk
            }
        )

    def _request_test_group_detail_api_view(self):
        return self.get(
            viewname='rest_api:group-detail', kwargs={
                'group_id': self._test_group.pk
            }
        )

    def _request_test_group_edit_patch_api_view(self):
        return self.patch(
            viewname='rest_api:group-detail', kwargs={
                'group_id': self._test_group.pk
            }, data={
                'name': TEST_GROUP_NAME_EDITED
            }
        )

    def _request_test_group_edit_put_api_view(self):
        return self.put(
            viewname='rest_api:group-detail', kwargs={
                'group_id': self._test_group.pk
            }, data={
                'name': TEST_GROUP_NAME_EDITED
            }
        )

    def _request_test_group_list_api_view(self):
        return self.get(viewname='rest_api:group-list')


class GroupLinkTestMixin:
    def _resolve_group_setup_link(self):
        self.add_test_view()
        context = self.get_test_view()
        return link_group_setup.resolve(context=context)


class GroupTestMixin:
    def setUp(self):
        super().setUp()
        self._test_group_list = []

    def _create_test_group(self, add_users=None):
        total_test_group_count = len(self._test_group_list)
        name = '{}_{}'.format(TEST_GROUP_NAME, total_test_group_count)

        self._test_group = Group.objects.create(name=name)

        self._test_group_list.append(self._test_group)

        for user in add_users or []:
            self._test_group.user_set.add(user)

    def _edit_test_group(self):
        self._test_group.name = TEST_GROUP_NAME_EDITED
        self._test_group.save()


class GroupUserAPIViewTestMixin:
    def _request_test_group_user_add_api_view(self):
        return self.post(
            viewname='rest_api:group-user-add', kwargs={
                'group_id': self._test_group.pk
            }, data={'user': self._test_user.pk}
        )

    def _request_test_group_user_list_api_view(self):
        return self.get(
            viewname='rest_api:group-user-list', kwargs={
                'group_id': self._test_group.pk
            }
        )

    def _request_test_group_user_remove_api_view(self):
        return self.post(
            viewname='rest_api:group-user-remove', kwargs={
                'group_id': self._test_group.pk
            }, data={'user': self._test_user.pk}
        )


class GroupUserViewTestMixin:
    def _request_test_group_user_add_remove_get_view(self):
        return self.get(
            viewname='user_management:group_members', kwargs={
                'group_id': self._test_group.pk
            }
        )

    def _request_test_group_user_add_view(self):
        return self.post(
            viewname='user_management:group_members', kwargs={
                'group_id': self._test_group.pk
            }, data={
                'available-submit': 'true',
                'available-selection': self._test_user.pk
            }
        )

    def _request_test_group_user_remove_view(self):
        return self.post(
            viewname='user_management:group_members', kwargs={
                'group_id': self._test_group.pk
            }, data={
                'added-submit': 'true',
                'added-selection': self._test_user.pk
            }
        )


class GroupViewTestMixin:
    def _request_test_group_create_view(self):
        response = self.post(
            viewname='user_management:group_create', data={
                'name': TEST_GROUP_NAME
            }
        )
        self._test_group = Group.objects.filter(name=TEST_GROUP_NAME).first()
        return response

    def _request_test_group_single_delete_view(self):
        return self.post(
            viewname='user_management:group_single_delete', kwargs={
                'group_id': self._test_group.pk
            }
        )

    def _request_test_group_multiple_delete_view(self):
        return self.post(
            viewname='user_management:group_multiple_delete', data={
                'id_list': self._test_group.pk
            }
        )

    def _request_test_group_edit_view(self):
        return self.post(
            viewname='user_management:group_edit', kwargs={
                'group_id': self._test_group.pk
            }, data={
                'name': TEST_GROUP_NAME_EDITED
            }
        )

    def _request_test_group_list_view(self):
        return self.get(viewname='user_management:group_list')

    def _request_test_group_members_view(self):
        return self.get(
            viewname='user_management:group_members', kwargs={
                'group_id': self._test_group.pk
            }
        )


class UserGroupAPIViewTestMixin:
    def _request_test_user_group_list_api_view(self):
        return self.get(
            viewname='rest_api:user-group-list', kwargs={
                'user_id': self._test_user.pk
            }
        )


class UserGroupViewTestMixin:
    def _request_test_user_group_add_remove_get_view(self):
        return self.get(
            viewname='user_management:user_groups', kwargs={
                'user_id': self._test_user.pk
            }
        )

    def _request_test_user_group_add_view(self):
        return self.post(
            viewname='user_management:user_groups', kwargs={
                'user_id': self._test_user.pk
            }, data={
                'available-submit': 'true',
                'available-selection': self._test_group.pk
            }
        )

    def _request_test_user_group_remove_view(self):
        return self.post(
            viewname='user_management:user_groups', kwargs={
                'user_id': self._test_user.pk
            }, data={
                'added-submit': 'true',
                'added-selection': self._test_group.pk
            }
        )
