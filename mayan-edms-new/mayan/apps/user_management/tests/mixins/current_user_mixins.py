class CurrentUserViewTestMixin:
    def _request_current_user_details_view(self, user=None):
        user = user or self._test_case_user

        return self.get(
            viewname='user_management:user_details', kwargs={
                'user_id': user.pk
            }
        )

    def _request_current_user_edit_view(self, user=None):
        user = user or self._test_case_user

        return self.get(
            viewname='user_management:user_edit', kwargs={
                'user_id': user.pk
            }
        )

    def _request_current_user_post_view(self, user=None):
        user = user or self._test_case_user

        return self.post(
            viewname='user_management:user_edit', kwargs={
                'user_id': user.pk
            }, data={
                'username': 'new_username', 'first_name': 'first name edited'
            }
        )
