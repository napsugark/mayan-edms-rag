from django.utils.translation import gettext_lazy as _


class CommentBusinessLogicMixin:
    def get_user_label(self):
        if self.user.get_full_name():
            return self.user.get_full_name()
        else:
            return self.user.username
    get_user_label.short_description = _(message='User')
