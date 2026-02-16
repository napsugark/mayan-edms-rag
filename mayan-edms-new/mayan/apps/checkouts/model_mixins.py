from django.utils.translation import gettext_lazy as _


class CheckedOutDocumentBusinessLogicMixin:
    def get_user_display(self):
        check_out_info = self.get_check_out_info()
        return check_out_info.user.get_full_name() or check_out_info.user

    get_user_display.short_description = _(message='User')

    def get_checkout_datetime(self):
        return self.get_check_out_info().checkout_datetime

    get_checkout_datetime.short_description = _(message='Checkout time and date')

    def get_checkout_expiration(self):
        return self.get_check_out_info().expiration_datetime

    get_checkout_expiration.short_description = _(message='Checkout expiration')
