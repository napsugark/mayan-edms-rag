from django.urls import re_path

from .views import (
    UserOTPDataDetailView, UserOTPDataDisableView, UserOTPDataEnableView,
    UserOTPDataVerifyTokenView
)

urlpatterns = [
    re_path(
        route=r'^otp/$', name='otp_detail',
        view=UserOTPDataDetailView.as_view()
    ),
    re_path(
        route=r'^otp/disable/$', name='otp_disable',
        view=UserOTPDataDisableView.as_view()
    ),
    re_path(
        route=r'^otp/enable/$', name='otp_enable',
        view=UserOTPDataEnableView.as_view()
    ),
    re_path(
        route=r'^otp/verify/$', name='otp_verify',
        view=UserOTPDataVerifyTokenView.as_view()
    )
]
