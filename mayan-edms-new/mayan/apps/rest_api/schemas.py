from django.utils.translation import gettext_lazy as _

from drf_yasg import openapi

import mayan

from .literals import API_VERSION

openapi_info_default_version = 'v{}'.format(API_VERSION)
openapi_info_title = _(message='%s API') % mayan.__title__
openapi_license = openapi.License(name=mayan.__license__)

openapi_info = openapi.Info(
    default_version=openapi_info_default_version,
    description=mayan.__description__, license=openapi_license,
    title=openapi_info_title
)
