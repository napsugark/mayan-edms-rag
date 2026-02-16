from django.apps import apps

from .settings import (
    setting_user_language_default, setting_user_timezone_default
)


def handler_user_locale_profile_create(sender, instance, created, **kwargs):
    UserLocaleProfile = apps.get_model(
        app_label='locales', model_name='UserLocaleProfile'
    )

    if created:
        UserLocaleProfile.objects.create(
            language=setting_user_language_default.value,
            timezone=setting_user_timezone_default.value, user=instance
        )
