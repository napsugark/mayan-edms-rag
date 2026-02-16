from django.apps import apps
from django.utils.translation import gettext_lazy as _

from .permissions import permission_credential_use


class BackendMixinCredentials:
    @classmethod
    def get_form_fields(cls):
        StoredCredential = apps.get_model(
            app_label='credentials', model_name='StoredCredential'
        )

        fields = super().get_form_fields()

        fields.update(
            {
                'stored_credential_id': {
                    'class': 'mayan.apps.forms.form_fields.FormFieldFilteredModelChoice',
                    'help_text': _(
                        message='The credential entry to use for '
                        'authentication.'
                    ),
                    'kwargs': {
                        'source_model': StoredCredential,
                        'permission': permission_credential_use
                    },
                    'label': _(message='Credential'),
                    'required': True
                }
            }
        )

        return fields

    @classmethod
    def get_form_fieldsets(cls):
        fieldsets = super().get_form_fieldsets()

        fieldsets += (
            (
                _(message='Authentication'), {
                    'fields': (
                        'stored_credential_id',
                    )
                },
            ),
        )

        return fieldsets

    def get_credential(self, action_object=None, user=None):
        StoredCredential = apps.get_model(
            app_label='credentials', model_name='StoredCredential'
        )

        stored_credential_id = self.kwargs.get('stored_credential_id')

        if stored_credential_id:
            stored_credential = StoredCredential.objects.get(
                pk=stored_credential_id
            )

            backend_instance = stored_credential.get_backend_instance()

            return backend_instance.get_credential(
                action_object=action_object, user=user
            )


class BackendMixinCredentialsOptional(BackendMixinCredentials):
    @classmethod
    def get_form_fields(cls):
        fields = super().get_form_fields()

        fields['stored_credential_id']['required'] = False
        fields['stored_credential_id']['help_text'] = _(
            message='Optional credential entry to use for authentication.'
        )

        return fields
