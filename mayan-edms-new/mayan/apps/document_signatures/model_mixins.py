from django.utils.translation import gettext_lazy as _

from mayan.apps.django_gpg.models import Key


class SignatureBaseModelBusinessLogicMixin:
    def get_key_id(self):
        if self.public_key_fingerprint:
            return self.public_key_fingerprint[-16:]
        else:
            return self.key_id

    def get_signature_type_display(self):
        if self.is_detached:
            return _(message='Detached')
        else:
            return _(message='Embedded')

    @property
    def is_detached(self):
        return hasattr(self, 'signature_file')

    @property
    def is_embedded(self):
        return not hasattr(self, 'signature_file')

    @property
    def key(self):
        try:
            return Key.objects.get(
                fingerprint=self.public_key_fingerprint
            )
        except Key.DoesNotExist:
            return None

    @property
    def key_algorithm(self):
        key = self.key

        if key:
            return key.algorithm

    @property
    def key_creation_date(self):
        key = self.key

        if key:
            return key.creation_date

    @property
    def key_expiration_date(self):
        key = self.key

        if key:
            return key.expiration_date

    @property
    def key_length(self):
        key = self.key

        if key:
            return key.length

    @property
    def key_type(self):
        key = self.key

        if key:
            return key.get_key_type_display()

    @property
    def key_user_id(self):
        key = self.key

        if key:
            return key.user_id
