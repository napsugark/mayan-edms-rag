from ..classes import MailerBackend, MailerBackendCredentials


class MailingProfileCredentialTest(MailerBackendCredentials):
    """
    Credential mailing profile backend to use with tests.
    """
    class_path = 'django.core.mail.backends.locmem.EmailBackend'
    label = 'Credential test local memory backend'


class MailingProfileTest(MailerBackend):
    """
    Mailing profile backend to use with tests.
    """
    class_path = 'django.core.mail.backends.locmem.EmailBackend'
    label = 'Test local memory backend'
