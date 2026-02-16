MAILER_BACKEND_DJANGO_FILE_PATH = 'mayan.apps.mailer.mailers.DjangoFileBased'
MAILER_BACKEND_DJANGO_SMTP_PATH = 'mayan.apps.mailer.mailers.DjangoSMTP'

TEST_DJANGO_FILE_PATH = '/tmp/email'

TEST_EMAIL_ADDRESS = 'test@example.com'
TEST_EMAIL_BODY = 'test body'
TEST_EMAIL_BODY_HTML = '<strong>test body</strong>'
TEST_EMAIL_FROM_ADDRESS = 'from.test@example.com'
TEST_EMAIL_SUBJECT = 'test subject'

TEST_RECIPIENTS_MULTIPLE_COMMA = 'test@example.com,test2@example.com'
TEST_RECIPIENTS_MULTIPLE_COMMA_RESULT = [
    'test@example.com', 'test2@example.com'
]
TEST_RECIPIENTS_MULTIPLE_SEMICOLON = 'test@example.com;test2@example.com'
TEST_RECIPIENTS_MULTIPLE_SEMICOLON_RESULT = [
    'test@example.com', 'test2@example.com'
]
TEST_RECIPIENTS_MULTIPLE_MIXED = 'test@example.com,test2@example.com;test2@example.com'
TEST_RECIPIENTS_MULTIPLE_MIXED_RESULT = [
    'test@example.com', 'test2@example.com', 'test2@example.com'
]

TEST_MAILING_OBJECT_CONTENT = b'test_content'
TEST_MAILING_OBJECT_MIME_TYPE = 'test/mime_type'

TEST_MAILING_PROFILE_BACKEND_PATH = 'mayan.apps.mailer.tests.mailers.MailingProfileTest'
TEST_MAILING_PROFILE_BACKEND_CREDENTIALS_PATH = 'mayan.apps.mailer.tests.mailers.MailingProfileCredentialTest'

TEST_MAILING_PROFILE_LABEL = 'test mailing profile label'
TEST_MAILING_PROFILE_LABEL_EDITED = 'test mailing profile label edited'
