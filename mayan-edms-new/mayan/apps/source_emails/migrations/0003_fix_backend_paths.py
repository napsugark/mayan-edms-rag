from django.db import migrations

SOURCE_BACKEND_MAPPING = {
    'mayan.apps.source_emails.source_backends.email_backends.SourceBackendIMAPEmail': 'mayan.apps.source_emails.source_backends.imap_source_backends.SourceBackendIMAPEmail',
    'mayan.apps.source_emails.source_backends.email_backends.SourceBackendPOP3Email': 'mayan.apps.source_emails.source_backends.pop3_source_backends.SourceBackendPOP3Email'
}


def code_source_backend_path_update(apps, schema_editor):
    Source = apps.get_model(app_label='sources', model_name='Source')

    for key, value in SOURCE_BACKEND_MAPPING.items():
        queryset = Source.objects.using(
            alias=schema_editor.connection.alias
        ).filter(backend_path=key)

        queryset.update(backend_path=value)


def reverse_code_source_backend_path_update(apps, schema_editor):
    Source = apps.get_model(app_label='sources', model_name='Source')

    for key, value in SOURCE_BACKEND_MAPPING.items():
        queryset = Source.objects.using(
            alias=schema_editor.connection.alias
        ).filter(backend_path=value)

        queryset.update(backend_path=key)


class Migration(migrations.Migration):
    dependencies = [
        ('source_emails', '0002_migrate_to_credentials')
    ]

    operations = [
        migrations.RunPython(
            code=code_source_backend_path_update,
            reverse_code=reverse_code_source_backend_path_update
        )
    ]
