from django.db import migrations

PATH_MAPPING = {
    'mayan.apps.credentials_google.credential_backends.CredentialBackendGoogleServiceAccount': 'mayan.apps.credentials.credential_backends.CredentialBackendGoogleServiceAccount'
}


def code_path_update(apps, schema_editor):
    StoredCredential = apps.get_model(
        app_label='credentials', model_name='StoredCredential'
    )

    for key, value in PATH_MAPPING.items():
        queryset = StoredCredential.objects.using(
            alias=schema_editor.connection.alias
        ).filter(backend_path=key)

        queryset.update(backend_path=value)


def reverse_code_path_update(apps, schema_editor):
    StoredCredential = apps.get_model(
        app_label='credentials', model_name='StoredCredential'
    )

    for key, value in PATH_MAPPING.items():
        queryset = StoredCredential.objects.using(
            alias=schema_editor.connection.alias
        ).filter(backend_path=value)

        queryset.update(backend_path=key)


class Migration(migrations.Migration):
    dependencies = [
        ('credentials', '0005_auto_20210207_0840')
    ]

    operations = [
        migrations.RunPython(
            code=code_path_update, reverse_code=reverse_code_path_update
        )
    ]
