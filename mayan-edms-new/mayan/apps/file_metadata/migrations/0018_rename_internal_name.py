from django.db import migrations


EVENT_NAME_MAP = {
    'file_metadata.document_version_submit': 'file_metadata.document_file_submitted',
    'file_metadata.document_version_finish': 'file_metadata.document_file_finished'
}


def code_rename_internal_event_name(apps, schema_editor):
    Action = apps.get_model(
        app_label='actstream', model_name='Action'
    )

    for key, value in EVENT_NAME_MAP.items():
        queryset = Action.objects.filter(verb=key)
        queryset.update(verb=value)


def reverse_code_rename_internal_event_name(apps, schema_editor):
    Action = apps.get_model(
        app_label='actstream', model_name='Action'
    )

    for key, value in EVENT_NAME_MAP.items():
        queryset = Action.objects.filter(verb=value)
        queryset.update(verb=key)


class Migration(migrations.Migration):
    dependencies = [
        ('file_metadata', '0017_documenttypedriverconfiguration_arguments')
    ]

    operations = [
        migrations.RunPython(
            code=code_rename_internal_event_name,
            reverse_code=reverse_code_rename_internal_event_name
        )
    ]
