from django.db import migrations

from mayan.apps.databases.literals import (
    DJANGO_POSITIVE_INTEGER_FIELD_MAX_VALUE
)


def code_document_file_size_copy(apps, schema_editor):
    DocumentFile = apps.get_model(
        app_label='documents', model_name='DocumentFile'
    )

    for document_file in DocumentFile.objects.iterator():
        name = document_file.file.name
        document_file.file.close()

        if document_file.file.storage.exists(name=name):
            document_file_size = document_file.file.storage.size(name=name)
            if document_file_size > DJANGO_POSITIVE_INTEGER_FIELD_MAX_VALUE:
                document_file_size = DJANGO_POSITIVE_INTEGER_FIELD_MAX_VALUE

            document_file.size = document_file_size
            document_file.save(
                update_fields=('size',)
            )


class Migration(migrations.Migration):
    dependencies = [
        ('documents', '0079_documentfile_size')
    ]

    operations = [
        migrations.RunPython(
            code=code_document_file_size_copy,
            reverse_code=migrations.RunPython.noop
        )
    ]
