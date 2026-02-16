from collections import defaultdict

from django.db import migrations

from mayan.apps.common.utils import convert_to_internal_name

BULK_UPDATE_SIZE = 100
ITERATOR_CHUNK_SIZE = 1000


def code_populate_internal_name(apps, schema_editor):
    FileMetadataEntry = apps.get_model(
        app_label='file_metadata', model_name='FileMetadataEntry'
    )

    manager_file_metadata_entry = FileMetadataEntry.objects.using(
        alias=schema_editor.connection.alias
    )

    queryset_file_metadata_entry = (
        manager_file_metadata_entry.only(
            'pk', 'document_file_driver_entry_id', 'key', 'internal_name'
        ).order_by(
            'document_file_driver_entry_id', 'key', 'pk'
        )
    )

    document_file_driver_entry_id = None
    list_to_update = []
    counters = defaultdict(int)

    for file_metadata_entry in queryset_file_metadata_entry.iterator(chunk_size=ITERATOR_CHUNK_SIZE):
        if file_metadata_entry.document_file_driver_entry_id != document_file_driver_entry_id:
            document_file_driver_entry_id = file_metadata_entry.document_file_driver_entry_id
            counters.clear()

        internal_name = convert_to_internal_name(
            value=file_metadata_entry.key
        )

        index = counters[internal_name]

        if index == 0:
            internal_name_final = internal_name
        else:
            internal_name_final = f'{internal_name}_{index}'

        counters[internal_name] = index + 1

        file_metadata_entry.internal_name = internal_name_final

        list_to_update.append(file_metadata_entry)

        if len(list_to_update) >= BULK_UPDATE_SIZE:
            manager_file_metadata_entry.bulk_update(
                list_to_update, ['internal_name'], batch_size=BULK_UPDATE_SIZE
            )
            list_to_update.clear()

    if list_to_update:
        manager_file_metadata_entry.bulk_update(
            list_to_update, ['internal_name'], batch_size=BULK_UPDATE_SIZE
        )


class Migration(migrations.Migration):
    dependencies = [
        ('file_metadata', '0010_add_internal_name')
    ]

    operations = [
        migrations.RunPython(
            code=code_populate_internal_name,
            reverse_code=migrations.RunPython.noop
        )
    ]
