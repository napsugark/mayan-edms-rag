from django.db import migrations


def code_document_file_download_event_and_permission_update(apps, schema_editor):
    Action = apps.get_model(
        app_label='actstream', model_name='Action'
    )
    AccessControlList = apps.get_model(
        app_label='acls', model_name='AccessControlList'
    )
    StoredEventType = apps.get_model(
        app_label='events', model_name='StoredEventType'
    )
    StoredPermission = apps.get_model(
        app_label='permissions', model_name='StoredPermission'
    )
    Role = apps.get_model(
        app_label='permissions', model_name='Role'
    )

    Action.objects.using(
        alias=schema_editor.connection.alias
    ).filter(
        verb='documents.document_file_downloaded'
    ).update(
        verb='document_downloads.document_file_downloaded'
    )

    StoredEventType.objects.using(
        alias=schema_editor.connection.alias
    ).filter(
        name='documents.document_file_downloaded'
    ).delete()

    try:
        stored_permission = StoredPermission.objects.using(
            alias=schema_editor.connection.alias
        ).get(
            namespace='document_downloads', name='document_file_download'
        )
    except StoredPermission.DoesNotExist:
        """Raised on initial migrations. Can be ignored."""
    else:
        AccessControlList.permissions.through.objects.using(
            alias=schema_editor.connection.alias
        ).filter(
            storedpermission__namespace='documents',
            storedpermission__name='document_file_download'
        ).update(storedpermission_id=stored_permission.pk)

        Role.permissions.through.objects.using(
            alias=schema_editor.connection.alias
        ).filter(
            storedpermission__namespace='documents',
            storedpermission__name='document_file_download'
        ).update(
            storedpermission_id=stored_permission.pk
        )

    StoredPermission.objects.using(
        alias=schema_editor.connection.alias
    ).filter(
        namespace='documents', name='document_file_download'
    ).delete()


def code_document_file_download_event_and_permission_update_reverse(apps, schema_editor):
    Action = apps.get_model(
        app_label='actstream', model_name='Action'
    )
    AccessControlList = apps.get_model(
        app_label='acls', model_name='AccessControlList'
    )
    StoredEventType = apps.get_model(
        app_label='events', model_name='StoredEventType'
    )
    StoredPermission = apps.get_model(
        app_label='permissions', model_name='StoredPermission'
    )
    Role = apps.get_model(
        app_label='permissions', model_name='Role'
    )

    Action.objects.using(
        alias=schema_editor.connection.alias
    ).filter(
        verb='document_downloads.document_file_downloaded'
    ).update(
        verb='documents.document_file_downloaded'
    )

    StoredEventType.objects.using(
        alias=schema_editor.connection.alias
    ).filter(
        name='document_downloads.document_file_downloaded'
    ).delete()

    stored_permission = StoredPermission.objects.using(
        alias=schema_editor.connection.alias
    ).create(
        namespace='documents', name='document_file_download'
    )

    AccessControlList.permissions.through.objects.using(
        alias=schema_editor.connection.alias
    ).filter(
        storedpermission__namespace='document_downloads',
        storedpermission__name='document_file_download'
    ).update(storedpermission_id=stored_permission.pk)

    Role.permissions.through.objects.using(
        alias=schema_editor.connection.alias
    ).filter(
        storedpermission__namespace='document_downloads',
        storedpermission__name='document_file_download'
    ).update(
        storedpermission_id=stored_permission.pk
    )

    StoredPermission.objects.using(
        alias=schema_editor.connection.alias
    ).filter(
        namespace='document_downloads', name='document_file_download'
    ).delete()


class Migration(migrations.Migration):
    dependencies = [
        ('acls', '0004_auto_20210130_0322'),
        ('documents', '0077_favoritedocumentproxy'),
        ('events', '0008_auto_20180315_0029'),
        ('permissions', '0004_auto_20191213_0044')
    ]

    operations = [
        migrations.RunPython(
            code=code_document_file_download_event_and_permission_update,
            reverse_code=code_document_file_download_event_and_permission_update_reverse
        )
    ]
