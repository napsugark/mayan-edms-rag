from django.db import migrations


def code_document_export_event_and_permission_update(apps, schema_editor):
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
        verb='documents.document_version_exported'
    ).update(
        verb='document_exports.document_version_exported'
    )

    StoredEventType.objects.using(
        alias=schema_editor.connection.alias
    ).filter(
        name='documents.document_version_exported'
    ).delete()

    stored_permission, created = StoredPermission.objects.using(
        alias=schema_editor.connection.alias
    ).get_or_create(
        namespace='document_exports', name='document_version_export'
    )

    AccessControlList.permissions.through.objects.using(
        alias=schema_editor.connection.alias
    ).filter(
        storedpermission__namespace='documents',
        storedpermission__name='document_version_export'
    ).update(storedpermission_id=stored_permission.pk)

    Role.permissions.through.objects.using(
        alias=schema_editor.connection.alias
    ).filter(
        storedpermission__namespace='documents',
        storedpermission__name='document_version_export'
    ).update(
        storedpermission_id=stored_permission.pk
    )

    StoredPermission.objects.using(
        alias=schema_editor.connection.alias
    ).filter(
        namespace='documents', name='document_version_export'
    ).delete()


def code_document_export_event_and_permission_update_reverse(apps, schema_editor):
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
        verb='document_exports.document_version_exported'
    ).update(
        verb='documents.document_version_exported'
    )

    StoredEventType.objects.using(
        alias=schema_editor.connection.alias
    ).filter(
        name='document_exports.document_version_exported'
    ).delete()

    stored_permission = StoredPermission.objects.using(
        alias=schema_editor.connection.alias
    ).create(
        namespace='documents', name='document_version_export'
    )

    AccessControlList.permissions.through.objects.using(
        alias=schema_editor.connection.alias
    ).filter(
        storedpermission__namespace='document_exports',
        storedpermission__name='document_version_export'
    ).update(storedpermission_id=stored_permission.pk)

    Role.permissions.through.objects.using(
        alias=schema_editor.connection.alias
    ).filter(
        storedpermission__namespace='document_exports',
        storedpermission__name='document_version_export'
    ).update(
        storedpermission_id=stored_permission.pk
    )

    StoredPermission.objects.using(
        alias=schema_editor.connection.alias
    ).filter(
        namespace='document_exports', name='document_version_export'
    ).delete()


class Migration(migrations.Migration):
    dependencies = [
        ('acls', '0004_auto_20210130_0322'),
        ('documents', '0080_populate_file_size'),
        ('events', '0009_alter_objecteventsubscription_options'),
        ('permissions', '0004_auto_20191213_0044')
    ]

    operations = [
        migrations.RunPython(
            code=code_document_export_event_and_permission_update,
            reverse_code=code_document_export_event_and_permission_update_reverse
        )
    ]
