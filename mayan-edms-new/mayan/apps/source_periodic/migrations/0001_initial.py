from django.db import migrations


def code_source_task_path_update(apps, schema_editor):
    PeriodicTask = apps.get_model(
        app_label='django_celery_beat', model_name='PeriodicTask'
    )

    PeriodicTask.objects.using(
        alias=schema_editor.connection.alias
    ).filter(
        task='mayan.apps.sources.tasks.task_source_process_document'
    ).update(
        task='mayan.apps.sources.tasks.task_source_backend_action_execute'
    )


def reverse_code_source_task_path_update(apps, schema_editor):
    PeriodicTask = apps.get_model(
        app_label='django_celery_beat', model_name='PeriodicTask'
    )

    PeriodicTask.objects.using(
        alias=schema_editor.connection.alias
    ).filter(
        task='mayan.apps.sources.tasks.task_source_backend_action_execute'
    ).update(
        task='mayan.apps.sources.tasks.task_source_process_document'
    )


class Migration(migrations.Migration):
    dependencies = [
        ('sources', '0030_documentfilesourcemetadata'),
        ('django_celery_beat', '0016_alter_crontabschedule_timezone')
    ]

    operations = [
        migrations.RunPython(
            code=code_source_task_path_update,
            reverse_code=reverse_code_source_task_path_update
        )
    ]
