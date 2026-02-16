from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ('document_states', '0028_workflowstateescalation')
    ]

    operations = [
        migrations.RenameField(
            model_name='workflowstateaction',
            old_name='action_data',
            new_name='backend_data'
        ),
        migrations.RenameField(
            model_name='workflowstateaction',
            old_name='action_path',
            new_name='backend_path'
        )
    ]
