from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('document_states', '0038_alter_workflowinstance_state_active')
    ]

    operations = [
        migrations.AddField(
            field=models.BooleanField(
                default=False, help_text='Ignore workflow instances if '
                'they are in their final state.',
                verbose_name='Ignore completed'
            ), model_name='workflow', name='ignore_completed'
        )
    ]
