from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('document_states', '0040_alter_workflowinstance_document_and_more')
    ]

    operations = [
        migrations.AlterField(
            field=models.CharField(
                help_text='A short text describing the action. Actions '
                'are execute by alphabetical order.', max_length=255,
                verbose_name='Label'
            ),
            model_name='workflowstateaction', name='label'
        )
    ]
