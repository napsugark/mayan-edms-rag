from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ('documents', '0091_fix_documenttype_verbose_name'),
        ('document_states', '0039_workflow_ignore_completed')
    ]

    operations = [
        migrations.AlterField(
            field=models.ForeignKey(
                help_text='The document to which the workflow instance is '
                'attached.', on_delete=django.db.models.deletion.CASCADE,
                related_name='workflows', to='documents.document',
                verbose_name='Document'
            ), model_name='workflowinstance', name='document'
        ),
        migrations.AlterField(
            field=models.ForeignKey(
                help_text='The currently active state of the workflow '
                'instance.', on_delete=django.db.models.deletion.CASCADE,
                related_name='workflow_instances',
                to='document_states.workflowstate',
                verbose_name='Active state'
            ),
            model_name='workflowinstance', name='state_active'
        ),
        migrations.AlterField(
            field=models.ForeignKey(
                help_text='The workflow template.',
                on_delete=django.db.models.deletion.CASCADE,
                related_name='instances', to='document_states.workflow',
                verbose_name='Workflow'
            ), model_name='workflowinstance', name='workflow'
        )
    ]
