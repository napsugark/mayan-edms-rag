from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ('document_states', '0035_workflowstate_final')
    ]

    operations = [
        migrations.AddField(
            field=models.ForeignKey(
                blank=True, null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name='workflow_instances',
                to='document_states.workflowstate',
                verbose_name='Active state'
            ), model_name='workflowinstance', name='state_active'
        )
    ]
