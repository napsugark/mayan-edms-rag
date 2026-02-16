from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('document_states', '0037_populate_state_active'),
    ]

    operations = [
        migrations.AlterField(
            model_name='workflowinstance',
            name='state_active',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='workflow_instances', to='document_states.workflowstate', verbose_name='Active state'),
        ),
    ]
