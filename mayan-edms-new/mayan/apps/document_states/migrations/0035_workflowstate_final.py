from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('document_states', '0034_workflowtransitionfield_default')
    ]

    operations = [
        migrations.AddField(
            field=models.BooleanField(
                default=False, help_text='The state at which the workflow '
                'will stop. Only one state can be the final state.',
                verbose_name='Final'
            ),
            model_name='workflowstate', name='final'
        )
    ]
