from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('document_states', '0029_auto_20230505_1716')
    ]

    operations = [
        migrations.AlterField(
            field=models.TextField(
                blank=True, help_text='JSON encoded data for the '
                'backend class.', verbose_name='Backend data'
            ), model_name='workflowstateaction', name='backend_data'
        ),
        migrations.AlterField(
            field=models.CharField(
                help_text='The dotted Python path to the backend class.',
                max_length=128, verbose_name='Backend path'
            ), model_name='workflowstateaction', name='backend_path'
        )
    ]
