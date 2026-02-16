from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('document_states', '0033_alter_workflowtransitionfield_options')
    ]

    operations = [
        migrations.AddField(
            model_name='workflowtransitionfield', name='default',
            field=models.TextField(
                blank=True, help_text='Optional default value for the field. '
                'Can be a template.', null=True, verbose_name='Default'
            )
        )
    ]
