from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('document_states', '0031_alter_workflowtransitionfield_required')
    ]

    operations = [
        migrations.AddField(
            model_name='workflowtransitionfield', name='lookup',
            field=models.TextField(
                blank=True, help_text='Enter a template to render. Must '
                'result in a comma delimited string.', null=True,
                verbose_name='Lookup'
            )
        )
    ]
