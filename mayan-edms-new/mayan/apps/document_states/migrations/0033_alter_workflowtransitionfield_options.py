from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ('document_states', '0032_workflowtransitionfield_lookup')
    ]

    operations = [
        migrations.AlterModelOptions(
            name='workflowtransitionfield', options={
                'ordering': ('label',),
                'verbose_name': 'Workflow transition field',
                'verbose_name_plural': 'Workflow transition fields'
            }
        )
    ]
