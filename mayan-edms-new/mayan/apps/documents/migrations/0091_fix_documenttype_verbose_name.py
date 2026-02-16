from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ('documents', '0090_alter_documentversion_active')
    ]

    operations = [
        migrations.AlterModelOptions(
            name='documenttype',
            options={
                'ordering': ('label',), 'verbose_name': 'Document type',
                'verbose_name_plural': 'Document types'
            }
        )
    ]
