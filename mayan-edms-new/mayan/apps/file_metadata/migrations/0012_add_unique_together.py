from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ('file_metadata', '0011_populate_internal_name')
    ]

    operations = [
        migrations.AlterModelOptions(
            name='filemetadataentry', options={
                'ordering': ('internal_name', 'value'),
                'verbose_name': 'File metadata entry',
                'verbose_name_plural': 'File metadata entries'
            }
        ),
        migrations.AlterUniqueTogether(
            name='filemetadataentry', unique_together={
                ('document_file_driver_entry', 'internal_name')
            }
        )
    ]
