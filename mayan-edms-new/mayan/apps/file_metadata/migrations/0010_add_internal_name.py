from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('file_metadata', '0009_alter_documenttypesettings_auto_process')
    ]

    operations = [
        migrations.AddField(
            model_name='filemetadataentry', name='internal_name',
            field=models.CharField(
                db_index=True, default='', help_text='Normalized name '
                'of the file metadata entry.', max_length=255,
                verbose_name='Internal name'
            ), preserve_default=False
        ),
        migrations.AlterField(
            model_name='filemetadataentry', name='key',
            field=models.CharField(
                help_text='Name of the file metadata entry as provided '
                'by the driver.', max_length=255, verbose_name='Key'
            )
        )
    ]
