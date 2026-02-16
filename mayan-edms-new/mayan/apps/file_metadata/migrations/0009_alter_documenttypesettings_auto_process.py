from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('file_metadata', '0008_alter_filemetadataentry_value')
    ]

    operations = [
        migrations.AlterField(
            model_name='documenttypesettings', name='auto_process',
            field=models.BooleanField(
                default=True, help_text='Automatically queue newly '
                'created documents for processing.',
                verbose_name='Auto process'
            )
        )
    ]
