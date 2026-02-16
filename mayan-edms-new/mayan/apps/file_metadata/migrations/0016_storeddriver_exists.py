from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('file_metadata', '0015_delete_documenttypesettings')
    ]

    operations = [
        migrations.AddField(
            model_name='storeddriver', name='exists',
            field=models.BooleanField(
                default=True, help_text='The class defined by this instance '
                'is valid and active.', verbose_name='Valid'
            )
        )
    ]
