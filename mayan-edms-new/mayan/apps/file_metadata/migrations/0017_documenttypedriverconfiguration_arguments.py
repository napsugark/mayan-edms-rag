from django.db import migrations, models
import mayan.apps.common.validators


class Migration(migrations.Migration):
    dependencies = [
        ('file_metadata', '0016_storeddriver_exists')
    ]

    operations = [
        migrations.AddField(
            field=models.TextField(
                blank=True, help_text='Enter the arguments for the drive '
                'for the specific document type as a YAML dictionary. ie: '
                '{"degrees": 180}', validators=[
                    mayan.apps.common.validators.YAMLValidator()
                ], verbose_name='Arguments'
            ),
            model_name='documenttypedriverconfiguration',
            name='arguments'
        )
    ]
