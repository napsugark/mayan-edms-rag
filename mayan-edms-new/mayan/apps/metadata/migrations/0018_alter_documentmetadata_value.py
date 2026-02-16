from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('metadata', '0017_auto_20211226_1036')
    ]

    operations = [
        migrations.AlterField(
            model_name='documentmetadata',
            name='value',
            field=models.TextField(
                blank=True, default='', help_text='The actual value '
                'stored in the metadata type field for the document.',
                verbose_name='Value'
            ), preserve_default=False
        )
    ]
