from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('file_metadata', '0007_auto_20200917_0708')
    ]

    operations = [
        migrations.AlterField(
            model_name='filemetadataentry',
            name='value',
            field=models.TextField(
                blank=True, help_text='Value of the file metadata entry.',
                max_length=255, verbose_name='Value'
            )
        )
    ]
