from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('documents', '0080_populate_file_size')
    ]

    operations = [
        migrations.AlterField(
            model_name='documentfile', name='size',
            field=models.PositiveBigIntegerField(
                blank=True, db_index=True, editable=False,
                help_text='The size of the file in bytes.', null=True,
                verbose_name='Size'
            )
        )
    ]
