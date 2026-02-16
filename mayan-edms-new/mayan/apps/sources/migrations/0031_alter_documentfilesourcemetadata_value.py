from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('sources', '0030_documentfilesourcemetadata')
    ]

    operations = [
        migrations.AlterField(
            field=models.TextField(
                blank=True, help_text='The actual value stored in the source '
                'metadata for the document file.', null=True,
                verbose_name='Value'
            ), model_name='documentfilesourcemetadata', name='value'
        )
    ]
