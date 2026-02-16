from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('metadata', '0018_alter_documentmetadata_value')
    ]

    operations = [
        migrations.AlterField(
            field=models.TextField(
                blank=True, help_text='The actual value stored '
                'in the metadata type field for the document.',
                null=True, verbose_name='Value'
            ), model_name='documentmetadata', name='value'
        )
    ]
