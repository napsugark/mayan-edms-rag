from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('document_parsing', '0011_delete_documentfileparseerror')
    ]

    operations = [
        migrations.AlterField(
            model_name='documenttypesettings', name='auto_parsing',
            field=models.BooleanField(
                default=True, help_text='Automatically queue newly '
                'created documents for parsing.', verbose_name='Auto parsing'
            )
        )
    ]
