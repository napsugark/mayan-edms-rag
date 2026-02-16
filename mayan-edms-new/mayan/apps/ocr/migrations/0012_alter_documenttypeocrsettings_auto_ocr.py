from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('ocr', '0011_delete_documentversionocrerror')
    ]

    operations = [
        migrations.AlterField(
            model_name='documenttypeocrsettings', name='auto_ocr',
            field=models.BooleanField(
                default=True, help_text='Automatically queue newly '
                'created documents for OCR.', verbose_name='Auto OCR'
            )
        )
    ]
