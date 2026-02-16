from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('duplicates', '0009_auto_20210124_0738')
    ]

    operations = [
        migrations.AlterField(
            model_name='duplicatebackendentry',
            name='documents',
            field=models.ManyToManyField(
                related_name='as_duplicate', to='documents.Document',
                verbose_name='Duplicated documents'
            )
        )
    ]
