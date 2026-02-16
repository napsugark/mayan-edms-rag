from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('documents', '0083_auto_20230325_0542')
    ]

    operations = [
        migrations.AlterField(
            model_name='documenttype', name='document_stub_pruning_enabled',
            field=models.BooleanField(
                default=True, help_text='Delete documents that do not '
                'contain any files after a configured expiration interval.',
                verbose_name='Document stub pruning'
            )
        )
    ]
