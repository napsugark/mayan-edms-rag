from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('documents', '0082_populate_document_file_size')
    ]

    operations = [
        migrations.AddField(
            model_name='documenttype',
            name='document_stub_expiration_interval',
            field=models.PositiveBigIntegerField(
                default=86400, help_text='Time (in seconds) after which '
                'a document stub will be considered invalid and deleted, '
                'if pruning is enabled. This an optimization setting and '
                'should only be changed for specific circumstances.',
                verbose_name='Document stub expiration interval'
            )
        ),
        migrations.AddField(
            model_name='documenttype',
            name='document_stub_pruning_enabled',
            field=models.BooleanField(
                default=True, help_text='Delete documents that do not '
                'contain any files.', verbose_name='Document stub pruning'
            )
        )
    ]
