from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rag_integration', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='ragdocumentversionsync',
            name='content_hash',
            field=models.CharField(
                blank=True,
                help_text='SHA-256 hash of the file content at the time '
                'of last sync. Used to detect file changes.',
                max_length=64,
                null=True,
                verbose_name='Content hash'
            ),
        ),
    ]
