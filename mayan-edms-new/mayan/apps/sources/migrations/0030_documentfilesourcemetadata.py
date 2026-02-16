from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ('documents', '0088_populate_version_active'),
        ('sources', '0029_update_source_backend_paths')
    ]

    operations = [
        migrations.CreateModel(
            name='DocumentFileSourceMetadata',
            fields=[
                (
                    'id', models.AutoField(
                        auto_created=True, primary_key=True,
                        serialize=False, verbose_name='ID'
                    )
                ),
                (
                    'key', models.CharField(
                        db_index=True, help_text='Name of the source '
                        'metadata entry.', max_length=255, verbose_name='Key'
                    )
                ),
                (
                    'value', models.TextField(
                        blank=True, help_text='The actual value '
                        'stored in the source metadata for the document.',
                        null=True, verbose_name='Value'
                    )
                ),
                (
                    'document_file', models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name='source_metadata',
                        to='documents.documentfile',
                        verbose_name='Document file')
                ),
                (
                    'source', models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name='metadata', to='sources.source',
                        verbose_name='Source'
                    )
                )
            ],
            options={
                'verbose_name': 'Document file source metadata',
                'verbose_name_plural': 'Document file source metadata',
                'ordering': ('key',),
                'unique_together': {
                    ('source', 'document_file', 'key')
                }
            }
        )
    ]
