from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ('documents', '0091_fix_documenttype_verbose_name'),
        ('file_metadata', '0012_add_unique_together')
    ]

    operations = [
        migrations.CreateModel(
            name='DocumentTypeDriverConfiguration',
            fields=[
                (
                    'id', models.AutoField(
                        auto_created=True, primary_key=True, serialize=False,
                        verbose_name='ID'
                    )
                ),
                (
                    'enabled', models.BooleanField(
                        default=True, help_text='Enable this driver to '
                        'process document files of the selected document '
                        'type.', verbose_name='Enabled'
                    )
                ),
                (
                    'document_type', models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name='file_metadata_driver_configurations',
                        to='documents.documenttype',
                        verbose_name='Document type'
                    )
                ),
                (
                    'stored_driver', models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name='document_type_configurations',
                        to='file_metadata.storeddriver',
                        verbose_name='Driver'
                    )
                )
            ],
            options={
                'verbose_name': 'Document type driver settings',
                'verbose_name_plural': 'Document type driver settings',
                'ordering': ('stored_driver',),
                'unique_together': {
                    ('document_type', 'stored_driver')
                }
            }
        )
    ]
