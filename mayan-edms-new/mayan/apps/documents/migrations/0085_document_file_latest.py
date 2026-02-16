from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        (
            'documents',
            '0084_alter_documenttype_document_stub_pruning_enabled'
        )
    ]

    operations = [
        migrations.AddField(
            model_name='document', name='file_latest',
            field=models.OneToOneField(
                blank=True, null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='document_latest', to='documents.documentfile',
                verbose_name='Latest document file'
            )
        )
    ]
