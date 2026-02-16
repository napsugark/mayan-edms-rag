from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ('documents', '0086_populate_latest_file')
    ]

    operations = [
        migrations.AddField(
            model_name='document', name='version_active',
            field=models.OneToOneField(
                blank=True, null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='document_active',
                to='documents.documentversion',
                verbose_name='Active document version'
            )
        )
    ]
