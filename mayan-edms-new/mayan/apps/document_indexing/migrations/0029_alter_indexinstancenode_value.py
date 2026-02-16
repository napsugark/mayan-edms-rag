from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        (
            'document_indexing',
            '0028_populate_existing_index_template_event_triggers'
        )
    ]

    operations = [
        migrations.AlterField(
            model_name='indexinstancenode', name='value',
            field=models.CharField(
                blank=True, db_index=True, max_length=255,
                verbose_name='Value'
            )
        )
    ]
