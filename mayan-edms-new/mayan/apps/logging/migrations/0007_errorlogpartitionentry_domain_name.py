from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('logging', '0006_alter_errorlogpartition_unique_together')
    ]

    operations = [
        migrations.AddField(
            model_name='errorlogpartitionentry', name='domain_name',
            field=models.CharField(
                db_index=True, default='logging', max_length=128,
                verbose_name='Domain name'
            )
        )
    ]
