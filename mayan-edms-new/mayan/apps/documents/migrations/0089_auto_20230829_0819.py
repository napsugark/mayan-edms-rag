from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('documents', '0088_populate_version_active')
    ]

    operations = [
        migrations.AlterField(
            model_name='documenttype',
            name='delete_time_unit',
            field=models.CharField(
                blank=True, choices=[
                    ('days', 'Days'), ('hours', 'Hours'),
                    ('minutes', 'Minutes')
                ], default='days', max_length=12, null=True,
                verbose_name='Delete time unit'
            )
        ),
        migrations.AlterField(
            model_name='documenttype',
            name='trash_time_unit',
            field=models.CharField(
                blank=True, choices=[
                    ('days', 'Days'), ('hours', 'Hours'),
                    ('minutes', 'Minutes')
                ], max_length=12, null=True, verbose_name='Trash time unit'
            )
        )
    ]
