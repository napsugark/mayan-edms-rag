from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('document_states', '0030_auto_20230505_1724')
    ]

    operations = [
        migrations.AlterField(
            model_name='workflowtransitionfield', name='required',
            field=models.BooleanField(
                default=False, help_text='Whether this field needs to be '
                'filled out or not to proceed.', verbose_name='Required'
            )
        )
    ]
