from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name='UpdatedSetting',
            fields=[
                (
                    'id', models.AutoField(
                        auto_created=True, primary_key=True,
                        serialize=False, verbose_name='ID'
                    )
                ),
                (
                    'global_name', models.CharField(
                        db_index=True, help_text='A short text used as '
                        'the tag name.', max_length=255, unique=True,
                        verbose_name='Global name'
                    )
                ),
                (
                    'value_new', models.TextField(
                        blank=True, null=True, verbose_name='New value'
                    )
                ),
                (
                    'value_old', models.TextField(
                        blank=True, null=True, verbose_name='Old value'
                    )
                )
            ],
            options={
                'verbose_name': 'Updated setting',
                'verbose_name_plural': 'Updated settings'
            }
        )
    ]
