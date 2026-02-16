from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL)
    ]

    operations = [
        migrations.CreateModel(
            name='RecentSearch',
            fields=[
                (
                    'id', models.AutoField(
                        auto_created=True, primary_key=True, serialize=False,
                        verbose_name='ID'
                    )
                ),
                (
                    'query', models.TextField(
                        editable=False, verbose_name='Query'
                    )
                ),
                (
                    'datetime_created', models.DateTimeField(
                        auto_now=True, db_index=True,
                        verbose_name='Datetime created'
                    )
                ),
                (
                    'hits', models.IntegerField(
                        editable=False, verbose_name='Hits'
                    )
                ),
                (
                    'user', models.ForeignKey(
                        on_delete=models.CASCADE,
                        to=settings.AUTH_USER_MODEL,
                        verbose_name='User'
                    )
                )
            ],
            options={
                'ordering': ('-datetime_created',),
                'verbose_name': 'Recent search',
                'verbose_name_plural': 'Recent searches'
            },
            bases=(models.Model,)
        )
    ]
