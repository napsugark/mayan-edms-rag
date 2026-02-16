from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion

import mayan.apps.databases.model_mixins
import mayan.apps.dynamic_search.model_mixins


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('dynamic_search', '0003_auto_20161028_0707')
    ]

    operations = [
        migrations.CreateModel(
            name='SavedResultset',
            fields=[
                (
                    'id', models.AutoField(
                        auto_created=True, primary_key=True, serialize=False,
                        verbose_name='ID'
                    )
                ),
                (
                    'time_to_live', models.PositiveIntegerField(
                        help_text='Time to keep the resultset in seconds. '
                        'This value is increased every time the resultset is '
                        'accessed.', verbose_name='Time to live'
                    )
                ),
                (
                    'timestamp', models.DateTimeField(
                        db_index=True, help_text='The server date and time '
                        'when the resultset was created.',
                        verbose_name='Timestamp'
                    )
                ),
                (
                    'epoch', models.PositiveBigIntegerField(
                        verbose_name='Epoch'
                    )
                ),
                (
                    'app_label', models.CharField(
                        max_length=64, verbose_name='App label'
                    )
                ),
                (
                    'model_name', models.CharField(
                        max_length=64, verbose_name='Model name'
                    )
                ),
                (
                    'search_query_text', models.TextField(
                        verbose_name='Search query'
                    )
                ),
                (
                    'search_explainer_text', models.TextField(
                        verbose_name='Search explainer text'
                    )
                ),
                (
                    'result_count', models.PositiveIntegerField(
                        help_text='Number of results stored in the '
                        'resultset.', verbose_name='Result count'
                    )
                ),
                (
                    'user', models.ForeignKey(
                        help_text='User for which the resultset was created.',
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name='saved_resultsets',
                        to=settings.AUTH_USER_MODEL, verbose_name='User'
                    )
                ),
            ],
            options={
                'verbose_name': 'Saved resultset',
                'verbose_name_plural': 'Saved resultsets',
                'ordering': ('-timestamp', 'user')
            },
            bases=(
                mayan.apps.databases.model_mixins.ExtraDataModelMixin,
                mayan.apps.dynamic_search.model_mixins.SavedResultsetBusinessLogicModelMixin,
                models.Model
            )
        ),
        migrations.CreateModel(
            name='SavedResultsetEntry',
            fields=[
                (
                    'id', models.AutoField(
                        auto_created=True, primary_key=True, serialize=False,
                        verbose_name='ID'
                    )
                ),
                (
                    'object_id', models.BigIntegerField()
                ),
                (
                    'saved_resultset', models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name='entries',
                        to='dynamic_search.savedresultset',
                        verbose_name='Saved resultset'
                    )
                )
            ],
            options={
                'verbose_name': 'Saved resultset entry',
                'verbose_name_plural': 'Saved resultset entries',
                'ordering': ('id',)
            }
        )
    ]
