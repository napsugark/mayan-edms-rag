from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ('contenttypes', '0002_remove_content_type_name'),
        ('converter', '0019_auto_20200819_0852')
    ]

    operations = [
        migrations.AlterField(
            model_name='objectlayer',
            name='content_type',
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                to='contenttypes.contenttype', verbose_name='Content type'
            )
        ),
        migrations.AlterField(
            model_name='objectlayer',
            name='object_id',
            field=models.PositiveIntegerField(verbose_name='Object ID')
        )
    ]
