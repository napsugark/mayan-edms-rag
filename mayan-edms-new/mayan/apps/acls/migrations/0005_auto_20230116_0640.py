from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ('contenttypes', '0002_remove_content_type_name'),
        ('acls', '0004_auto_20210130_0322')
    ]

    operations = [
        migrations.AlterField(
            model_name='accesscontrollist',
            name='content_type',
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name='object_content_type',
                to='contenttypes.contenttype', verbose_name='Content type'
            )
        ),
        migrations.AlterField(
            model_name='accesscontrollist',
            name='object_id',
            field=models.PositiveIntegerField(verbose_name='Object ID')
        )
    ]
