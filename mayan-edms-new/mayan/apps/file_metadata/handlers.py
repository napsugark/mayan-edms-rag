from django.apps import apps


def handler_initialize_new_document_type_file_metadata_driver_configuration(
    sender, instance, **kwargs
):
    DocumentTypeDriverConfiguration = apps.get_model(
        app_label='file_metadata',
        model_name='DocumentTypeDriverConfiguration'
    )
    StoredDriver = apps.get_model(
        app_label='file_metadata', model_name='StoredDriver'
    )

    if kwargs['created']:
        for stored_driver in StoredDriver.objects.all():
            driver_class = stored_driver.driver_class
            enabled = driver_class.get_enabled_value()

            DocumentTypeDriverConfiguration.objects.create(
                document_type=instance, enabled=enabled,
                stored_driver=stored_driver
            )


def handler_post_document_file_upload(sender, instance, **kwargs):
    instance.submit_for_file_metadata_processing()
