from django.utils.translation import gettext_lazy as _

from mayan.apps.smart_settings.settings import setting_cluster

from .literals import (
    DEFAULT_CONVERTER_ASSET_CACHE_MAXIMUM_SIZE,
    DEFAULT_CONVERTER_ASSET_CACHE_STORAGE_BACKEND,
    DEFAULT_CONVERTER_ASSET_CACHE_STORAGE_BACKEND_ARGUMENTS,
    DEFAULT_CONVERTER_ASSET_STORAGE_BACKEND,
    DEFAULT_CONVERTER_ASSET_STORAGE_BACKEND_ARGUMENTS,
    DEFAULT_CONVERTER_GRAPHICS_BACKEND,
    DEFAULT_CONVERTER_GRAPHICS_BACKEND_ARGUMENTS,
    DEFAULT_CONVERTER_IMAGE_CACHE_TIME,
    DEFAULT_CONVERTER_IMAGE_GENERATION_MAX_RETRIES,
    DEFAULT_CONVERTER_IMAGE_GENERATION_TIMEOUT,
    DEFAULT_CONVERTER_LOAD_TRUNCATED_IMAGES
)
from .setting_callbacks import callback_update_asset_cache_size
from .setting_migrations import ConvertSettingMigration

setting_namespace = setting_cluster.do_namespace_add(
    label=_(message='Converter'), migration_class=ConvertSettingMigration,
    name='converter', version='0002'
)


setting_asset_cache_maximum_size = setting_namespace.do_setting_add(
    default=DEFAULT_CONVERTER_ASSET_CACHE_MAXIMUM_SIZE,
    global_name='CONVERTER_ASSET_CACHE_MAXIMUM_SIZE',
    help_text=_(
        message='The threshold at which the '
        'CONVERTER_ASSET_CACHE_STORAGE_BACKEND '
        'will start deleting the oldest asset cache files. '
        'Specify the size in bytes.'
    ), post_edit_function=callback_update_asset_cache_size
)
setting_asset_cache_storage_backend = setting_namespace.do_setting_add(
    default=DEFAULT_CONVERTER_ASSET_CACHE_STORAGE_BACKEND,
    global_name='CONVERTER_ASSET_CACHE_STORAGE_BACKEND', help_text=_(
        message='Path to the Storage subclass to use when storing the '
        'cached asset files.'
    )
)
setting_asset_cache_storage_backend_arguments = setting_namespace.do_setting_add(
    default=DEFAULT_CONVERTER_ASSET_CACHE_STORAGE_BACKEND_ARGUMENTS,
    global_name='CONVERTER_ASSET_CACHE_STORAGE_BACKEND_ARGUMENTS',
    help_text=_(
        message='Arguments to pass to the '
        'CONVERTER_ASSET_CACHE_STORAGE_BACKEND.'
    )
)
setting_asset_storage_backend = setting_namespace.do_setting_add(
    default=DEFAULT_CONVERTER_ASSET_STORAGE_BACKEND,
    global_name='CONVERTER_ASSET_STORAGE_BACKEND', help_text=_(
        message='Path to the Storage subclass to use when storing assets.'
    )
)
setting_asset_storage_backend_arguments = setting_namespace.do_setting_add(
    default=DEFAULT_CONVERTER_ASSET_STORAGE_BACKEND_ARGUMENTS,
    global_name='CONVERTER_ASSET_STORAGE_BACKEND_ARGUMENTS', help_text=_(
        message='Arguments to pass to the CONVERTER_ASSET_STORAGE_BACKEND.'
    )
)
setting_graphics_backend = setting_namespace.do_setting_add(
    default=DEFAULT_CONVERTER_GRAPHICS_BACKEND,
    global_name='CONVERTER_GRAPHICS_BACKEND', help_text=_(
        message='Graphics conversion backend to use.'
    )
)
setting_graphics_backend_arguments = setting_namespace.do_setting_add(
    default=DEFAULT_CONVERTER_GRAPHICS_BACKEND_ARGUMENTS,
    global_name='CONVERTER_GRAPHICS_BACKEND_ARGUMENTS', help_text=_(
        message='Configuration options for the graphics conversion backend.'
    )
)
setting_image_cache_time = setting_namespace.do_setting_add(
    default=DEFAULT_CONVERTER_IMAGE_CACHE_TIME,
    global_name='CONVERTER_IMAGE_CACHE_TIME',
    help_text=_(
        message='Time in seconds that the browser should cache the '
        'supplied image.'
    )
)
setting_image_generation_max_retries = setting_namespace.do_setting_add(
    default=DEFAULT_CONVERTER_IMAGE_GENERATION_MAX_RETRIES,
    global_name='CONVERTER_IMAGE_GENERATION_MAX_RETRIES',
    help_text=_(
        message='Maximum number of retries before giving up. A value '
        'of None means the task will retry forever.'
    )
)
setting_image_generation_timeout = setting_namespace.do_setting_add(
    default=DEFAULT_CONVERTER_IMAGE_GENERATION_TIMEOUT,
    global_name='CONVERTER_IMAGE_GENERATION_TIMEOUT',
    help_text=_(
        message='Time in seconds after which the image generation task '
        'will stop running and raise an error.'
    )
)
setting_load_truncated_images = setting_namespace.do_setting_add(
    default=DEFAULT_CONVERTER_LOAD_TRUNCATED_IMAGES,
    global_name='CONVERTER_LOAD_TRUNCATED_IMAGES',
    help_text=_(
        message='Whether or not to load truncated image files.'
    )
)
