from django.utils.module_loading import import_string


def execute_callback(callback_dict, name, **kwargs):
    callback_info = callback_dict.pop(
        name, {}
    )

    if callback_info:
        callback_obj = import_string(
            dotted_path=callback_info['dotted_path']
        )
        callback_kwargs = callback_info.get(
            'kwargs', {}
        )
        function = getattr(
            callback_obj, callback_info['function_name']
        )
        function(**callback_kwargs, **kwargs)
