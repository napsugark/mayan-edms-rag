from .http_adapters import TestClientAdapter


def request_method_factory(test_case, content=None):
    def get_adapter(url):
        return TestClientAdapter(
            response_content=content, test_case=test_case
        )

    return get_adapter
