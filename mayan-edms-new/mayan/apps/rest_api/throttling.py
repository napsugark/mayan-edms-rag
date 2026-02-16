from django.core.cache import caches

from rest_framework import throttling


class MayanAnonRateThrottle(throttling.AnonRateThrottle):
    cache = caches['rest_api_throttling']


class MayanUserRateThrottle(throttling.UserRateThrottle):
    cache = caches['rest_api_throttling']
