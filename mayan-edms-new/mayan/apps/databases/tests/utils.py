import random


class RandomSeedIdempotent:
    _random_seed_called = False

    @classmethod
    def seed(cls):
        if not cls._random_seed_called:
            random.seed()
            cls._random_seed_called = True
