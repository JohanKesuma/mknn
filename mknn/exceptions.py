class DistanceException(Exception):
    def __init__(self, *args, **kwargs):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return 'DistanceException, {0}'.format(self.message)
        else:
            return 'DistanceException raised'