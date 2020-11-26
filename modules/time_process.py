from timeit import default_timer as timer

class Timer:
    def __init__(self):
        self.start = timer()
        self.end = timer()
        self.runtime = timer()

    def end_timer(self):
        self.end = timer()
        self.runtime = self.end - self.start

        return round(self.runtime, 4)

