class Average:
    def __init__(self):
        self.total = None
        self.count = 0

    def add(self, value):
        if self.total is None:
            self.total = value
        else:
            self.total += value
        self.count += 1

    def get(self):
        return self.total / self.count if self.count != 0 else 0
    
    def reset(self):
        self.total = None
        self.count = 0
