class Stats:
    def __init__(self):
        self.win = 0
        self.battle = 0
        self.time = 0
        self.start_time = 0

    def __repr__(self):
        return f'Stats(win={self.win}, battle={self.battle}, time={self.time}, start_time={self.start_time})'

    def __str__(self):
        return repr(self)


class Result:
    def __init__(self, my_ink: int, his_ink: int):
        self.my_ink = my_ink
        self.his_ink = his_ink

    def __repr__(self):
        return f'Result(my_ink={self.my_ink}, his_ink={self.his_ink})'

    def __str__(self):
        return repr(self)
