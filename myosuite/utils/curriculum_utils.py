import numpy as np
from matplotlib import pyplot as plt


class Curriculum():
    """
    Set up an curriculum factoring the current progress of agents
    """
    def __init__(self,
                threshold = 0.9,   # value above which curriculum is active
                rate = 1.0/100.0,   # rate of progress for curriculum
                start = 0.0,        # starting value of curriculum
                end = 1.0,          # ending value of curriculum
                # filter_coef = 0.95, # filter for updating the progress
                filter_coef = 0.99, # filter for updating the progress
                ):

        self._threshold = threshold
        self._rate = rate
        self._start = start
        self._end = end
        self._filter_coef = filter_coef

        self._value = 0.0           # curriculum's current value
        self._progress = 0.0        # curriculum's measure of overall progress
        self._recording = True

        assert self._rate>0, "rate should always be positive"

    # update the curriculum based on current progress made by the agent
    def update(self, current_success):
        if self._recording:
            # update the progress measure
            self._progress = self._progress*self._filter_coef + current_success*(1.-self._filter_coef)

            # if sufficient progress, bump curriculum
            if self._value <= 1.0: # if not saturated
                if(current_success>=self._threshold): # if maintaining quality
                    if(self._progress>=self._threshold): # if progress is satisfactory
                        self._value += self._rate

    # get the current curriculum status
    def status(self):
        return self._start + self._value*(self._end - self._start)

    def override(self, end):
        self._value = 1.0
        self._end = end
        self._recording = False


class TerrainCurriculum(Curriculum):
    """
    Set up an curriculum factoring the current progress of agents
    """
    def __init__(self,
                 level_up_threshold = 0.9,
                 max_level = 3,
                 *args,
                 **kwargs,
                ):
        super().__init__(*args, **kwargs)
        self.current_level = 0
        self._level_up_threshold = level_up_threshold
        self._max_level = max_level

    # update the curriculum based on current progress made by the agent
    def update(self, current_success):
        if self._recording:
            # update the progress measure
            self._progress = self._progress*self._filter_coef + current_success*(1.-self._filter_coef)

            # if sufficient progress, bump curriculum
            if self._value <= 1.0: # if not saturated
                if(current_success>=self._threshold): # if maintaining quality
                    if(self._progress>=self._threshold): # if progress is satisfactory
                        self._value += self._rate
            # level up
            if self._value >= self._level_up_threshold:
                self.current_level = min(self.current_level + 1, self._max_level)
                self._value = 0
                self._progress = 0

    # get the current curriculum status
    def status(self):
        return self._start + self._value*(self._end - self._start)

    def override(self, end, level=None):
        self._value = 1.0
        self._end = end
        self._recording = False
        self.current_level = self._max_level if level is None else level


if __name__ == '__main__':
    curric = TerrainCurriculum(rate=1/1000)
    success = np.zeros((3000,))
    success[400:600] = 0.6
    success[600:800] = 0.95
    success[800:] = 1.0
    vals = []
    progress = []
    status = []
    for succ in success:
        # print(succ)
        curric.update(succ)
        vals.append(succ)
        print(curric.current_level)
        progress.append(curric._progress)
        status.append(curric.status())
    plt.plot(vals, label='value')
    plt.plot(progress, label='progress')
    plt.plot(status, label='status')
    # plt.plot([x/100 for x in success])
    plt.legend()
    plt.show()


