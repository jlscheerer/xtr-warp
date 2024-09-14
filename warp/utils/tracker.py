import time
import matplotlib.pyplot as plt
import pandas as pd


class ExecutionTrackerIteration:
    def __init__(self, tracker):
        self._tracker = tracker

    def __enter__(self):
        self._tracker.next_iteration()

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self._tracker.end_iteration()

class ExecutionTracker:
    def __init__(self, name, steps):
        self._name = name
        self._steps = steps
        self._num_iterations = 0
        self._time = None
        self._time_per_step = {}
        for step in steps:
            self._time_per_step[step] = 0
        self._iter_begin = None
        self._iter_time = 0

    def next_iteration(self):
        self._num_iterations += 1
        self._iterating = True
        self._current_steps = []
        self._iter_begin = time.time()

    def end_iteration(self):
        tok = time.time()
        if self._steps != self._current_steps:
            print(self._steps, self._current_steps)
        assert self._steps == self._current_steps
        self._iterating = False
        self._iter_time += tok - self._iter_begin

    def iteration(self):
        return ExecutionTrackerIteration(self)

    def begin(self, name):
        assert self._time is None and self._iterating
        self._current_steps.append(name)
        self._time = time.time()

    def end(self, name):
        tok = time.time()
        assert self._current_steps[-1] == name
        self._time_per_step[name] += tok - self._time
        self._time = None

    def summary(self, steps=None):
        if steps is None:
            steps = self._steps
        iteration_time = self._iter_time / self._num_iterations
        breakdown = [
            (step, self._time_per_step[step] / self._num_iterations) for step in steps
        ]
        return iteration_time, breakdown

    def as_dict(self):
        return {
            "name": self._name,
            "steps": self._steps,
            "time_per_step": self._time_per_step,
            "num_iterations": self._num_iterations,
            "iteration_time": self._iter_time,
        }

    @staticmethod
    def from_dict(data):
        tracker = ExecutionTracker(data["name"], data["steps"])
        tracker._time_per_step = data["time_per_step"]
        tracker._num_iterations = data["num_iterations"]
        tracker._iter_time = data["iteration_time"]
        return tracker

    def __getitem__(self, key):
        assert key in self._steps
        return self._time_per_step[key] / self._num_iterations

    def display(self, steps=None, bound=None):
        iteration_time, breakdown = self.summary(steps)
        df = pd.DataFrame(
            {
                "Task": [x[0] for x in breakdown],
                "Duration": [x[1] * 1000 for x in breakdown],
            }
        )
        df["Start"] = df["Duration"].cumsum().shift(fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 2))

        for i, task in enumerate(df["Task"]):
            start = df["Start"][i]
            duration = df["Duration"][i]
            ax.barh("Tasks", duration, left=start, height=0.5, label=task)

        plt.xlabel("Latency (ms)")
        accumulated = round(sum([x[1] for x in breakdown]) * 1000, 1)
        actual = round(iteration_time * 1000, 1)
        plt.title(
            f"{self._name} (iterations={self._num_iterations}, accumulated={accumulated}ms, actual={actual}ms)"
        )
        ax.set_yticks([])
        ax.set_ylabel("")

        if bound is not None:
            ax.set_xlim([0, bound])

        plt.legend()
        plt.show()


class NOPTracker:
    def __init__(self):
        pass

    def next_iteration(self):
        pass

    def begin(self, name):
        pass  # NOP

    def end(self, name):
        pass  # NOP

    def end_iteration(self):
        pass

    def summary(self):
        raise AssertionError

    def display(self):
        raise AssertionError
