

import pandas as pd
from swe_opt.scenarios.scenario_base import PerfScenario

class PR60121Scenario(PerfScenario):
    """
    PR #60121 – Benchmark DataFrame.astype when converting to extension floating dtypes

    Parameters:
    - R: Number of rows
    - C: Number of columns
    """

    def setup(self):
        params = self.params

        rows = int(params["R"])
        cols = int(params["C"])

        df = pd.DataFrame(self.rng.random((rows, cols)))

        self.args = {"df": df}

    def run(self):
        return self.args["df"].astype("Float64")

