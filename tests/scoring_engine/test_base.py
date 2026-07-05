import pandas as pd
from rabbit_hunter.scoring_engine.base import BaseStrategy, ScoreOutput


class Dummy(BaseStrategy):
    name = "dummy"
    version = "0.1.0"

    def score(self, features_row, features_history):
        return ScoreOutput(long=0.5, short=0.1, components={"x": 0.4}, metadata={"note": "hi"})


def test_score_output_frozen_and_typed():
    d = Dummy()
    out = d.score({}, pd.DataFrame())
    assert out.long == 0.5
    assert out.short == 0.1
    assert out.components == {"x": 0.4}
    assert out.metadata == {"note": "hi"}
