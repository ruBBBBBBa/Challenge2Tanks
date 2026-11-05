# ============================================================================
# LOCKED FILE - do not modify. The tournament runs the original version of
# everything in tanks/ and tools/. Put your own code in student/.
# ============================================================================
"""
Curriculum by win rate: move to a harder opponent (or mixture) when you win enough.

    stages = [
        "random",
        {"basic": 1.0},
        {"basic": 0.5, "aggressive": 0.5},
        {"aggressive": 0.4, "defensive": 0.3, "baseline:easy": 0.3},
    ]
    callback = WinRateCurriculum(stages, threshold=0.7, window=100)
    model.learn(1_000_000, callback=callback)

The win rate and the current stage are logged to TensorBoard (curriculum/...).
"""
from collections import deque
from typing import List, Union

from stable_baselines3.common.callbacks import BaseCallback

Stage = Union[str, dict]


class WinRateCurriculum(BaseCallback):
    def __init__(self, stages: List[Stage], threshold: float = 0.7, window: int = 100, verbose: int = 1):
        super().__init__(verbose)
        if not stages:
            raise ValueError("Give at least one stage")
        self.stages = stages
        self.threshold = threshold
        self.window = window
        self.stage = 0
        self.results = deque(maxlen=window)

    def _on_training_start(self) -> None:
        self._apply_stage()

    def _apply_stage(self) -> None:
        self.training_env.env_method("set_opponent", self.stages[self.stage])
        if self.verbose:
            print(f"[curriculum] stage {self.stage}: {self.stages[self.stage]}")

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and "result" in info:
                self.results.append(1.0 if info["result"] == "win" else 0.0)
        if self.results:
            win_rate = sum(self.results) / len(self.results)
            self.logger.record("curriculum/win_rate", win_rate)
            self.logger.record("curriculum/stage", self.stage)
            if len(self.results) >= self.window and win_rate >= self.threshold \
                    and self.stage < len(self.stages) - 1:
                if self.verbose:
                    print(f"[curriculum] win rate {win_rate:.2f} >= {self.threshold} -> next stage")
                self.stage += 1
                self.results.clear()
                self._apply_stage()
        return True
