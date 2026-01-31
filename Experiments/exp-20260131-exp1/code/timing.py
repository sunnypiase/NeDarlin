from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class StageTimer:
    stages: Dict[str, float] = field(default_factory=dict)
    _starts: Dict[str, float] = field(default_factory=dict)

    def start(self, name: str) -> None:
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        start = self._starts.pop(name, None)
        if start is None:
            return
        self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - start)

    def summary_lines(self) -> List[str]:
        return [f"{name}: {duration:.2f}s" for name, duration in self.stages.items()]
