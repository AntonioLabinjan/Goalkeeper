"""Calculates, aggregates, and stores structured drill performance data."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import Direction


@dataclass(frozen=True)
class RepetitionResult:
    repetition_index: int
    target_direction: Direction
    detected_direction: Optional[Direction]
    reaction_time_s: Optional[float]
    success: bool
    timestamp: str


@dataclass(frozen=True)
class SessionSummary:
    total_attempts: int
    successful_saves: int
    accuracy_percentage: float
    avg_reaction_time_s: Optional[float]
    fastest_reaction_time_s: Optional[float]
    slowest_save_time_s: Optional[float]
    max_streak: int


class Scoreboard:
    """Manages raw results for an active session and calculates summaries."""

    def __init__(self) -> None:
        self.results: List[RepetitionResult] = []

    def record_attempt(
        self,
        index: int,
        target: Direction,
        detected: Optional[Direction],
        reaction_time: Optional[float],
        success: bool,
    ) -> RepetitionResult:
        res = RepetitionResult(
            repetition_index=index,
            target_direction=target,
            detected_direction=detected,
            reaction_time_s=reaction_time,
            success=success,
            timestamp=datetime.now().isoformat(),
        )
        self.results.append(res)
        return res

    def compute_summary(self) -> SessionSummary:
        total = len(self.results)
        if total == 0:
            return SessionSummary(0, 0, 0.0, None, None, None, 0)

        saves = sum(1 for r in self.results if r.success)
        accuracy = (saves / total) * 100.0

        valid_times = [r.reaction_time_s for r in self.results if r.reaction_time_s is not None]
        save_times = [r.reaction_time_s for r in self.results if r.success and r.reaction_time_s is not None]

        avg_time = float(sum(valid_times)) / len(valid_times) if valid_times else None
        fastest = min(valid_times) if valid_times else None
        slowest_save = max(save_times) if save_times else None

        current_streak = 0
        max_streak = 0
        for r in self.results:
            if r.success:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return SessionSummary(
            total_attempts=total,
            successful_saves=saves,
            accuracy_percentage=accuracy,
            avg_reaction_time_s=avg_time,
            fastest_reaction_time_s=fastest,
            slowest_save_time_s=slowest_save,
            max_streak=max_streak,
        )

    def export_json(self, output_dir: str, metadata: Dict[str, Any]) -> str:
        """Saves full session history as a structured JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        summary = self.compute_summary()

        # Enums must be serialized into plain strings for JSON validation.
        serialized_results = []
        for r in self.results:
            d = asdict(r)
            d["target_direction"] = r.target_direction.value
            d["detected_direction"] = r.detected_direction.value if r.detected_direction else None
            serialized_results.append(d)

        payload = {
            "metadata": metadata,
            "summary": asdict(summary),
            "repetitions": serialized_results,
        }

        filename = f"session_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
        return filepath