"""
tracer.py — Per-run execution trace written to logs/<run_id>.json.

Usage:
    from pipeline.tracer import RunTracer
    tracer = RunTracer()
    tracer.set_jd(jd)
    tracer.record_pruning(folder_names, decisions, kept)
    tracer.record_cascade_step("l1", before=200, after=185)
    tracer.record_folder_load("bangalore/active", loaded=80, active=75)
    tracer.finish(output_path, rows_written=100, elapsed=12.4)
    # → writes logs/run_20240115_143000_123456.json
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_LOGS_DIR = Path(__file__).parent.parent / "logs"


class RunTracer:
    """Collects execution trace throughout a ranking run and writes a JSON report."""

    def __init__(self):
        self._lock = threading.Lock()
        ts = datetime.now()
        self.run_id = ts.strftime("%Y%m%d_%H%M%S_") + f"{ts.microsecond:06d}"
        self._trace: dict = {
            "run_id": self.run_id,
            "started_at": ts.isoformat(),
            "jd_summary": {},
            "pruning": {
                "discovered_folders": [],
                "folder_decisions": [],
                "kept_folders": [],
                "dropped_count": 0,
            },
            "folder_loads": [],
            "cascade": {
                "steps": [],
                "total_input": 0,
                "total_output": 0,
            },
            "l7": {},
            "setup_timings": [],
            "output": {
                "candidates_ranked": 0,
                "top_scores": [],
                "output_file": None,
                "output_json": None,
                "elapsed_seconds": None,
            },
            "errors": [],
        }

    # ── JD summary ───────────────────────────────────────────────────────────
    def set_jd(self, jd: dict) -> None:
        """Record the parsed JD fields that are relevant to pruning decisions."""
        with self._lock:
            self._trace["jd_summary"] = {
                "job_title": jd.get("job_title", ""),
                "location": jd.get("location", ""),
                "experience_min": jd.get("experience_min", 0),
                "experience_max": jd.get("experience_max", 99),
                "required_seniority": jd.get("required_seniority", ""),
                "explicit_required": jd.get("explicit_required", [])[:15],
                "inferred_required": jd.get("inferred_required", [])[:10],
                "role_description_excerpt": jd.get("role_description", "")[:200],
            }

    # ── Pruning ───────────────────────────────────────────────────────────────
    def record_pruning(
        self,
        discovered: List[str],
        decisions: Dict[str, str],
        kept: List[Tuple[str, float]],
    ) -> None:
        """
        Record the full pruning outcome.

        decisions: {folder_key: "KEEP (score=0.42)" | "DROP:reason"}
        kept:      [(folder_key, score), ...]
        """
        kept_names = [k for k, _ in kept]
        dropped = [
            {"folder": f, "reason": d.replace("DROP:", "")}
            for f, d in decisions.items()
            if d.startswith("DROP")
        ]
        folder_decision_list = [
            {
                "folder": f,
                "decision": "KEEP" if f in kept_names else "DROP",
                "detail": decisions.get(f, ""),
            }
            for f in discovered
        ]
        with self._lock:
            self._trace["pruning"].update({
                "discovered_folders": list(discovered),
                "folder_decisions": folder_decision_list,
                "kept_folders": kept_names,
                "dropped_count": len(dropped),
                "dropped_folders": dropped,
            })

    # ── Folder loads ──────────────────────────────────────────────────────────
    def record_folder_load(
        self, folder: str, loaded: int, active: int
    ) -> None:
        """Record how many candidates were loaded and how many passed the inactive filter."""
        with self._lock:
            self._trace["folder_loads"].append({
                "folder": folder,
                "loaded": loaded,
                "after_inactive_filter": active,
                "inactive_removed": loaded - active,
            })
            self._trace["cascade"]["total_input"] += active

    # ── Cascade steps ─────────────────────────────────────────────────────────
    def record_cascade_step(
        self, layer: str, before: int, after: int, notes: dict = None
    ) -> None:
        """Record candidate counts before/after a layer. notes: optional extra stats."""
        entry = {"layer": layer, "before": before, "after": after, "removed": before - after}
        if notes:
            entry["notes"] = notes
        with self._lock:
            self._trace["cascade"]["steps"].append(entry)
            self._trace["cascade"]["total_output"] = after

    # ── L7 FIS + FlashRank ────────────────────────────────────────────────────
    def record_l7(
        self,
        fis_input: int,
        fis_output: int,
        flashrank_ran: bool,
        flashrank_top_n: int = 0,
        score_range: tuple = (0.0, 0.0),
    ) -> None:
        """Record the L7 FIS aggregation + FlashRank polish outcome."""
        with self._lock:
            self._trace["l7"] = {
                "fis_input": fis_input,
                "fis_output": fis_output,
                "flashrank_ran": flashrank_ran,
                "flashrank_top_n_reranked": flashrank_top_n,
                "score_min": round(score_range[0], 4),
                "score_max": round(score_range[1], 4),
            }

    # ── Setup timings (model load / JD parse / I/O) ───────────────────────────
    def record_timing(self, name: str, elapsed_s: float) -> None:
        """Record a named elapsed time (seconds) for setup / I/O operations."""
        with self._lock:
            self._trace["setup_timings"].append(
                {"name": name, "elapsed_s": round(elapsed_s, 3)}
            )

    # ── Errors ────────────────────────────────────────────────────────────────
    def record_error(self, context: str, message: str) -> None:
        with self._lock:
            self._trace["errors"].append({"context": context, "message": message})
            logger.error(f"[tracer] {context}: {message}")

    # ── Finish & write ────────────────────────────────────────────────────────
    def finish(
        self,
        output_path: Optional[Path] = None,
        output_json: Optional[Path] = None,
        rows_written: int = 0,
        top_scores: Optional[List[float]] = None,
        elapsed: float = 0.0,
    ) -> Path:
        """
        Finalise the trace and write it to logs/<run_id>.json.
        Returns the path of the written trace file.
        """
        with self._lock:
            self._trace["output"].update({
                "candidates_ranked": rows_written,
                "top_scores": (top_scores or [])[:10],
                "output_file": str(output_path) if output_path else None,
                "output_json": str(output_json) if output_json else None,
                "elapsed_seconds": round(elapsed, 2),
                "finished_at": datetime.now().isoformat(),
            })

        _LOGS_DIR.mkdir(parents=True, exist_ok=True)
        trace_path = _LOGS_DIR / f"run_{self.run_id}.json"
        try:
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(self._trace, f, indent=2, ensure_ascii=False)
            logger.info(f"Run trace → {trace_path}")
        except Exception as exc:
            logger.warning(f"Failed to write run trace: {exc}")
        return trace_path

    # ── Convenience: read back ────────────────────────────────────────────────
    def as_dict(self) -> dict:
        with self._lock:
            return dict(self._trace)
