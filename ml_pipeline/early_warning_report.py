"""Structured Early Warning report generator.

Produces a comprehensive JSON report summarising the results computed by the
Early Warning pipeline so users can download and archive them outside the
Streamlit dashboard.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_pipeline.artifacts import save_json, target_dir as _target_dir
from ml_pipeline.early_warning import (
    alert_thresholds,
    build_danger_zone_intervals,
    compute_alert_metrics,
    load_quality_specs,
    resolve_quality_spec,
    spec_description,
    threshold_analysis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metric_value(metrics: dict | None, key: str, default: Any = 0) -> Any:
    if metrics is None:
        return default
    value = metrics.get(key, default)
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return default
    return value


def _format_intervals(intervals: list[tuple[float, float]]) -> str:
    return ", ".join(f"[{low:g}, {high:g}]" for low, high in intervals)


def _early_warning_status(
    result: dict,
    target: str,
    specs: dict[str, Any],
) -> tuple[str, dict | None, pd.DataFrame]:
    _, spec = resolve_quality_spec(target, specs)
    predictions = result.get("early_warning_predictions")
    if predictions is None:
        predictions = pd.DataFrame()

    if spec is None:
        return "No spec", spec, predictions
    if not predictions.empty:
        return "Ready", spec, predictions
    if result.get("early_warning_error"):
        return "Error", spec, predictions
    return "Legacy run", spec, predictions


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_metadata(
    target_results: dict,
    run_path: str,
    regression_targets: list[str],
) -> dict:
    ready_count = sum(
        1
        for t in regression_targets
        if _early_warning_status(target_results.get(t, {}), t, load_quality_specs())[0]
        in ("Ready", "Legacy run")
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_path": str(run_path),
        "total_targets_in_run": len(target_results),
        "regression_targets": len(regression_targets),
        "targets_with_early_warning": ready_count,
    }


def _build_specs_summary(specs: dict[str, Any]) -> list[dict]:
    rows = []
    for name, config in specs.items():
        spec_type = str(config.get("spec", {}).get("type", "two_sided"))
        rows.append(
            {
                "spec_name": name,
                "aliases": config.get("aliases", []),
                "spec_type": spec_type,
                "spec_description": spec_description(config),
                "danger_zone": _format_intervals(build_danger_zone_intervals(config)),
                "alert_thresholds": alert_thresholds(config),
                "uncertainty_coverage": float(
                    (config.get("uncertainty") or {}).get("coverage", 0.90)
                ),
                "max_interval_width": (config.get("uncertainty") or {}).get(
                    "max_interval_width"
                ),
            }
        )
    return rows


def _build_targets_summary(
    target_results: dict,
    specs: dict[str, Any],
    regression_targets: list[str],
) -> list[dict]:
    rows = []
    for target in regression_targets:
        result = target_results.get(target) or {}
        status, spec, predictions = _early_warning_status(result, target, specs)
        metrics = result.get("early_warning_metrics") or {}
        rows.append(
            {
                "target": target,
                "status": status,
                "spec": "-" if spec is None else spec_description(spec),
                "spec_name": result.get("quality_spec_key", target),
                "rows": int(len(predictions)),
                "alerts": int(_metric_value(metrics, "alert_count", 0)),
                "events": int(_metric_value(metrics, "event_count", 0)),
                "true_positives": int(_metric_value(metrics, "true_positives", 0)),
                "false_positives": int(_metric_value(metrics, "false_positives", 0)),
                "false_negatives": int(_metric_value(metrics, "false_negatives", 0)),
                "false_alerts_per_batch": float(
                    _metric_value(metrics, "false_alerts_per_batch", 0.0)
                ),
                "precision": float(_metric_value(metrics, "precision", 0.0)),
                "recall": float(_metric_value(metrics, "recall", 0.0)),
                "f1": float(_metric_value(metrics, "f1", 0.0)),
                "f3": float(_metric_value(metrics, "f3", 0.0)),
                "specificity": float(_metric_value(metrics, "specificity", 0.0)),
                "balanced_accuracy": float(
                    _metric_value(metrics, "balanced_accuracy", 0.0)
                ),
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
            }
        )
    return rows


def _build_aggregated_confusion_matrices(
    target_results: dict,
    specs: dict[str, Any],
    regression_targets: list[str],
) -> dict:
    all_alerts: list[pd.Series] = []
    all_actual_events: list[pd.Series] = []
    all_dz_alerts: list[pd.Series] = []
    all_actual_in_dz: list[pd.Series] = []
    all_oos_alerts: list[pd.Series] = []
    all_actual_oos: list[pd.Series] = []

    for target in regression_targets:
        result = target_results.get(target)
        if result is None:
            continue
        preds = result.get("early_warning_predictions")
        if preds is None or preds.empty:
            continue
        status_str, target_spec, _ = _early_warning_status(result, target, specs)
        if status_str not in ("Ready", "Legacy run"):
            continue
        if target_spec is None:
            continue
        try:
            thresholds = alert_thresholds(target_spec)
        except Exception:
            continue

        p_dz_thresh = thresholds["p_dz"]
        p_oos_thresh = thresholds["p_oos"]

        if "alert" in preds.columns and "actual_event" in preds.columns:
            all_alerts.append(preds["alert"].astype(bool))
            all_actual_events.append(preds["actual_event"].astype(bool))

        if "p_dz" in preds.columns and "actual_in_dz" in preds.columns:
            all_dz_alerts.append(preds["p_dz"] >= p_dz_thresh)
            all_actual_in_dz.append(preds["actual_in_dz"].astype(bool))

        if "p_oos" in preds.columns and "actual_oos" in preds.columns:
            all_oos_alerts.append(preds["p_oos"] >= p_oos_thresh)
            all_actual_oos.append(preds["actual_oos"].astype(bool))

    def _cm_section(alert_list, event_list, label) -> dict | None:
        if not alert_list:
            return None
        combined_alerts = pd.concat(alert_list, ignore_index=True)
        combined_events = pd.concat(event_list, ignore_index=True)
        metrics = compute_alert_metrics(
            combined_events.to_numpy(), combined_alerts.to_numpy(), None
        )
        return {
            "label": label,
            "n": metrics["n"],
            "confusion_matrix": metrics["confusion_matrix"],
            "true_positives": metrics["true_positives"],
            "false_positives": metrics["false_positives"],
            "true_negatives": metrics["true_negatives"],
            "false_negatives": metrics["false_negatives"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "specificity": metrics["specificity"],
            "f1": metrics["f1"],
            "f3": metrics["f3"],
            "balanced_accuracy": metrics["balanced_accuracy"],
        }

    return {
        "general_alerts": _cm_section(
            all_alerts, all_actual_events, "alert vs actual_event"
        ),
        "danger_zone": _cm_section(
            all_dz_alerts, all_actual_in_dz, "p_dz >= threshold vs actual_in_dz"
        ),
        "oos": _cm_section(
            all_oos_alerts, all_actual_oos, "p_oos >= threshold vs actual_oos"
        ),
    }


def _build_per_target_detail(
    target_results: dict,
    specs: dict[str, Any],
    regression_targets: list[str],
) -> list[dict]:
    details = []
    for target in regression_targets:
        result = target_results.get(target) or {}
        status, spec, predictions = _early_warning_status(result, target, specs)
        if status not in ("Ready", "Legacy run"):
            continue
        if spec is None or predictions.empty:
            continue

        try:
            thresholds = alert_thresholds(spec)
            dz_intervals = build_danger_zone_intervals(spec)
        except Exception:
            continue

        metrics = result.get("early_warning_metrics") or {}

        # Alert tier distribution
        tier_counts = (
            predictions["alert_tier"].value_counts().to_dict()
            if "alert_tier" in predictions.columns
            else {}
        )

        # Threshold analysis
        sweep = threshold_analysis(
            predictions,
            p_oos_threshold=thresholds["p_oos"],
            p_dz_thresholds=(0.05, 0.10, 0.20),
        )

        detail = {
            "target": target,
            "spec_name": result.get("quality_spec_key", target),
            "spec_description": spec_description(spec),
            "danger_zone": _format_intervals(dz_intervals),
            "p_dz_threshold": thresholds["p_dz"],
            "p_oos_threshold": thresholds["p_oos"],
            "metrics": {
                "alert_count": int(_metric_value(metrics, "alert_count", 0)),
                "high_risk_count": int(_metric_value(metrics, "high_risk_count", 0)),
                "event_count": int(_metric_value(metrics, "event_count", 0)),
                "true_positives": int(_metric_value(metrics, "true_positives", 0)),
                "false_positives": int(_metric_value(metrics, "false_positives", 0)),
                "true_negatives": int(_metric_value(metrics, "true_negatives", 0)),
                "false_negatives": int(_metric_value(metrics, "false_negatives", 0)),
                "false_alerts_per_batch": float(
                    _metric_value(metrics, "false_alerts_per_batch", 0.0)
                ),
                "precision": float(_metric_value(metrics, "precision", 0.0)),
                "recall": float(_metric_value(metrics, "recall", 0.0)),
                "specificity": float(_metric_value(metrics, "specificity", 0.0)),
                "f1": float(_metric_value(metrics, "f1", 0.0)),
                "f3": float(_metric_value(metrics, "f3", 0.0)),
                "balanced_accuracy": float(
                    _metric_value(metrics, "balanced_accuracy", 0.0)
                ),
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
                "confusion_matrix": metrics.get("confusion_matrix"),
            },
            "alert_tier_distribution": tier_counts,
            "threshold_analysis": [
                {
                    "p_dz_threshold": float(row["p_dz_threshold"]),
                    "p_oos_threshold": float(row["p_oos_threshold"]),
                    "alert_count": int(row["alert_count"]),
                    "true_positives": int(row["true_positives"]),
                    "false_positives": int(row["false_positives"]),
                    "false_negatives": int(row["false_negatives"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "specificity": float(row["specificity"]),
                    "f1": float(row["f1"]),
                    "f3": float(row.get("f3", 0.0)),
                    "balanced_accuracy": float(row["balanced_accuracy"]),
                }
                for _, row in sweep.iterrows()
            ],
            "artifacts": {
                "quality_specs": "config/quality_specs.json",
                "calibration_residuals": result.get("calibration_residuals_path"),
                "early_warning_predictions": result.get(
                    "early_warning_predictions_path"
                ),
                "early_warning_metrics": result.get("early_warning_metrics_path"),
            },
        }
        details.append(detail)

    return details


def _build_executive_summary(
    targets_summary: list[dict],
    aggregated_cms: dict,
    per_target: list[dict],
) -> dict:
    """Generate a plain-text executive summary with key findings."""
    if not targets_summary:
        return {
            "title": "Executive Summary",
            "text": "No regression targets with Early Warning results available.",
        }

    ready_targets = [
        t for t in targets_summary if t["status"] in ("Ready", "Legacy run")
    ]
    total_alerts = sum(t["alerts"] for t in ready_targets)
    total_events = sum(t["events"] for t in ready_targets)
    total_fp = sum(t["false_positives"] for t in ready_targets)
    total_fn = sum(t["false_negatives"] for t in ready_targets)

    # Targets with worst F1
    sorted_by_f1 = sorted(
        [t for t in ready_targets if t["rows"] > 0],
        key=lambda x: x["f1"],
    )
    worst_f1 = sorted_by_f1[:3] if sorted_by_f1 else []

    # Targets with most alerts
    sorted_by_alerts = sorted(ready_targets, key=lambda x: x["alerts"], reverse=True)
    top_alerts = sorted_by_alerts[:3] if sorted_by_alerts else []

    lines = [
        f"Early Warning Report — {len(ready_targets)} regression target(s) analysed.",
        f"Total alerts across all targets: {total_alerts:,}",
        f"Total quality events: {total_events:,}",
        f"Total false positives: {total_fp:,}",
        f"Total false negatives: {total_fn:,}",
        "",
    ]

    if top_alerts:
        lines.append("Targets with the most alerts:")
        for t in top_alerts:
            lines.append(
                f"  - {t['target']}: {t['alerts']:,} alerts, "
                f"{t['events']:,} events, F1={t['f1']:.3f}, F3={t.get('f3', 0.0):.3f}"
            )
        lines.append("")

    if worst_f1 and any(t["f1"] < 1.0 for t in worst_f1):
        lines.append("Targets with lowest F1 score:")
        for t in worst_f1:
            lines.append(
                f"  - {t['target']}: F1={t['f1']:.3f}, "
                f"F3={t.get('f3', 0.0):.3f}, "
                f"Precision={t['precision']:.3f}, Recall={t['recall']:.3f}"
            )
        lines.append("")

    # Aggregated confusion matrices summary
    general = aggregated_cms.get("general_alerts")
    if general:
        lines.append(
            f"Aggregated confusion matrix (alert vs event): "
            f"TP={general['true_positives']}, FP={general['false_positives']}, "
            f"TN={general['true_negatives']}, FN={general['false_negatives']}, "
            f"F1={general['f1']:.3f}, F3={general.get('f3', 0.0):.3f}"
        )

    return {
        "title": "Executive Summary",
        "text": "\n".join(lines),
        "key_stats": {
            "targets_analysed": len(ready_targets),
            "total_alerts": total_alerts,
            "total_events": total_events,
            "total_false_positives": total_fp,
            "total_false_negatives": total_fn,
        },
    }


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def generate_early_warning_report(
    target_results: dict[str, Any],
    specs: dict[str, Any] | None = None,
    run_path: str | Path = "",
) -> dict[str, Any]:
    """Generate a comprehensive structured Early Warning report dictionary.

    Parameters
    ----------
    target_results : dict
        The ``run["target_results"]`` dictionary produced by the AutoML runner.
    specs : dict, optional
        Quality specifications.  Loaded from ``config/quality_specs.json`` if
        not provided.
    run_path : str or Path, optional
        Base path of the AutoML run (used for metadata and artifact saving).

    Returns
    -------
    dict
        A dictionary ready to be serialised to JSON with sections:
        - metadata
        - quality_specs
        - targets_summary
        - aggregated_confusion_matrices
        - per_target_details
        - executive_summary
    """
    if specs is None:
        specs = load_quality_specs()

    regression_targets = [
        target
        for target, result in target_results.items()
        if (result.get("config") or {}).get("task") == "regression"
    ]

    targets_summary = _build_targets_summary(target_results, specs, regression_targets)
    aggregated_cms = _build_aggregated_confusion_matrices(
        target_results, specs, regression_targets
    )
    per_target = _build_per_target_detail(target_results, specs, regression_targets)

    return {
        "metadata": _build_metadata(target_results, str(run_path), regression_targets),
        "quality_specs": _build_specs_summary(specs),
        "targets_summary": targets_summary,
        "aggregated_confusion_matrices": aggregated_cms,
        "per_target_details": per_target,
        "executive_summary": _build_executive_summary(
            targets_summary, aggregated_cms, per_target
        ),
    }


def save_early_warning_report(
    target_results: dict[str, Any],
    specs: dict[str, Any] | None = None,
    run_path: str | Path = "",
) -> Path:
    """Generate the Early Warning report and persist it as JSON inside the run
    directory.

    Returns
    -------
    Path
        Path to the saved ``early_warning_report.json`` file.
    """
    report = generate_early_warning_report(target_results, specs, run_path)

    run_dir = Path(run_path) if run_path else Path(".")
    t_dir = _target_dir(run_dir, "_early_warning_report")

    report_path = save_json(t_dir / "early_warning_report.json", report)
    return Path(report_path)