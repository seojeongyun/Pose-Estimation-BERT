import csv
import html
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _svg_polyline(x_values, y_values, left, top, width, height):
    return " ".join(
        "{:.2f},{:.2f}".format(
            left + float(x_value) * width,
            top + (1.0 - float(y_value)) * height,
        )
        for x_value, y_value in zip(x_values, y_values)
    )


def _save_roc_svg(curves, micro_curve, macro_curve, output_path, epoch):
    canvas_width = 1500
    canvas_height = max(780, 180 + 30 * len(curves))
    left, top = 90, 90
    plot_width, plot_height = 820, 620
    legend_left = 960

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
        "#3182bd", "#31a354", "#756bb1", "#636363", "#e6550d",
    ]

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">'.format(
            canvas_width, canvas_height
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="{}" y="40" font-size="24" font-weight="bold">{}</text>'.format(
            left, html.escape("Validation Exercise ROC Curves - Best Epoch {}".format(epoch))
        ),
    ]

    for tick in np.linspace(0.0, 1.0, 6):
        x = left + tick * plot_width
        y = top + (1.0 - tick) * plot_height
        svg.extend([
            '<line x1="{0:.1f}" y1="{1}" x2="{0:.1f}" y2="{2}" stroke="#dddddd"/>'.format(
                x, top, top + plot_height
            ),
            '<line x1="{0}" y1="{1:.1f}" x2="{2}" y2="{1:.1f}" stroke="#dddddd"/>'.format(
                left, y, left + plot_width
            ),
            '<text x="{:.1f}" y="{}" font-size="13" text-anchor="middle">{:.1f}</text>'.format(
                x, top + plot_height + 24, tick
            ),
            '<text x="{}" y="{:.1f}" font-size="13" text-anchor="end">{:.1f}</text>'.format(
                left - 10, y + 5, tick
            ),
        ])

    svg.extend([
        '<rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="black" stroke-width="2"/>'.format(
            left, top, plot_width, plot_height
        ),
        '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#999999" stroke-dasharray="8,8"/>'.format(
            left, top + plot_height, left + plot_width, top
        ),
        '<text x="{}" y="{}" font-size="16" text-anchor="middle">False Positive Rate</text>'.format(
            left + plot_width / 2, top + plot_height + 58
        ),
        '<text x="25" y="{}" font-size="16" text-anchor="middle" transform="rotate(-90 25 {})">True Positive Rate</text>'.format(
            top + plot_height / 2, top + plot_height / 2
        ),
        '<text x="{}" y="75" font-size="18" font-weight="bold">Supported validation exercises</text>'.format(
            legend_left
        ),
    ])

    for index, curve in enumerate(curves):
        color = colors[index % len(colors)]
        points = _svg_polyline(
            curve["fpr"], curve["tpr"], left, top, plot_width, plot_height
        )
        legend_y = 105 + index * 30
        label = "{:02d} {} | AP {:.4f} | AUC {:.4f} | n={}".format(
            curve["class_index"],
            curve["exercise_name"],
            curve["average_precision"],
            curve["roc_auc"],
            curve["support"],
        )
        svg.extend([
            '<polyline points="{}" fill="none" stroke="{}" stroke-width="1.8" opacity="0.85"/>'.format(
                points, color
            ),
            '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="3"/>'.format(
                legend_left, legend_y, legend_left + 28, legend_y, color
            ),
            '<text x="{}" y="{}" font-size="13">{}</text>'.format(
                legend_left + 36, legend_y + 5, html.escape(label)
            ),
        ])

    summary_y = 125 + len(curves) * 30
    for label, curve, color in (
        ("micro-average", micro_curve, "#000000"),
        ("macro-average", macro_curve, "#e41a1c"),
    ):
        points = _svg_polyline(
            curve["fpr"], curve["tpr"], left, top, plot_width, plot_height
        )
        svg.extend([
            '<polyline points="{}" fill="none" stroke="{}" stroke-width="4" stroke-dasharray="12,6"/>'.format(
                points, color
            ),
            '<text x="{}" y="{}" font-size="14" font-weight="bold">{} AUC: {:.4f}</text>'.format(
                legend_left, summary_y, label, curve["roc_auc"]
            ),
        ])
        summary_y += 28

    svg.append("</svg>")
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(svg))


def save_exercise_ap_roc_analysis(
    probabilities,
    targets,
    class_names,
    output_dir,
    epoch,
    validation_accuracy,
):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)

    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [N, C]")
    if probabilities.shape[0] != targets.size or targets.size == 0:
        raise ValueError("probabilities and targets must contain the same non-zero N")

    num_classes = probabilities.shape[1]
    if len(class_names) != num_classes:
        raise ValueError(
            "class_names length mismatch: expected={}, actual={}".format(
                num_classes, len(class_names)
            )
        )
    if targets.min() < 0 or targets.max() >= num_classes:
        raise ValueError("target class index is out of range")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain NaN or infinity")

    os.makedirs(output_dir, exist_ok=True)

    one_hot_targets = np.zeros_like(probabilities, dtype=np.int64)
    one_hot_targets[np.arange(targets.size), targets] = 1

    rows = []
    curves = []
    for class_index in range(num_classes):
        binary_target = one_hot_targets[:, class_index]
        support = int(binary_target.sum())
        negative_count = int(binary_target.size - support)

        row = {
            "class_index": class_index,
            "raw_exercise_id": class_index + 22,
            "exercise_name": str(class_names[class_index]),
            "support": support,
            "negative_count": negative_count,
            "average_precision": None,
            "roc_auc": None,
            "status": "skipped",
            "reason": "no_positive_samples" if support == 0 else "no_negative_samples",
        }

        if support > 0 and negative_count > 0:
            class_score = probabilities[:, class_index]
            average_precision = float(
                average_precision_score(binary_target, class_score)
            )
            roc_auc = float(roc_auc_score(binary_target, class_score))
            fpr, tpr, thresholds = roc_curve(binary_target, class_score)

            row.update({
                "average_precision": average_precision,
                "roc_auc": roc_auc,
                "status": "evaluated",
                "reason": "",
            })
            curves.append({
                **row,
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
            })

        rows.append(row)

    if not curves:
        raise ValueError("no exercise class has both positive and negative validation samples")

    supported_indices = [curve["class_index"] for curve in curves]
    supported_targets = one_hot_targets[:, supported_indices]
    supported_probabilities = probabilities[:, supported_indices]

    micro_fpr, micro_tpr, _ = roc_curve(
        supported_targets.reshape(-1),
        supported_probabilities.reshape(-1),
    )
    micro_auc = float(
        roc_auc_score(
            supported_targets.reshape(-1),
            supported_probabilities.reshape(-1),
        )
    )
    micro_ap = float(
        average_precision_score(
            supported_targets.reshape(-1),
            supported_probabilities.reshape(-1),
        )
    )

    macro_fpr = np.linspace(0.0, 1.0, 1001)
    macro_tpr = np.mean(
        [np.interp(macro_fpr, curve["fpr"], curve["tpr"]) for curve in curves],
        axis=0,
    )
    macro_tpr[0] = 0.0
    macro_tpr[-1] = 1.0

    macro_ap = float(np.mean([curve["average_precision"] for curve in curves]))
    macro_auc = float(np.mean([curve["roc_auc"] for curve in curves]))

    csv_path = os.path.join(output_dir, "exercise_ap_roc.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "class_index", "raw_exercise_id", "exercise_name", "support",
            "negative_count", "average_precision", "roc_auc", "status", "reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    curve_path = os.path.join(output_dir, "exercise_roc_curves.svg")
    _save_roc_svg(
        curves=curves,
        micro_curve={"fpr": micro_fpr, "tpr": micro_tpr, "roc_auc": micro_auc},
        macro_curve={"fpr": macro_fpr, "tpr": macro_tpr, "roc_auc": macro_auc},
        output_path=curve_path,
        epoch=epoch,
    )

    summary = {
        "best_epoch": int(epoch),
        "validation_accuracy_percent": float(validation_accuracy),
        "num_validation_samples": int(targets.size),
        "num_model_classes": int(num_classes),
        "num_evaluated_classes": int(len(curves)),
        "num_skipped_classes": int(num_classes - len(curves)),
        "macro_average_precision_supported_classes": macro_ap,
        "macro_roc_auc_supported_classes": macro_auc,
        "micro_average_precision_supported_classes": micro_ap,
        "micro_roc_auc_supported_classes": micro_auc,
        "csv_path": csv_path,
        "roc_svg_path": curve_path,
    }
    with open(
        os.path.join(output_dir, "exercise_ap_roc_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    return summary
