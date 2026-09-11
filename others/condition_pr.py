import csv
import html
import json
import os
import re

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


# 원본 vocab의 raw ID를 사용한다. 그래프와 파일명에는 영문만 출력한다.
EXERCISE_NAMES_EN = {
    22: "Pull-up",
    23: "Lat Pulldown",
    24: "Dips",
    25: "Cable Pushdown",
    26: "Rowing Machine",
    27: "Face Pull",
    28: "Cable Crunch",
    29: "Hanging Leg Raise",
    30: "Y Exercise",
    31: "Forward Dynamic Lunge",
    32: "Burpee Test",
    33: "Standing Knee-up",
    34: "Standing Side Crunch",
    35: "Backward Dynamic Lunge",
    38: "Barbell/Dumbbell Row",
    47: "Barbell Squat",
    48: "Barbell Deadlift",
    49: "Barbell Lunge",
}


CONDITION_NAMES_KO = {
    54: "시선 위쪽 유지",
    55: "이완 시 숄더패킹",
    56: "몸통 흔들림 없음",
    57: "수축 시 몸통-팔꿈치 사이 모아줌",
    58: "수축 시 적당한 상체 젖힘",
    59: "손목의 중립",
    60: "상체 살짝 숙임 유지",
    61: "수축 시 고개 안 젖힘",
    62: "이완 시 팔꿈치 각도 90도",
    63: "팔꿈치-몸통의 적당한 거리",
    64: "팔꿈치 위치 고정",
    65: "이완 시 팔 긴장 유지",
    66: "상체 과도한 숙임 없음",
    67: "체스트 업 유지",
    68: "상체 과도한 젖힘 없음",
    69: "수축 시 팔 당김보다 다리 펴짐이 우선",
    70: "이완 시 팔 펴짐이 다리 접힘보다 우선",
    71: "어깨 으쓱 없음",
    72: "수축 시 팔꿈치 각도 90도",
    73: "수축 시 양 손과 이마 동일선상 위치",
    74: "상완의 외회전",
    75: "상완과 전완의 각도 고정",
    76: "팔꿈치와 몸통 각도 고정",
    77: "수축 시 등의 굽힘",
    78: "어깨와 귀 사이 적당한 거리 유지",
    79: "두 다리 사이 모아줌 유지",
    80: "이완 시 다리 긴장 유지",
    81: "양 팔 높이 동일",
    82: "엄지손가락 하늘방향",
    83: "경추 중립 또는 후인(retraction) 유지",
    84: "앞다리 무릎 각도 90도",
    85: "몸통 발 무릎 방향 일치여부",
    86: "뒤다리 무릎 각도 90도",
    87: "척추의 중립",
    88: "상체의 과도한 숙임/젖힘 여부",
    89: "팔 펴고 엎드렸을 때 허리 휨 없음",
    90: "이완 시 팔꿈치 90도",
    91: "가슴의 충분한 이동",
    92: "손의 위치 가슴 중앙 여부",
    93: "고개 젖힘/숙임 여부",
    94: "시선 정면",
    95: "척추 중립",
    96: "무릎 충분히 올라옴",
    97: "이완 시 다리 긴장 유지",
    98: "시선 정면 유지",
    99: "수축 시 무릎과 팔꿈치가 충분히 가까움",
    100: "무릎이 몸통 측면에서 올라오는지 여부",
    101: "양 손이 머리 뒤에 위치",
    104: "무릎 반동 없음",
    105: "상체 반동 없음",
    106: "바벨-덤벨 궤적과 몸 밀착",
    110: "바벨 궤적과 몸 밀착",
    123: "고개 정면",
    124: "발과 무릎의 방향 일치",
    125: "발바닥 지면 고정",
    126: "무릎과 골반이 동시에 펴짐",
}


COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#17becf",
)


def _slugify(value):
    value = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return value or "exercise"


def _polyline(recall, precision, left, top, width, height):
    return " ".join(
        "{:.2f},{:.2f}".format(
            left + float(x_value) * width,
            top + (1.0 - float(y_value)) * height,
        )
        for x_value, y_value in zip(recall, precision)
    )


def _smooth_pr_curve(recall, precision, num_points=301):
    """Build a display-only interpolated precision envelope."""
    recall = np.asarray(recall, dtype=np.float64)
    precision = np.asarray(precision, dtype=np.float64)

    order = np.argsort(recall, kind="stable")
    recall = recall[order]
    precision = precision[order]

    unique_recall, inverse = np.unique(recall, return_inverse=True)
    unique_precision = np.full(unique_recall.shape, -np.inf, dtype=np.float64)
    np.maximum.at(unique_precision, inverse, precision)

    precision_envelope = np.maximum.accumulate(unique_precision[::-1])[::-1]
    recall_grid = np.linspace(
        unique_recall[0],
        unique_recall[-1],
        num_points,
    )
    smooth_precision = np.interp(
        recall_grid,
        unique_recall,
        precision_envelope,
    )
    return recall_grid, np.clip(smooth_precision, 0.0, 1.0)


def _calculate_threshold_metrics(scores, targets, thresholds):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    thresholds = np.asarray(thresholds, dtype=np.float64).reshape(-1)

    if scores.size == 0 or scores.size != targets.size:
        raise ValueError("scores and targets must be non-empty and have the same size")

    threshold_metrics = []
    for threshold in thresholds:
        prediction = scores > threshold
        true_positive = int(np.logical_and(prediction, targets == 1).sum())
        false_positive = int(np.logical_and(prediction, targets == 0).sum())
        false_negative = int(np.logical_and(~prediction, targets == 1).sum())
        correct_count = int((prediction == targets).sum())

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        metrics = {
            "best_threshold": float(threshold),
            "best_precision": float(precision),
            "best_recall": float(recall),
            "best_f1": float(f1),
            "element_wise_binary_accuracy": float(correct_count / targets.size),
            "element_correct_count": correct_count,
            "element_count": int(targets.size),
            "positive_incorrect_count": int((targets == 1).sum()),
            "negative_correct_count": int((targets == 0).sum()),
        }

        threshold_metrics.append(metrics)

    return threshold_metrics


def _find_best_threshold_metrics(scores, targets, thresholds):
    threshold_metrics = _calculate_threshold_metrics(
        scores,
        targets,
        thresholds,
    )
    # thresholds가 오름차순이므로 F1 동률이면 더 낮은 threshold를 선택한다.
    return max(threshold_metrics, key=lambda metrics: metrics["best_f1"])


def _save_element_wise_accuracy_threshold_svg(
    threshold_metrics,
    output_path,
    epoch,
):
    canvas_width, canvas_height = 1200, 760
    left, top = 105, 130
    plot_width, plot_height = 990, 500

    thresholds = np.asarray(
        [metrics["best_threshold"] for metrics in threshold_metrics],
        dtype=np.float64,
    )
    accuracies = np.asarray(
        [
            metrics["element_wise_binary_accuracy"]
            for metrics in threshold_metrics
        ],
        dtype=np.float64,
    )
    best_index = int(np.argmax(accuracies))
    best_threshold = float(thresholds[best_index])
    best_accuracy = float(accuracies[best_index])

    x_min = float(thresholds.min())
    x_max = float(thresholds.max())

    def x_position(threshold):
        return left + (float(threshold) - x_min) / (x_max - x_min) * plot_width

    def y_position(accuracy):
        return top + (1.0 - float(accuracy)) * plot_height

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">'.format(
            canvas_width, canvas_height
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="{}" y="46" font-size="28" font-weight="bold" fill="#17365d">{}</text>'.format(
            left,
            html.escape(
                "Element-wise Binary Accuracy by Threshold "
                "(Validation, Epoch {})".format(epoch)
            ),
        ),
        '<text x="{}" y="82" font-size="17" fill="#475569">{}</text>'.format(
            left,
            html.escape(
                "Applicable condition elements only; "
                "threshold candidates = 0.05 to 0.95 (step 0.05)"
            ),
        ),
    ]

    for accuracy_tick in np.linspace(0.0, 1.0, 11):
        y_value = y_position(accuracy_tick)
        svg.extend([
            '<line x1="{}" y1="{:.2f}" x2="{}" y2="{:.2f}" stroke="#e2e8f0" stroke-width="1"/>'.format(
                left, y_value, left + plot_width, y_value
            ),
            '<text x="{}" y="{:.2f}" font-size="14" text-anchor="end" fill="#475569">{:.1f}</text>'.format(
                left - 12, y_value + 5, accuracy_tick
            ),
        ])

    for threshold in thresholds:
        x_value = x_position(threshold)
        svg.extend([
            '<line x1="{:.2f}" y1="{}" x2="{:.2f}" y2="{}" stroke="#f1f5f9" stroke-width="1"/>'.format(
                x_value, top, x_value, top + plot_height
            ),
            '<text x="{:.2f}" y="{}" font-size="13" text-anchor="end" '
            'transform="rotate(-45 {:.2f} {})" fill="#475569">{:.2f}</text>'.format(
                x_value,
                top + plot_height + 28,
                x_value,
                top + plot_height + 28,
                threshold,
            ),
        ])

    points = " ".join(
        "{:.2f},{:.2f}".format(
            x_position(threshold),
            y_position(accuracy),
        )
        for threshold, accuracy in zip(thresholds, accuracies)
    )
    svg.extend([
        '<rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="#334155" stroke-width="2"/>'.format(
            left, top, plot_width, plot_height
        ),
        '<polyline points="{}" fill="none" stroke="#2563eb" stroke-width="4" '
        'stroke-linejoin="round" stroke-linecap="round"/>'.format(points),
    ])

    for threshold, accuracy in zip(thresholds, accuracies):
        svg.append(
            '<circle cx="{:.2f}" cy="{:.2f}" r="5" fill="#2563eb" stroke="white" stroke-width="2"/>'.format(
                x_position(threshold),
                y_position(accuracy),
            )
        )

    best_x = x_position(best_threshold)
    best_y = y_position(best_accuracy)
    label_x = min(best_x + 20, left + plot_width - 315)
    label_y = max(best_y - 70, top + 15)
    svg.extend([
        '<circle cx="{:.2f}" cy="{:.2f}" r="10" fill="#dc2626" stroke="white" stroke-width="3"/>'.format(
            best_x, best_y
        ),
        '<line x1="{:.2f}" y1="{:.2f}" x2="{:.2f}" y2="{:.2f}" stroke="#dc2626" stroke-width="2"/>'.format(
            best_x, best_y, label_x, label_y + 12
        ),
        '<rect x="{:.2f}" y="{:.2f}" width="300" height="58" rx="9" '
        'fill="#fef2f2" stroke="#dc2626" stroke-width="2"/>'.format(
            label_x, label_y
        ),
        '<text x="{:.2f}" y="{:.2f}" font-size="16" font-weight="bold" fill="#991b1b">{}</text>'.format(
            label_x + 14,
            label_y + 24,
            html.escape("Best element-wise accuracy"),
        ),
        '<text x="{:.2f}" y="{:.2f}" font-size="16" fill="#991b1b">{}</text>'.format(
            label_x + 14,
            label_y + 47,
            html.escape(
                "{:.6f} at threshold {:.2f}".format(
                    best_accuracy,
                    best_threshold,
                )
            ),
        ),
        '<text x="{}" y="{}" font-size="17" text-anchor="middle" fill="#334155">Threshold</text>'.format(
            left + plot_width / 2,
            canvas_height - 35,
        ),
        '<text x="28" y="{}" font-size="17" text-anchor="middle" '
        'transform="rotate(-90 28 {})" fill="#334155">Element-wise Binary Accuracy</text>'.format(
            top + plot_height / 2,
            top + plot_height / 2,
        ),
        '</svg>',
    ])

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(svg))


def _save_exercise_svg(
    exercise_name,
    curves,
    skipped,
    output_path,
    epoch,
    macro_average_precision,
):
    canvas_width, canvas_height = 1600, 900
    left, top = 100, 120
    plot_width, plot_height = 820, 600
    legend_left = 970

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">'.format(
            canvas_width, canvas_height
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="{}" y="42" font-size="26" font-weight="bold">{}</text>'.format(
            left,
            html.escape("{} - Condition PR Curves (Validation, Epoch {})".format(
                exercise_name, epoch
            )),
        ),
        '<text x="{}" y="76" font-size="17" font-weight="bold" fill="#b2182b">{}</text>'.format(
            left,
            html.escape("Positive class (label 1) = Incorrect posture"),
        ),
        '<text x="{}" y="100" font-size="14" fill="#444444">{}</text>'.format(
            left,
            html.escape(
                "Faint lines are raw PR; bold lines are interpolated precision envelopes."
            ),
        ),
    ]

    for tick in np.linspace(0.0, 1.0, 6):
        x_value = left + tick * plot_width
        y_value = top + (1.0 - tick) * plot_height
        svg.extend([
            '<line x1="{0:.1f}" y1="{1}" x2="{0:.1f}" y2="{2}" stroke="#dddddd"/>'.format(
                x_value, top, top + plot_height
            ),
            '<line x1="{0}" y1="{1:.1f}" x2="{2}" y2="{1:.1f}" stroke="#dddddd"/>'.format(
                left, y_value, left + plot_width
            ),
            '<text x="{:.1f}" y="{}" font-size="14" text-anchor="middle">{:.1f}</text>'.format(
                x_value, top + plot_height + 25, tick
            ),
            '<text x="{}" y="{:.1f}" font-size="14" text-anchor="end">{:.1f}</text>'.format(
                left - 12, y_value + 5, tick
            ),
        ])

    evaluated_condition_count = len(curves)
    ap_sum = sum(curve["average_precision"] for curve in curves)
    if macro_average_precision is None:
        macro_ap_text = "Mean posture AP unavailable"
        ap_detail_text = "No evaluable posture conditions"
    else:
        macro_ap_text = "Mean posture AP = {:.4f}".format(
            macro_average_precision
        )
        ap_detail_text = "AP sum {:.4f} / {} evaluated conditions".format(
            ap_sum,
            evaluated_condition_count,
        )

    svg.extend([
        '<rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="black" stroke-width="2"/>'.format(
            left, top, plot_width, plot_height
        ),
        '<text x="{}" y="{}" font-size="17" text-anchor="middle">{}</text>'.format(
            left + plot_width / 2,
            top + plot_height + 65,
            html.escape("Recall = detected incorrect-posture samples / all incorrect-posture samples"),
        ),
        '<text x="28" y="{}" font-size="17" text-anchor="middle" transform="rotate(-90 28 {})">{}</text>'.format(
            top + plot_height / 2,
            top + plot_height / 2,
            html.escape("Precision = correct incorrect-posture alerts / all incorrect-posture alerts"),
        ),
        '<rect x="{}" y="112" width="590" height="82" rx="12" fill="#e0f2fe" stroke="#0284c7"/>'.format(
            legend_left
        ),
        '<text x="{}" y="145" font-size="21" font-weight="bold" fill="#075985">{}</text>'.format(
            legend_left + 20, html.escape(macro_ap_text)
        ),
        '<text x="{}" y="174" font-size="14" fill="#334155">{}</text>'.format(
            legend_left + 20, html.escape(ap_detail_text)
        ),
        '<text x="{}" y="225" font-size="18" font-weight="bold">자세 평가 항목</text>'.format(
            legend_left
        ),
    ])

    legend_y = 260
    for index, curve in enumerate(curves):
        color = COLORS[index % len(COLORS)]
        raw_points = _polyline(
            curve["recall"], curve["precision"], left, top, plot_width, plot_height
        )
        plot_recall, plot_precision = _smooth_pr_curve(
            curve["recall"],
            curve["precision"],
        )
        points = _polyline(
            plot_recall, plot_precision, left, top, plot_width, plot_height
        )
        baseline_y = top + (1.0 - curve["prevalence"]) * plot_height
        label = "{} | AP={:.4f}".format(
            curve["condition_name"], curve["average_precision"]
        )
        support = "incorrect={} / total={} | prevalence={:.3f}".format(
            curve["positive_count"], curve["sample_count"], curve["prevalence"]
        )
        svg.extend([
            '<line x1="{}" y1="{:.2f}" x2="{}" y2="{:.2f}" stroke="{}" stroke-width="1" stroke-dasharray="5,5" opacity="0.28"/>'.format(
                left, baseline_y, left + plot_width, baseline_y, color
            ),
            '<polyline points="{}" fill="none" stroke="{}" stroke-width="1.2" opacity="0.18"/>'.format(
                raw_points, color
            ),
            '<polyline points="{}" fill="none" stroke="{}" stroke-width="3" opacity="0.9" stroke-linejoin="round" stroke-linecap="round"/>'.format(
                points, color
            ),
            '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="4"/>'.format(
                legend_left, legend_y, legend_left + 30, legend_y, color
            ),
            '<text x="{}" y="{}" font-size="14" font-weight="bold">{}</text>'.format(
                legend_left + 40, legend_y + 5, html.escape(label)
            ),
            '<text x="{}" y="{}" font-size="12" fill="#555555">{}</text>'.format(
                legend_left + 40, legend_y + 24, html.escape(support)
            ),
        ])
        legend_y += 82

    for item in skipped:
        label = "{} | AP unavailable: {}".format(
            item["condition_name"], item["reason"].replace("_", " ")
        )
        svg.append(
            '<text x="{}" y="{}" font-size="13" fill="#777777">{}</text>'.format(
                legend_left, legend_y, html.escape(label)
            )
        )
        legend_y += 26

    svg.append("</svg>")
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(svg))


def _save_exercise_mean_ap_table_svg(
    exercise_summaries,
    output_path,
    epoch,
):
    """Save one image table containing the mean posture AP of each exercise."""
    rows = sorted(
        exercise_summaries,
        key=lambda item: item["exercise_raw_id"],
    )
    row_height = 42
    canvas_width = 1400
    table_left = 50
    table_top = 125
    header_height = 50
    canvas_height = table_top + header_height + row_height * len(rows) + 55
    column_x = (70, 185, 675, 845, 1115)

    valid_mean_aps = [
        item["macro_average_precision"]
        for item in rows
        if item["macro_average_precision"] is not None
    ]
    overall_mean_ap = (
        float(np.mean(valid_mean_aps)) if valid_mean_aps else None
    )
    overall_text = (
        "Mean across exercises = {:.4f}".format(overall_mean_ap)
        if overall_mean_ap is not None
        else "Mean across exercises unavailable"
    )

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">'.format(
            canvas_width, canvas_height
        ),
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="{}" y="45" font-size="28" font-weight="bold" fill="#17365d">{}</text>'.format(
            table_left,
            html.escape(
                "Mean Posture AP by Exercise (Validation, Epoch {})".format(epoch)
            ),
        ),
        '<text x="{}" y="80" font-size="17" fill="#475569">{}</text>'.format(
            table_left,
            html.escape(
                overall_text
                + " | Conditions without both positive and negative samples are excluded."
            ),
        ),
        '<rect x="{}" y="{}" width="{}" height="{}" fill="#24446f"/>'.format(
            table_left,
            table_top,
            canvas_width - table_left * 2,
            header_height,
        ),
    ]

    headers = (
        "Raw ID",
        "Exercise",
        "Samples",
        "Evaluated / Applicable Conditions",
        "Mean Posture AP",
    )
    for x_value, header in zip(column_x, headers):
        svg.append(
            '<text x="{}" y="{}" font-size="16" font-weight="bold" fill="white">{}</text>'.format(
                x_value,
                table_top + 32,
                html.escape(header),
            )
        )

    for index, item in enumerate(rows):
        row_top = table_top + header_height + index * row_height
        fill = "#ffffff" if index % 2 == 0 else "#eef3f8"
        mean_ap = item["macro_average_precision"]
        values = (
            str(item["exercise_raw_id"]),
            item["exercise_name"],
            "{:,}".format(item["sample_count"]),
            "{} / {}".format(
                item["evaluated_condition_count"],
                item["condition_count"],
            ),
            "{:.4f}".format(mean_ap) if mean_ap is not None else "N/A",
        )
        svg.extend([
            '<rect x="{}" y="{}" width="{}" height="{}" fill="{}" stroke="#cbd5e1" stroke-width="1"/>'.format(
                table_left,
                row_top,
                canvas_width - table_left * 2,
                row_height,
                fill,
            ),
            *[
                '<text x="{}" y="{}" font-size="15" fill="{}"{}>{}</text>'.format(
                    x_value,
                    row_top + 27,
                    "#075985" if value_index == 4 else "#1e293b",
                    ' font-weight="bold"' if value_index == 4 else "",
                    html.escape(str(value)),
                )
                for value_index, (x_value, value) in enumerate(zip(column_x, values))
            ],
        ])

    svg.append("</svg>")
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(svg))


def save_condition_pr_analysis(
    probabilities,
    targets,
    masks,
    exercise_targets,
    output_dir,
    epoch,
    class_num=32,
    exercise_offset=22,
    exercise_names_by_raw_id=None,
    condition_names_by_raw_id=None,
):
    """Validation 운동별로 applicable condition의 PR curve와 AP를 저장한다.

    targets의 1은 잘못된 자세(positive), 0은 올바른 자세(negative)이다.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    masks = np.asarray(masks, dtype=bool)
    exercise_targets = np.asarray(exercise_targets, dtype=np.int64).reshape(-1)

    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [N, D]")
    if probabilities.shape != targets.shape or targets.shape != masks.shape:
        raise ValueError("probabilities, targets, and masks must have the same [N, D] shape")
    if probabilities.shape[0] == 0 or probabilities.shape[0] != exercise_targets.size:
        raise ValueError("exercise_targets must contain the same non-zero N")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain NaN or infinity")
    if np.any((targets[masks] != 0) & (targets[masks] != 1)):
        raise ValueError("masked targets must be binary labels (0 or 1)")

    os.makedirs(output_dir, exist_ok=True)
    condition_offset = exercise_offset + int(class_num)
    exercise_names = (
        EXERCISE_NAMES_EN
        if exercise_names_by_raw_id is None
        else exercise_names_by_raw_id
    )
    condition_names = (
        CONDITION_NAMES_KO
        if condition_names_by_raw_id is None
        else condition_names_by_raw_id
    )
    threshold_grid = np.arange(1, 20, dtype=np.float64) / 20.0
    global_threshold_metrics = _calculate_threshold_metrics(
        probabilities[masks],
        targets[masks],
        threshold_grid,
    )
    global_best_metrics = max(
        global_threshold_metrics,
        key=lambda metrics: metrics["best_f1"],
    )
    global_best_accuracy_metrics = max(
        global_threshold_metrics,
        key=lambda metrics: metrics["element_wise_binary_accuracy"],
    )
    element_wise_accuracy_curve_path = os.path.join(
        output_dir,
        "condition_element_wise_binary_accuracy_by_threshold.svg",
    )
    _save_element_wise_accuracy_threshold_svg(
        global_threshold_metrics,
        element_wise_accuracy_curve_path,
        epoch,
    )
    rows = []
    exercise_summaries = []
    exercise_threshold_rows = []

    for exercise_local_id in sorted(np.unique(exercise_targets).tolist()):
        exercise_raw_id = int(exercise_local_id) + int(exercise_offset)
        if exercise_raw_id not in exercise_names:
            raise KeyError("missing exercise name for raw ID {}".format(exercise_raw_id))

        exercise_name = exercise_names[exercise_raw_id]
        sample_selector = exercise_targets == exercise_local_id
        condition_local_ids = np.flatnonzero(masks[sample_selector].any(axis=0))
        exercise_probabilities = probabilities[sample_selector]
        exercise_binary_targets = targets[sample_selector]
        exercise_masks = masks[sample_selector]
        exercise_best_metrics = _find_best_threshold_metrics(
            exercise_probabilities[exercise_masks],
            exercise_binary_targets[exercise_masks],
            threshold_grid,
        )
        exercise_threshold_rows.append({
            "epoch": int(epoch),
            "exercise_local_id": int(exercise_local_id),
            "exercise_raw_id": exercise_raw_id,
            "exercise_name": exercise_name,
            "sample_count": int(sample_selector.sum()),
            "applicable_condition_count": int(len(condition_local_ids)),
            "applicable_element_count": exercise_best_metrics["element_count"],
            "positive_incorrect_count": exercise_best_metrics["positive_incorrect_count"],
            "negative_correct_count": exercise_best_metrics["negative_correct_count"],
            "best_threshold": exercise_best_metrics["best_threshold"],
            "best_precision": exercise_best_metrics["best_precision"],
            "best_recall": exercise_best_metrics["best_recall"],
            "best_f1": exercise_best_metrics["best_f1"],
            "element_wise_binary_accuracy": (
                exercise_best_metrics["element_wise_binary_accuracy"]
            ),
        })
        curves = []
        skipped = []

        for condition_local_id in condition_local_ids:
            condition_raw_id = int(condition_local_id) + condition_offset
            if condition_raw_id not in condition_names:
                raise KeyError("missing Korean condition name for raw ID {}".format(condition_raw_id))

            selector = sample_selector & masks[:, condition_local_id]
            binary_target = targets[selector, condition_local_id]
            score = probabilities[selector, condition_local_id]
            positive_count = int(binary_target.sum())
            sample_count = int(binary_target.size)
            negative_count = sample_count - positive_count
            prevalence = float(positive_count / sample_count) if sample_count else 0.0
            row = {
                "epoch": int(epoch),
                "exercise_local_id": int(exercise_local_id),
                "exercise_raw_id": exercise_raw_id,
                "exercise_name": exercise_name,
                "condition_local_id": int(condition_local_id),
                "condition_raw_id": condition_raw_id,
                "condition_name": condition_names[condition_raw_id],
                "sample_count": sample_count,
                "positive_incorrect_count": positive_count,
                "negative_correct_count": negative_count,
                "positive_prevalence": prevalence,
                "average_precision": None,
                "status": "skipped",
                "reason": "no_positive_samples" if positive_count == 0 else "no_negative_samples",
            }

            if positive_count > 0 and negative_count > 0:
                average_precision = float(average_precision_score(binary_target, score))
                precision, recall, thresholds = precision_recall_curve(binary_target, score)
                row.update({
                    "average_precision": average_precision,
                    "status": "evaluated",
                    "reason": "",
                })
                curves.append({
                    "condition_name": row["condition_name"],
                    "positive_count": positive_count,
                    "sample_count": sample_count,
                    "prevalence": prevalence,
                    "average_precision": average_precision,
                    "precision": precision,
                    "recall": recall,
                    "thresholds": thresholds,
                })
            else:
                skipped.append({
                    "condition_name": row["condition_name"],
                    "reason": row["reason"],
                })
            rows.append(row)

        if not curves and not skipped:
            raise ValueError("exercise {} has no applicable condition".format(exercise_name))

        macro_average_precision = (
            float(np.mean([
                curve["average_precision"] for curve in curves
            ]))
            if curves else None
        )

        image_stem = _slugify(exercise_name)
        if image_stem == "exercise":
            image_stem = "exercise_{}".format(exercise_raw_id)
        image_path = os.path.join(output_dir, image_stem + ".svg")
        _save_exercise_svg(
            exercise_name,
            curves,
            skipped,
            image_path,
            epoch,
            macro_average_precision,
        )
        exercise_summaries.append({
            "exercise_raw_id": exercise_raw_id,
            "exercise_name": exercise_name,
            "sample_count": int(sample_selector.sum()),
            "condition_count": int(len(condition_local_ids)),
            "evaluated_condition_count": int(len(curves)),
            "macro_average_precision": macro_average_precision,
            "image_path": image_path,
        })

    exercise_mean_ap_table_path = os.path.join(
        output_dir,
        "condition_mean_ap_by_exercise.svg",
    )
    _save_exercise_mean_ap_table_svg(
        exercise_summaries,
        exercise_mean_ap_table_path,
        epoch,
    )

    csv_path = os.path.join(output_dir, "condition_ap_by_exercise.csv")
    fieldnames = [
        "epoch", "exercise_local_id", "exercise_raw_id", "exercise_name",
        "condition_local_id", "condition_raw_id", "condition_name", "sample_count",
        "positive_incorrect_count", "negative_correct_count", "positive_prevalence",
        "average_precision", "status", "reason",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    exercise_threshold_csv_path = os.path.join(
        output_dir,
        "condition_best_threshold_by_exercise.csv",
    )
    exercise_threshold_fieldnames = [
        "epoch", "exercise_local_id", "exercise_raw_id", "exercise_name",
        "sample_count", "applicable_condition_count", "applicable_element_count",
        "positive_incorrect_count", "negative_correct_count", "best_threshold",
        "best_precision", "best_recall", "best_f1",
        "element_wise_binary_accuracy",
    ]
    with open(
        exercise_threshold_csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=exercise_threshold_fieldnames,
        )
        writer.writeheader()
        writer.writerows(exercise_threshold_rows)

    evaluated_ap = [
        row["average_precision"] for row in rows if row["status"] == "evaluated"
    ]
    summary = {
        "epoch": int(epoch),
        "positive_label": 1,
        "positive_label_meaning": "incorrect_posture",
        "negative_label": 0,
        "negative_label_meaning": "correct_posture",
        "threshold_candidates": [float(value) for value in threshold_grid],
        "best_threshold_selection_metric": "f1",
        "best_threshold": global_best_metrics["best_threshold"],
        "best_precision": global_best_metrics["best_precision"],
        "best_recall": global_best_metrics["best_recall"],
        "best_f1": global_best_metrics["best_f1"],
        "element_wise_binary_accuracy_at_best_threshold": (
            global_best_metrics["element_wise_binary_accuracy"]
        ),
        "element_correct_count_at_best_threshold": (
            global_best_metrics["element_correct_count"]
        ),
        "element_count": global_best_metrics["element_count"],
        "element_wise_binary_accuracy_by_threshold": [
            {
                "threshold": metrics["best_threshold"],
                "element_wise_binary_accuracy": (
                    metrics["element_wise_binary_accuracy"]
                ),
            }
            for metrics in global_threshold_metrics
        ],
        "best_element_wise_binary_accuracy_threshold": (
            global_best_accuracy_metrics["best_threshold"]
        ),
        "best_element_wise_binary_accuracy": (
            global_best_accuracy_metrics["element_wise_binary_accuracy"]
        ),
        "element_wise_binary_accuracy_curve_path": (
            element_wise_accuracy_curve_path
        ),
        "exercise_mean_ap_table_path": exercise_mean_ap_table_path,
        "num_validation_samples": int(probabilities.shape[0]),
        "num_exercise_images": int(len(exercise_summaries)),
        "num_condition_pairs": int(len(rows)),
        "num_evaluated_condition_pairs": int(len(evaluated_ap)),
        "macro_average_precision_over_evaluated_exercise_condition_pairs": (
            float(np.mean(evaluated_ap)) if evaluated_ap else None
        ),
        "csv_path": csv_path,
        "exercise_best_threshold_csv_path": exercise_threshold_csv_path,
        "exercises": exercise_summaries,
    }
    with open(
        os.path.join(output_dir, "condition_pr_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        "[VALID][Epoch {}] Condition PR threshold analysis: "
        "{{'best_threshold': {:.2f}, 'best_precision': {:.6f}, "
        "'best_recall': {:.6f}, 'best_f1': {:.6f}, "
        "'element_wise_binary_accuracy': {:.6f}}} | "
        "exercise thresholds: {} | output: {}".format(
            int(epoch),
            summary["best_threshold"],
            summary["best_precision"],
            summary["best_recall"],
            summary["best_f1"],
            summary["element_wise_binary_accuracy_at_best_threshold"],
            exercise_threshold_csv_path,
            output_dir,
        )
    )

    return summary
