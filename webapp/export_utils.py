"""
webapp/export_utils.py
-------------------------
Builds downloadable CSV and PDF exports of a prediction result. Kept
separate from app.py so the route handlers stay thin -- these functions
take the same cached result payload the results page renders from and
return raw bytes ready to send as a file download.
"""

import base64
import csv
import io
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
)


def _fmt(value, spec: str = ".4f") -> str:
    """
    Formats a metric value, degrading to a readable placeholder instead of
    raising when the value isn't a number at all (a missing metric key, or a
    None). A report with one blank cell is strictly better than a download
    that 500s.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    # NaN/inf reach here as perfectly valid floats, but "nan" in a report cell
    # reads as a broken export rather than an absent metric.
    if number != number or number in (float("inf"), float("-inf")):
        return "n/a"
    return format(number, spec)


def build_csv(result: dict) -> str:
    """
    Returns CSV text: a metrics summary block, the final signal, then the
    full date-aligned actual/predicted table.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)

    writer.writerow(["Stock Prediction Ensemble -- Export"])
    writer.writerow(["Ticker", result["ticker"]])
    writer.writerow(["Generated", datetime.now().isoformat(timespec="seconds")])
    writer.writerow([])

    writer.writerow(["Metric", "Linear Regression", "LSTM"])
    for label, key in [
        ("RMSE", "rmse"), ("MAE", "mae"), ("R2 Score", "r2"),
        ("MAPE (%)", "mape"), ("Directional Accuracy (%)", "accuracy_pct"),
    ]:
        # float("nan") as the default, NOT "" -- an absent metric key used to
        # reach the :.4f format spec as an empty STRING and raise ValueError,
        # turning a missing metric into a failed CSV download (build_pdf
        # already did this correctly; the CSV path did not).
        writer.writerow([label,
                          _fmt(result["lr_eval"].get(key, float("nan"))),
                          _fmt(result["lstm_eval"].get(key, float("nan")))])
    writer.writerow([])

    final = result["final_ensemble"]
    writer.writerow(["Current Price", f"{final['current_price']:.2f}"])
    writer.writerow(["Predicted Next Price (Ensemble)", f"{final['final_predicted_price']:.2f}"])
    writer.writerow(["Predicted % Change", f"{final['predicted_pct_change']:.2f}%"])
    writer.writerow(["Signal", result["final_signal"]["action"]])
    writer.writerow([])

    writer.writerow(["Date", "Actual", "LR Predicted", "LSTM Predicted"])
    for row in result["export_rows"]:
        writer.writerow([
            row["date"],
            f"{row['actual']:.4f}" if row["actual"] is not None else "",
            f"{row['lr_predicted']:.4f}" if row["lr_predicted"] is not None else "",
            f"{row['lstm_predicted']:.4f}" if row["lstm_predicted"] is not None else "",
        ])

    return buf.getvalue()


def build_pdf(result: dict) -> bytes:
    """Returns PDF bytes: a one-page summary report plus the chart images."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                             topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()
    muted = ParagraphStyle("muted", parent=styles["Normal"], textColor=colors.grey, fontSize=9)
    story = []

    story.append(Paragraph(f"Stock Prediction Report &mdash; {result['ticker']}", styles["Title"]))
    story.append(Paragraph(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", muted))
    story.append(Spacer(1, 16))

    final = result["final_ensemble"]
    signal = result["final_signal"]["action"]
    story.append(Paragraph(
        f"Current price: <b>{final['current_price']:.2f}</b> &nbsp;&nbsp; "
        f"Predicted next: <b>{final['final_predicted_price']:.2f}</b> "
        f"({final['predicted_pct_change']:+.2f}%) &nbsp;&nbsp; "
        f"Signal: <b>{signal}</b>", styles["Normal"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(result["final_signal"]["reasoning"], muted))
    story.append(Spacer(1, 16))

    story.append(Paragraph("Price Error &amp; Directional Accuracy", styles["Heading2"]))
    table_data = [["Metric", "Linear Regression", "LSTM"]]
    for label, key, fmt in [
        ("RMSE", "rmse", "{:.3f}"), ("MAE", "mae", "{:.3f}"),
        ("R2 Score", "r2", "{:.3f}"), ("MAPE (%)", "mape", "{:.2f}%"),
        ("Directional Accuracy (%)", "accuracy_pct", "{:.1f}%"),
    ]:
        table_data.append([
            label,
            fmt.format(result["lr_eval"].get(key, float("nan"))),
            fmt.format(result["lstm_eval"].get(key, float("nan"))),
        ])
    t = Table(table_data, colWidths=[200, 150, 150])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A2133")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 20))

    if result["charts"].get("comparison"):
        story.append(Paragraph("Actual vs Predicted", styles["Heading2"]))
        img_bytes = base64.b64decode(result["charts"]["comparison"])
        story.append(Image(io.BytesIO(img_bytes), width=6.5 * inch, height=3.25 * inch))
        story.append(Spacer(1, 12))

    if result["charts"].get("ensemble"):
        story.append(PageBreak())
        story.append(Paragraph("Ensemble Comparison", styles["Heading2"]))
        img_bytes = base64.b64decode(result["charts"]["ensemble"])
        story.append(Image(io.BytesIO(img_bytes), width=5 * inch, height=3.1 * inch))

    story.append(Spacer(1, 24))
    story.append(Paragraph(
        "Educational tool, not investment advice. Predictions come from statistical "
        "models and LLM opinions -- none of them can reliably predict markets.", muted))

    doc.build(story)
    return buf.getvalue()
