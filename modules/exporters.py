"""Export rankings, portfolio, indicators and report files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_excel(output_dir: Path, ranking: pd.DataFrame, portfolio: pd.DataFrame, indicators: pd.DataFrame) -> None:
    """Write Excel artifacts using OpenPyXL."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_excel(output_dir / "ranking.xlsx", index=False, engine="openpyxl")
    portfolio.to_excel(output_dir / "carteira.xlsx", index=False, engine="openpyxl")
    indicators.to_excel(output_dir / "indicadores.xlsx", index=False, engine="openpyxl")


def export_pdf_report(output_dir: Path, metrics: dict[str, float]) -> Path:
    """Generate a lightweight PDF report without optional binary dependencies."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "relatorio.pdf"
    text = "Relatorio Quantitativo Value Investing\\n" + "\\n".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    stream = f"BT /F1 12 Tf 72 760 Td ({text[:900].replace('(', '[').replace(')', ']')}) Tj ET"
    pdf = f"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n5 0 obj<</Length {len(stream)}>>stream\n{stream}\nendstream endobj\ntrailer<</Root 1 0 R>>\n%%EOF"
    path.write_text(pdf, encoding="latin-1")
    return path
