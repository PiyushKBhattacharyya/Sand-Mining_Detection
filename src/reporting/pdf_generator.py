"""
PDF Incident Report Generator
Generates a professional field-ready PDF report with:
- Cover page with mission summary
- Incident table with severity color coding
- GPS coordinates per detection
- Embedded evidence images (from data/detections/)
"""
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from io import BytesIO

from fpdf import FPDF, XPos, YPos
# from fpdf.enums import TableCellFillMode

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Severity  RGB color
SEVERITY_COLORS = {
    "CRITICAL": (220, 38,  38),   # Red
    "HIGH":     (234, 88,  12),   # Orange
    "MEDIUM":   (202, 138, 4),    # Amber
    "LOW":      (22,  163, 74),   # Green
}


class IncidentReportPDF(FPDF):
    """Custom FPDF subclass with header/footer and tactical styling."""

    BRAND_DARK  = (8,  14, 30)
    BRAND_CYAN  = (0,  220, 255)
    BRAND_WHITE = (230, 236, 248)
    BRAND_GRAY  = (100, 116, 139)

    def __init__(self, mission_id = "BRH-01"):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.mission_id  = mission_id
        self.report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(left=16, top=16, right=16)

    def header(self):
        self.set_fill_color(*self.BRAND_DARK)
        self.rect(0, 0, 210, 14, style="F")
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*self.BRAND_CYAN)
        self.set_y(4)
        self.cell(0, 6, f"BRAHMAPUTRA DRONE SURVEILLANCE  |  Mission: {self.mission_id}  |  CONFIDENTIAL", align="C")
        self.ln(12)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*self.BRAND_GRAY)
        self.cell(0, 6, f"Page {self.page_no()}/{{nb}}  |  Generated: {self.report_time}", align="C")


def _section_title(pdf, title):
    pdf.set_fill_color(16, 24, 50)
    pdf.set_text_color(0, 220, 255)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 9, f"  {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.ln(2)
    pdf.set_text_color(220, 228, 240)


def generate_incident_report(
    incidents,
    output_path = None,
    mission_id = "BRH-01"
):
    """
    Generates a full PDF report and returns raw bytes.
    Also saves to output_path if provided.

    incidents: list of dicts from GET /api/incidents (+ optional 'detections' list)
    """
    pdf = IncidentReportPDF(mission_id=mission_id)
    pdf.alias_nb_pages()

    #  Cover Page 
    pdf.add_page()

    # Title block
    pdf.set_fill_color(8, 14, 30)
    pdf.rect(0, 14, 210, 60, style="F")

    pdf.set_y(22)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(0, 220, 255)
    pdf.cell(0, 12, "ILLEGAL SAND MINING", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 12, "INCIDENT REPORT", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(140, 160, 200)
    pdf.cell(0, 8, f"Mission ID: {mission_id}  |  Brahmaputra River Corridor, Guwahati", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 8, f"Report Generated: {datetime.now().strftime('%d %b %Y  %H:%M')} IST", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(20)

    # Summary stats box
    total     = len(incidents)
    critical  = sum(1 for i in incidents if i.get("severity") == "CRITICAL")
    high      = sum(1 for i in incidents if i.get("severity") == "HIGH")
    medium    = sum(1 for i in incidents if i.get("severity") == "MEDIUM")
    low       = sum(1 for i in incidents if i.get("severity") == "LOW")

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(220, 228, 240)
    pdf.cell(0, 8, "SUMMARY STATISTICS", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    stats = [
        ("Total Incidents Detected", str(total),  (60, 80, 120)),
        ("CRITICAL Severity",         str(critical), (180, 30, 30)),
        ("HIGH Severity",             str(high),     (180, 80, 20)),
        ("MEDIUM / LOW",              f"{medium} / {low}", (100, 100, 40)),
    ]
    for label, val, color in stats:
        pdf.set_fill_color(*color)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(130, 9, f"  {label}", fill=True)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(48, 9, val, align="C", fill=True,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    #  Incident Table 
    pdf.add_page()
    _section_title(pdf, "DETECTED INCIDENTS LOG")

    # Table header
    col_widths = [12, 34, 30, 30, 24, 48]
    headers    = ["ID", "Timestamp", "Latitude", "Longitude", "Severity", "Zone Status"]

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(14, 22, 48)
    pdf.set_text_color(0, 220, 255)

    for w, h in zip(col_widths, headers):
        pdf.cell(w, 8, h, border=0, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    row_bg = [(18, 26, 52), (12, 18, 40)]

    for i, inc in enumerate(incidents):
        sev   = inc.get("severity", "UNKNOWN")
        color = SEVERITY_COLORS.get(sev, (100, 100, 100))

        # Row bg alternates
        pdf.set_fill_color(*row_bg[i % 2])
        pdf.set_text_color(210, 220, 235)

        ts = str(inc.get("timestamp", ""))[:19]
        vals = [
            str(inc.get("id", "")),
            ts,
            f"{inc.get('centroid_latitude', 0):.5f}",
            f"{inc.get('centroid_longitude', 0):.5f}",
            "",    # Severity cell gets special coloring
            "ILLEGAL ZONE" if inc.get("illegal_zone") else "Legal Zone",
        ]

        # Draw all cells except severity
        for j, (w, v) in enumerate(zip(col_widths, vals)):
            if j == 4:
                # Severity cell with color
                pdf.set_fill_color(*color)
                pdf.set_text_color(255, 255, 255)
                pdf.set_font("Helvetica", "B", 7)
                pdf.cell(w, 7, sev, border=0, fill=True, align="C")
                pdf.set_fill_color(*row_bg[i % 2])
                pdf.set_text_color(210, 220, 235)
                pdf.set_font("Helvetica", "", 8)
            else:
                pdf.cell(w, 7, v, border=0, fill=True, align="C")
        pdf.ln()

    #  Evidence Image Gallery 
    evidence_dir = PROJECT_ROOT / "data" / "detections"
    evidence_files = sorted(evidence_dir.glob("evidence_*.jpg"))

    if evidence_files:
        pdf.add_page()
        _section_title(pdf, "EVIDENCE IMAGE GALLERY")

        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(140, 160, 190)
        pdf.cell(0, 6, f"Total evidence frames captured: {len(evidence_files)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)

        col = 0
        x_positions = [16, 110]
        img_w, img_h = 85, 55

        for ef in evidence_files[:20]:   # Max 20 images per report
            x = x_positions[col]
            y = pdf.get_y()

            if y + img_h + 15 > 270:
                pdf.add_page()
                y = pdf.get_y()

            try:
                pdf.image(str(ef), x=x, y=y, w=img_w, h=img_h)
                # Caption
                pdf.set_xy(x, y + img_h + 1)
                pdf.set_font("Helvetica", "", 6)
                pdf.set_text_color(100, 120, 150)
                pdf.cell(img_w, 4, ef.stem[:40], align="C")
            except Exception as e:
                logger.warning(f"Could not embed image {ef.name}: {e}")

            col += 1
            if col >= 2:
                col = 0
                pdf.ln(img_h + 8)

    #  GPS Coordinates Appendix 
    pdf.add_page()
    _section_title(pdf, "GPS COORDINATES  INCIDENT SITES")
    pdf.set_font("Courier", "", 8)
    pdf.set_text_color(180, 200, 230)

    for inc in incidents:
        lat = inc.get("centroid_latitude", 0)
        lon = inc.get("centroid_longitude", 0)
        sev = inc.get("severity", "?")
        inc_id = inc.get("id", "?")
        ts  = str(inc.get("timestamp", ""))[:19]
        line = f"INC-{inc_id:03}  [{sev:8}]  LAT: {lat:.6f}  LON: {lon:.6f}  @ {ts}"
        pdf.cell(0, 6, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    raw_bytes = pdf.output()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(raw_bytes)
        logger.info(f"Report saved: {output_path}")

    return raw_bytes
