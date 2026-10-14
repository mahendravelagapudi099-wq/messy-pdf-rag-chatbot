"""Generate a small, redistributable demo corpus of "messy" PDFs.

These documents are synthetic -- written by this script, not scraped from any
real scheme -- and are deliberately awkward so that the pipeline is exercised on
the things that actually break real-world extraction:

* multi-column layout (text placed in two frames per page)
* a bordered table spanning several rows
* hyphenated words broken across lines
* a scanned-style page (text rendered as an image, so there is no text layer)
* running headers and page numbers to be stripped
* a deliberately blank page

Run with::

    python scripts/make_demo_corpus.py --out data/corpus

The corresponding evaluation questions live in data/eval_dataset.jsonl.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------
# Document content
# --------------------------------------------------------------------------

RBI_PAGES = [
    # page 1
    [
        ("H1", "Reserve Bank of India"),
        ("SUB", "Department of Regulation / Circular No. DPSS.CO.PD No. 118/21.04.048/2025-26"),
        ("P", "Date: March 14, 2026"),
        ("P", "To: The Chairpersons / Managing Directors / CEOs of all Commercial Banks, "
              "Payment System Participants and System Providers"),
        ("H2", "1. Preamble"),
        ("P", "The Reserve Bank of India, in exercise of the powers conferred on it under "
              "sections 25A and 25B of the Banking Regulation Act, 1949 and sections 14 "
              "and 16 of the Payment and Settlement Systems Act, 2007, hereby issues the "
              "following directions to protect customers from fraud."),
        ("P", "These directions come into force with immediate effect and supersede all "
              "earlier circulars issued on the subject matter described below."),
        ("H2", "2. Enhanced Due Diligence"),
        ("P", "All regulated entities shall carry out enhanced due diligence of customers "
              "including the authentication of information, irrespective of the risk "
              "profile of the customer and the volume of the transaction."),
        ("P", "The due diligence process shall include, at a minimum, the measures set out "
              "in the table below."),
        ("H2", "3. Compliance Threshold"),
        ("P", "A customer transaction above the threshold of INR 50,000 per annum shall "
              "require an enhanced certification from the responsible officer under the "
              "Master Direction. Transactions below this threshold may be certified at "
              "manager level."),
    ],
    # page 2 -- table lives here
    [
        ("H2", "4. Measures by Risk Category"),
        ("P", "The measures prescribed for each category of customer risk are set out below."),
        ("P", "Table 1: Due diligence measures by risk category"),
        ("TABLE", [
            ["Risk category", "Customer type", "Measure", "Frequency"],
            ["Low", "Salaried individuals with salary account", "Verify address once at onboarding", "Once"],
            ["Medium", "Self-employed and small business", "Verify address and source of income", "Every 2 years"],
            ["High", "Politically exposed persons", "Verify address, source of funds and senior management approval", "Every year"],
            ["Very high", "Cross-border correspondent banking", "Enhanced due diligence plus transaction monitoring", "Continuous"],
        ]),
        ("H2", "5. Reporting"),
        ("P", "Every regulated entity shall furnish a half-yearly return of suspicious "
              "transactions to the Central Office of the Financial Intelligence Unit in the "
              "prescribed form, within thirty days of the close of the half year ending on "
              "the 31st of March and the 30th of September respectively."),
        ("P", "The officer designated under the Master Direction shall be the nodal officer "
              "for the purpose of all reporting under these directions."),
    ],
    # page 3
    [
        ("H2", "6. Application of KYC to Digital Lending"),
        ("P", "A digital lending platform must be able to demonstrate that it has obtained "
              "the explicit consent of the borrower and that the terms of the loan, the "
              "annualised interest rate and the penalty for late payment are disclosed at "
              "the point of sanction and at the point of disbursal."),
        ("P", "Key elec-", "trical requirements"),
        ("P", "The lender must provide a digitally signed key loan agreement, a key fact "
              "statement and a cooling off period of at least three days during which the "
              "borrower may exit without penalty."),
        ("P", "A one-time processing fee of not more than three per cent of the sanctioned "
              "amount may be charged and must be disclosed upfront in the key fact statement."),
        ("H2", "7. Non-compliance"),
        ("P", "Non-compliance with these directions will attract the supervisory action laid "
              "down in sections 35A and 35AD of the Banking Regulation Act, 1949, and may "
              "include a monetary penalty for each day of continued violation."),
    ],
]

SCHEME_PAGES = [
    [
        ("H1", "Government of India"),
        ("SUB", "Ministry of Rural Development"),
        ("H2", "Pradhan Mantri Gram Sadak Yojana - Programme Guidelines"),
        ("P", "Reference: PMGSY-G/2026/PMGSY-GUIDELINES/001"),
        ("P", "1. Short Title and Commencement"),
        ("P", "These guidelines may be called the Pradhan Mantri Gram Sadak Yojana "
              "Guidelines, 2026. They come into force on the date of their publication in "
              "the Official Gazette."),
        ("P", "2. Objective"),
        ("P", "The objective of the scheme is to provide all-weather rural connectivity to "
              "every eligible rural household and to improve the quality of existing rural "
              "roads so that the assured all-weather road length is achieved."),
        ("P", "3. Definitions"),
    ],
    [
        ("H2", "3.1 Terms used in these guidelines"),
        ("P", "In these guidelines, unless the context otherwise requires:"),
        ("TABLE", [
            ["Term", "Meaning"],
            ["RCC", "Rural Connectivity Certificate issued under the PMGSY-G norms"],
            ["AWWL", "Assured all-weather road length measured in kilometres"],
            ["VRR", "Vehicle retention rate expressed as a percentage"],
            ["MHC", "Maximum habitat coverage, expressed as a percentage of the project area"],
        ]),
        ("H2", "4. Funding Pattern"),
        ("P", "The funding pattern of the scheme is as given in Table 2."),
        ("P", "Table 2: Funding pattern (percentage of project cost)"),
        ("TABLE", [
            ["Item", "Centre share", "State share", "Total"],
            ["Roads (new)", "60", "40", "100"],
            ["Roads (upgradation)", "75", "25", "100"],
            ["Bridge construction", "90", "10", "100"],
            ["Toll plaza", "100", "0", "100"],
        ]),
        ("H2", "5. Unit of Execution"),
        ("P", "The unit of execution shall be the gram panchayat, which shall be responsible "
              "for execution and maintenance of the roads taken up under the scheme."),
    ],
    [
        ("H2", "6. Selection of Candidate Roads"),
        ("P", "Selection of candidate roads shall follow the prescribed ranking in the Table 3."),
        ("P", "Table 3: Core ranking parameters"),
        ("TABLE", [
            ["Parameter", "Weightage", "Maximum score"],
            ["Population served", "25", "25"],
            ["Traffic intensity", "20", "20"],
            ["Social importance", "15", "15"],
            ["Economic importance", "20", "20"],
            ["Length of the road", "20", "20"],
        ]),
        ("P", "A road scoring a cumulative score of not less than 50 out of 100 shall be "
              "considered for sanction under the scheme."),
        ("H2", "7. Quality Assurance"),
        ("P", "Third party technical audit shall be carried out on not less than 30 per cent "
              "of the length of the roads completed in a financial year."),
    ],
    [
        ("H2", "8. Vehicle Retention Rate"),
        ("P", "The vehicle retention rate shall be not less than 80 per cent after three "
              "years of operation. A road falling below this threshold shall be rectified at "
              "the cost of the executing agency and the amount recovered from the "
              "performance guarantee."),
        ("H2", "9. Maximum Habitat Coverage"),
        ("P", "The maximum habitat coverage shall not be less than 20 per cent of the total "
              "road project area. Where the requirement cannot be met, the shortfall shall "
              "be compensated by plantation over the succeeding three years."),
        ("H2", "10. Fund Utilisation"),
        ("P", "Unspent balances remaining at the end of the financial year shall be reported "
              "to the Ministry and shall be adjusted against the instalment due in the "
              "succeeding year."),
    ],
]

SYLLABUS_PAGES = [
    [
        ("H1", "Department of Mechanical Engineering"),
        ("H2", "Course ME501: Thermodynamics and Fluid Mechanics"),
        ("P", "Credit Hours: 4 | L-T-P: 3-1-0 | Examination: End of Semester"),
        ("P", "Course Objective"),
        ("P", "To provide a working knowledge of the laws of thermodynamics and the "
              "principles governing the flow and application of fluids, and to enable "
              "students to apply them to the analysis of engineering systems."),
        ("P", "Course Outcomes"),
        ("TABLE", [
            ["Code", "Course outcome", "Bloom level"],
            ["CO1", "Apply the laws of thermodynamics to closed and open systems", "Apply"],
            ["CO2", "Analyse the performance of steam and gas power cycles", "Analyse"],
            ["CO3", "Solve governing equations for steady and transient pipe flow", "Apply"],
            ["CO4", "Evaluate pumping power and pipeline design trade-offs", "Evaluate"],
        ]),
        ("P", "Unit 1: Thermodynamic Systems"),
    ],
    [
        ("P", "Thermodynamic system, surroundings and boundary. Open and closed systems, "
              "isolated systems and adiabatic systems. State postulate. Zeroth law of "
              "thermodynamics and the concept of temperature."),
        ("P", "Unit 2: Properties and Relations"),
        ("P", "Intensive and extensive properties. Specific heat at constant pressure and at "
              "constant volume. Characteristic gas equation. Availability and irreversibility."),
        ("P", "Assessment: The assessment is 40 per cent internal and 60 per cent end of "
              "semester. Internal assessment is divided as follows:"),
        ("TABLE", [
            ["Component", "Marks", "Weight"],
            ["Mid-semester examination", "20", "20%"],
            ["Assignment 1", "6", "6%"],
            ["Assignment 2", "6", "6%"],
            ["Laboratory record", "8", "8%"],
        ]),
        ("P", "Unit 3: First and Second Laws"),
    ],
    [
        ("P", "First law for a closed system, enthalpy and entropy. Second law, Carnot "
              "cycles, Clausius and Kelvin-Planck statements. Carnot efficiency for heat "
              "engines."),
        ("P", "Unit 4: Fluid Mechanics"),
        ("P", "Continuity equation and Navier-Stokes equations in one dimension. Bernoulli "
              "equation and its applications. Friction losses in pipe flow, friction factor "
              "and Reynolds number."),
        ("P", "Reforms: A student having attendance below 75 per cent is not eligible to "
              "appear in the end of semester examination in any subject."),
        ("P", "Prerequisite: ME301 Engineering Thermodynamics. Students who have not "
              "completed the prerequisite will be de-registered at the start of the "
              "semester."),
    ],
]

SCANNED_TEXT = (
    "MEASUREMENT OF VISCOSITY OF FLUID USING FALLING SPHERE VISCOMETER\n\n"
    "OBJECTIVE\n"
    "To determine the dynamic viscosity of an unknown liquid using a falling sphere "
    "viscometer and to compare the result with the value obtained from a standard "
    "reference liquid of known viscosity.\n\n"
    "APPARATUS\n"
    "Falling sphere viscometer tube, calibrated spheres of known diameter, stop watch, "
    "thermometer, density meter, and a stop clock with least count of 0.01 second.\n\n"
    "PROCEDURE\n"
    "1. Fill the viscometer tube with the liquid under test up to the reference mark.\n"
    "2. Note the temperature of the liquid using the thermometer and record it.\n"
    "3. Release the calibrated sphere from the top reference mark and start the stop "
    "watch simultaneously.\n"
    "4. Time the interval taken by the sphere to travel between the upper and the lower "
    "reference marks. Repeat the measurement at least five times and take the mean.\n"
    "5. Determine the density of the liquid and of the sphere using the density meter.\n\n"
    "RESULT\n"
    "The dynamic viscosity is computed from the relation mu equals d squared times "
    "rho of the liquid times g times t divided by six times the length times rho of "
    "sphere minus rho of the liquid.\n\n"
    "PRECAUTIONS\n"
    "The tube must be held vertical and free from vibration. The sphere must be clean "
    "and free from adhering air bubbles. The temperature must be recorded for every "
    "measurement because viscosity varies strongly with temperature.\n"
)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

MARGIN = 54
PAGE_W, PAGE_H = 595.0, 842.0  # A4 points


def _wrapped(text: str, font: str, size: float, width: float) -> list[str]:
    """Greedy word wrap using the real font metrics."""
    from reportlab.pdfbase.pdfmetrics import stringWidth

    words, lines, cur = text.split(), [], ""
    for w in words:
        candidate = f"{cur} {w}".strip()
        if stringWidth(candidate, font, size) <= width or not cur:
            cur = candidate
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _render_page(canvas, blocks, *, two_column: bool = False, header: str = "", page_no: int = 0):
    from reportlab.lib.colors import black

    canvas.setFillColor(black)

    if header:
        canvas.setFont("Helvetica-Oblique", 7.5)
        canvas.setFillGray(0.45)
        canvas.drawString(MARGIN, PAGE_H - 34, header)
        canvas.drawRightString(PAGE_W - MARGIN, 34, f"Page {page_no}")
        canvas.setFillColor(black)

    y = PAGE_H - 60
    body_top = y

    if two_column:
        col_w = (PAGE_W - 2 * MARGIN - 18) / 2
        left_x, right_x = MARGIN, MARGIN + col_w + 18
        # Flow content into the left column, then the right one.
        rendered: list[tuple[str, float, object]] = []
        for block in blocks:
            kind = block[0]
            if kind == "TABLE":
                rendered.append(("TABLE", y, block[1]))
                continue
            text = "".join(str(p) for p in block[1:])
            if kind == "H1":
                rendered.append(("H1", y, text))
            elif kind == "H2":
                rendered.append(("H2", y, text))
            elif kind == "SUB":
                rendered.append(("SUB", y, text))
            else:
                rendered.append(("P", y, text))

        for kind, _y, payload in rendered:
            if kind == "TABLE":
                rows = payload
                col_widths = [col_w * r for r in (0.18, 0.28, 0.36, 0.18)]
                y = _draw_table(canvas, rows, MARGIN, y, col_widths, size=6.2, header_text=str(payload[0]))
                y -= 8
                continue
            x = left_x if y > body_top - (body_top - 300) else right_x
            y = _draw_block(canvas, kind, str(payload), x, y, col_w)
    else:
        for block in blocks:
            kind = block[0]
            if kind == "TABLE":
                rows = block[1]
                col_widths = [width * r for width, r in zip(
                    [PAGE_W - 2 * MARGIN], [0.1, 0.2, 0.2, 0.5]
                )]
                y = _draw_table(canvas, rows, MARGIN, y, col_widths, size=8)
                y -= 8
                continue
            text = "".join(str(p) for p in block[1:])
            y = _draw_block(canvas, kind, text, MARGIN, y, PAGE_W - 2 * MARGIN)

    canvas.showPage()


def _draw_block(canvas, kind: str, text: str, x: float, y: float, width: float) -> float:
    specs = {
        "H1": ("Helvetica-Bold", 15, 16, 12),
        "H2": ("Helvetica-Bold", 10.5, 13, 8),
        "SUB": ("Helvetica-Oblique", 9, 11, 8),
        "P": ("Helvetica", 9.2, 12.4, 6),
    }
    font, size, leading, gap = specs.get(kind, specs["P"])
    canvas.setFont(font, size)
    for line in _wrapped(text, font, size, width):
        canvas.drawString(x, y, line)
        y -= leading
    return y - gap


def _draw_table(canvas, rows, x, y, col_widths, *, size=8, header_text=""):
    """Draw a bordered table. The first row is treated as the header."""
    from reportlab.lib.colors import black

    font, bold = "Helvetica", "Helvetica-Bold"
    row_h = size * 1.9
    start_y = y

    for r, row in enumerate(rows):
        cx = x
        for c, cell in enumerate(row):
            cw = col_widths[c] if c < len(col_widths) else col_widths[-1]
            use_font = bold if r == 0 else font
            canvas.setFont(use_font, size)
            canvas.rect(cx, y - row_h, cw, row_h, stroke=1, fill=0)
            # Wrap the cell text inside its box.
            avail = cw - 8
            lines = _wrapped(str(cell), use_font, size, max(avail, 20))[:3]
            ty = y - row_h + row_h - 5
            for line in lines:
                canvas.drawString(cx + 4, ty, line)
                ty -= size * 1.25
            cx += cw
        y -= row_h

    return y


def build_text_pdf(path: Path, pages, *, two_column_doc: bool = False) -> None:
    from reportlab.pdfgen import canvas as canvas_mod

    c = canvas_mod.Canvas(str(path), pagesize=(PAGE_W, PAGE_H))
    header = path.stem.replace("_", " ")
    for i, blocks in enumerate(pages, start=1):
        _render_page(c, blocks, header=header, page_no=i)
    # One deliberately blank page -- extraction must not crash on it.
    c.showPage()
    c.save()


def build_scanned_pdf(path: Path) -> None:
    """Render text to an image so the PDF has NO text layer, forcing OCR."""
    from PIL import Image, ImageDraw
    from reportlab.pdfgen import canvas as canvas_mod
    from reportlab.lib.utils import ImageReader

    scale = 2
    img = Image.new("RGB", (int(PAGE_W * scale), int(PAGE_H * scale)), "white")
    d = ImageDraw.Draw(img)
    y = 60
    for line in SCANNED_TEXT.split("\n"):
        if not line.strip():
            y += 14
            continue
        bold = line.isupper() and len(line) < 60
        size = 26 if bold else 19
        d.text((56, y), line, fill="black")
        y += int(size * 1.55)
        if y > PAGE_H * scale - 80:
            break

    c = canvas_mod.Canvas(str(path), pagesize=(PAGE_W, PAGE_H))
    c.drawImage(ImageReader(img), 0, 0, width=PAGE_W, height=PAGE_H)
    c.showPage()
    c.save()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the synthetic demo corpus.")
    ap.add_argument("--out", default="data/corpus")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("rbi_due_diligence_circular.pdf", lambda p: build_text_pdf(p, RBI_PAGES)),
        ("pmgsy_scheme_guidelines.pdf", lambda p: build_text_pdf(p, SCHEME_PAGES)),
        ("me501_thermodynamics_syllabus.pdf", lambda p: build_text_pdf(p, SYLLABUS_PAGES)),
        ("viscosity_lab_manual_scanned.pdf", build_scanned_pdf),
    ]
    for name, fn in jobs:
        target = out / name
        fn(target)
        print(f"  wrote {target} ({target.stat().st_size:,} bytes)")

    print(f"\nDemo corpus ready in {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())