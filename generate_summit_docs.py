#!/usr/bin/env python3
"""
Generate the GlycoTwin summit deck and architecture diagram.

Outputs (all in docs/):
    GlycoTwin_Summit_Presentation.pptx / .pdf   12-slide 16:9 deck
    GlycoTwin_Architecture_Diagram.pptx / .pdf  slide 1: file-level pipeline; slide 2: care pathway
    img/architecture.png                        PNG of slide 1 of the diagram (needs pdftoppm)

Every performance number is read at generation time from models/metrics.json
(written by `python models/evaluate.py`). Cohort facts are computed from
data/synthetic_ehr.csv. Feature lists and hyperparameters are imported from
models/train_fusion_model.py. Nothing about model performance is typed in here.

The PDFs are exported from the PPTX files with LibreOffice, so the PDF and the
PPTX are always the same deck. If `soffice` is not installed, the script prints
the manual export step instead.

Run from anywhere (paths are resolved from this file):
    pip install -r requirements.txt -r requirements-dev.txt
    python generate_summit_docs.py
"""

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_TICK_LABEL_POSITION, XL_TICK_MARK
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
IMG = DOCS / "img"
METRICS_PATH = ROOT / "models" / "metrics.json"
EHR_PATH = ROOT / "data" / "synthetic_ehr.csv"
BUNDLE_PATH = ROOT / "models" / "glycotwin_model.joblib"
DECK_PPTX = DOCS / "GlycoTwin_Summit_Presentation.pptx"
ARCH_PPTX = DOCS / "GlycoTwin_Architecture_Diagram.pptx"
ARCH_PNG = IMG / "architecture.png"

sys.path.insert(0, str(ROOT / "models"))
from train_fusion_model import (  # noqa: E402
    EHR_FEATURES, HGB_PARAMS, HYPER_MG_DL, HYPO_MG_DL, WEARABLE_FEATURES,
)

LIVE_SITE = "glycotwin-app.web.app"
SAFETY_LINE = "Synthetic data only · research prototype, not a medical device · not for dosing decisions"

# ---------------------------------------------------------------------------
# Palette (matches the dashboards: teal = model / in range, red = low, amber = high)
# ---------------------------------------------------------------------------
NAVY = RGBColor(0x0F, 0x17, 0x2A)
NAVY_2 = RGBColor(0x1E, 0x29, 0x3B)
INK = RGBColor(0x1E, 0x29, 0x3B)
BODY = RGBColor(0x33, 0x41, 0x55)
MUTED = RGBColor(0x64, 0x74, 0x8B)
LINE = RGBColor(0xCB, 0xD5, 0xE1)
CARD = RGBColor(0xF1, 0xF5, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ON_DARK = RGBColor(0xE2, 0xE8, 0xF0)
ON_DARK_MUTED = RGBColor(0x94, 0xA3, 0xB8)
TEAL = RGBColor(0x0F, 0x76, 0x6E)
TEAL_MID = RGBColor(0x14, 0xB8, 0xA6)
TEAL_TINT = RGBColor(0xE6, 0xF4, 0xF1)
TEAL_SOFT = RGBColor(0x99, 0xF6, 0xE4)
RED = RGBColor(0xB9, 0x1C, 0x1C)
RED_TINT = RGBColor(0xFE, 0xF2, 0xF2)
AMBER = RGBColor(0xB4, 0x53, 0x09)
AMBER_TINT = RGBColor(0xFF, 0xF7, 0xED)
GREY_BAR = RGBColor(0x94, 0xA3, 0xB8)

FONT = "Calibri"
NBSP = "\u00a0"
ZW = "\u200b"  # zero-width space: lets long file names wrap after "_" or "/"
NBH = "-\u2060"  # hyphen + word joiner: keeps "PR-AUC" on one line
PR_AUC, ROC_AUC = f"PR{NBH}AUC", f"ROC{NBH}AUC"


def rng(a, b):
    """a-b range joined by an en dash that never breaks across lines."""
    return f"{a}\u2060\u2013\u2060{b}"

SLIDE_W, SLIDE_H = 13.333, 7.5

# Screenshot crops (left, top, right, bottom) in pixels of the 1440x900 (phone: 390x844) captures in docs/img.
# Each slide shows only the part it discusses, so the text stays legible when projected. If the screenshots
# are re-captured with a different layout, adjust these boxes and check the rendered slides.
SHOT_CROPS = {
    "hero": ("streamlit_live_monitor.png", (992, 262, 1362, 866)),    # current CGM card + model alert card
    "live": ("streamlit_live_monitor.png", (400, 155, 1362, 866)),    # patient line, 8 h chart, alert card
    "phone": ("web_mobile.png", (12, 462, 378, 830)),                 # phone: current CGM + alert card
    "board": ("streamlit_cohort.png", (400, 455, 1337, 876)),         # risk board: badges + top rows to Δ 15 min
    "whatif": ("streamlit_what_if.png", (752, 386, 1362, 896)),       # sandbox: result tiles, chart, legend
}


def shot(key):
    """(path, crop fractions) for a SHOT_CROPS entry, or (None, None) if the screenshot is missing."""
    name, box = SHOT_CROPS[key]
    path = IMG / name
    return (path, px_crop(path, *box)) if path.exists() else (None, None)


# ---------------------------------------------------------------------------
# Inputs: metrics.json, EHR CSV, model bundle, README team table
# ---------------------------------------------------------------------------
def load_inputs():
    if not METRICS_PATH.exists():
        sys.exit(f"Missing {METRICS_PATH}. Run `python models/evaluate.py` first.")
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    ehr = pd.read_csv(EHR_PATH, keep_default_na=False)  # keeps the literal "None" insulin regimen

    test_ids = None
    try:
        import joblib
        bundle = joblib.load(BUNDLE_PATH)
        test_ids = set(bundle.get("test_patients", []))
    except Exception as exc:  # the deck can still be built without the bundle
        print(f"  note: could not read {BUNDLE_PATH.name} ({exc}); held-out insulin count omitted")

    insulin = ehr["insulin_regimen"] != "None"
    sulfonylurea = ehr["primary_medication"].str.contains("Sulfonylurea")
    cohort = {
        "n": len(ehr),
        "hba1c_min": ehr["baseline_hba1c"].min(),
        "hba1c_max": ehr["baseline_hba1c"].max(),
        "hba1c_mean": ehr["baseline_hba1c"].mean(),
        "hba1c_ge7": int((ehr["baseline_hba1c"] >= 7.0).sum()),
        "sulfonylurea": int(sulfonylurea.sum()),
        "insulin": int(insulin.sum()),
        "insulin_heldout": (int(ehr.loc[insulin, "patient_id"].isin(test_ids).sum()) if test_ids else None),
        "egfr_lt60": int((ehr["egfr"] < 60).sum()),
        "egfr_min": ehr["egfr"].min(),
        "tiers": ehr["clinical_risk_tier"].value_counts().to_dict(),
    }
    return metrics, cohort, read_team()


def read_team():
    """Team details exactly as written in README.md (falls back to the same text)."""
    team = {
        "team": "Classy Leopards",
        "college": "Goa Institute of Management, Sanquelim, Goa",
        "leader": "Shivam Raj",
        "members": ["Shubham Kumar", "Shiva Kumar Bachatty", "Yashita Dube"],
        "repo": "https://github.com/kumarshubhammorya/glycotwin-digital-twin",
        "video": None,
    }
    readme = ROOT / "README.md"
    if not readme.exists():
        return team
    txt = readme.read_text(encoding="utf-8")

    def field(label):
        m = re.search(r"\|\s*\*\*" + re.escape(label) + r"\*\*\s*\|\s*\*\*(.+?)\*\*", txt)
        return m.group(1).strip() if m else None

    found = {
        "team": field("Team Name"),
        "college": field("College / Institute"),
        "leader": field("Team Leader"),
    }
    members = [field(f"Team Member {i}") for i in (2, 3, 4)]
    repo = re.search(r"\*\*Public GitHub Repository\*\*\s*\|\s*\[(https://github\.com/[^\]\s]+)\]", txt)
    for k, v in found.items():
        if v:
            team[k] = v
    if all(members):
        team["members"] = members
    if repo:
        team["repo"] = repo.group(1)
    # the video link, once the team pastes it into README section 5 (the deck points to the README otherwise)
    sec = re.search(r"^## 5\..*?(?=^## 6\.)", txt, re.S | re.M)
    links = re.findall(r"https://(?:www\.)?(?:youtube\.com/watch\?v=[\w-]{6,}|youtu\.be/[\w-]{6,}|"
                       r"drive\.google\.com/[^\s)\]]+)", sec.group(0) if sec else "")
    links = [u for u in links if "YOUR_VIDEO_ID" not in u.upper()]
    if links:
        team["video"] = links[0]
    return team


# ---------------------------------------------------------------------------
# Number formatting (inputs come from metrics.json)
# ---------------------------------------------------------------------------
def pct(frac, d=1):
    return f"{frac * 100:.{d}f}%"


def pctv(value, d=1):
    return f"{value:.{d}f}%"


def num(x, d=1):
    return f"{x:,.{d}f}"


def mgdl(x, d=1):
    return f"{x:.{d}f}{NBSP}mg/dL"


def intc(x):
    return f"{int(x):,}"


# ---------------------------------------------------------------------------
# Low-level drawing helpers (python-pptx)
# ---------------------------------------------------------------------------
def _rgb(c):
    return c if isinstance(c, RGBColor) else RGBColor.from_string(c)


def setup_frame(tf, margin=0.06, anchor=MSO_ANCHOR.TOP):
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    m = Inches(margin)
    tf.margin_left = tf.margin_right = m
    tf.margin_top = tf.margin_bottom = Inches(min(margin, 0.05))
    tf.vertical_anchor = anchor


def _set_bullet(p, color, indent=0.24):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(indent))))
    pPr.set("indent", str(-int(Inches(indent))))
    for tag in ("a:buClr", "a:buFont", "a:buNone", "a:buChar", "a:buAutoNum"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    clr = etree.SubElement(pPr, qn("a:buClr"))
    srgb = etree.SubElement(clr, qn("a:srgbClr"))
    srgb.set("val", str(color))
    font = etree.SubElement(pPr, qn("a:buFont"))
    font.set("typeface", "Arial")
    bu = etree.SubElement(pPr, qn("a:buChar"))
    bu.set("char", "•")


def fill_text(tf, paras, size=16, color=BODY, bold=False, align=PP_ALIGN.LEFT,
              space_after=4, bullet_color=TEAL, line_spacing=None):
    """paras: list of str | dict(runs=[str | (text, opts)], bullet, size, color, bold, align, space_after)."""
    first = True
    for para in paras:
        if isinstance(para, str):
            para = {"runs": [para]}
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = para.get("align", align)
        p.space_after = Pt(para.get("space_after", space_after))
        if para.get("space_before"):
            p.space_before = Pt(para["space_before"])
        if line_spacing or para.get("line_spacing"):
            p.line_spacing = para.get("line_spacing", line_spacing)
        if para.get("bullet"):
            _set_bullet(p, para.get("bullet_color", bullet_color), para.get("indent", 0.24))
        for run in para["runs"]:
            text, opts = (run, {}) if isinstance(run, str) else run
            r = p.add_run()
            r.text = text
            f = r.font
            f.name = FONT
            f.size = Pt(opts.get("size", para.get("size", size)))
            f.bold = opts.get("bold", para.get("bold", bold))
            f.italic = opts.get("italic", False)
            f.color.rgb = _rgb(opts.get("color", para.get("color", color)))


def add_text(slide, x, y, w, h, paras, margin=0.04, anchor=MSO_ANCHOR.TOP, name=None, **kw):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    setup_frame(tb.text_frame, margin, anchor)
    fill_text(tb.text_frame, paras, **kw)
    return tb


def add_box(slide, x, y, w, h, fill=CARD, line=None, radius=0.06, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
            paras=None, margin=0.14, anchor=MSO_ANCHOR.TOP, line_width=1.0, dash=None, name=None, **kw):
    sh = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        sh.name = name
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = radius
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(line_width)
        if dash:
            sh.line.dash_style = dash
    sh.shadow.inherit = False
    style = sh._element.find(qn("p:style"))
    if style is not None:  # drop the theme style: no theme shadow, no theme font colour
        sh._element.remove(style)
    setup_frame(sh.text_frame, margin, anchor)
    if paras:
        fill_text(sh.text_frame, paras, **kw)
    return sh


def _arrowhead(line_fmt, head=False):
    ln = line_fmt._get_or_add_ln()
    for tag in ("a:headEnd", "a:tailEnd"):
        for el in ln.findall(qn(tag)):
            ln.remove(el)
    if head:
        h = etree.SubElement(ln, qn("a:headEnd"))
        h.set("type", "triangle")
    t = etree.SubElement(ln, qn("a:tailEnd"))
    t.set("type", "triangle")
    t.set("w", "med")
    t.set("len", "med")


def add_line(slide, x1, y1, x2, y2, color=MUTED, width=1.75, arrow=True, dash=None):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    if dash:
        c.line.dash_style = dash
    if arrow:
        _arrowhead(c.line)
    return c


def add_picture(slide, path, x, y, w=None, h=None, border=LINE, crop=None):
    kw = {}
    if w is not None:
        kw["width"] = Inches(w)
    if h is not None:
        kw["height"] = Inches(h)
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)
    if crop:
        # crop = (left, top, right, bottom) fractions; keep the requested width and fix the aspect ratio
        l, t, r, b = crop
        pic.crop_left, pic.crop_top, pic.crop_right, pic.crop_bottom = l, t, r, b
        img_w, img_h = pic.image.size
        vis_ratio = (img_w * (1 - l - r)) / (img_h * (1 - t - b))
        if w is not None:
            pic.height = Emu(int(Inches(w) / vis_ratio))
        elif h is not None:
            pic.width = Emu(int(Inches(h) * vis_ratio))
    if border is not None:
        pic.line.color.rgb = border
        pic.line.width = Pt(1)
    return pic


def px_crop(path, left, top, right, bottom):
    """Crop box in image pixels -> (left, top, right, bottom) fractions for add_picture(crop=...)."""
    from PIL import Image
    with Image.open(path) as im:
        img_w, img_h = im.size
    return (left / img_w, top / img_h, 1 - right / img_w, 1 - bottom / img_h)


def circle_badge(slide, x, y, d, text, fill=TEAL, color=WHITE, size=16):
    return add_box(slide, x, y, d, d, fill=fill, shape=MSO_SHAPE.OVAL, margin=0.0,
                   anchor=MSO_ANCHOR.MIDDLE,
                   paras=[{"runs": [(text, {"bold": True})], "align": PP_ALIGN.CENTER}],
                   size=size, color=color, space_after=0)


def new_presentation():
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    return prs


def blank_slide(prs, dark=False):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = NAVY if dark else WHITE
    return slide


def slide_header(slide, kicker, title, dark=False):
    add_text(slide, 0.6, 0.32, 12.1, 0.36,
             [{"runs": [(kicker.upper(), {"bold": True})]}],
             size=14, color=TEAL_SOFT if dark else TEAL, name="Kicker")
    add_text(slide, 0.6, 0.66, 12.1, 1.0, [title], size=28, bold=True,
             color=WHITE if dark else NAVY, name="Title", space_after=0, line_spacing=0.95)


def footer(slide, number, total, source=None, dark=False):
    color = ON_DARK_MUTED if dark else MUTED
    text = SAFETY_LINE if source is None else f"Source: {source}  ·  {SAFETY_LINE}"
    add_text(slide, 0.6, 7.04, 11.2, 0.34, [text], size=11, color=color, name="Footer")
    add_text(slide, 11.9, 7.04, 0.83, 0.34, [{"runs": [f"{number} / {total}"], "align": PP_ALIGN.RIGHT}],
             size=11, color=color, name="Slide number")


def bar_chart(slide, x, y, w, h, categories, values, colors, number_format="0.0", horizontal=True,
              label_size=14, axis_size=14, gap=55, value_max=None):
    data = CategoryChartData()
    data.categories = categories
    data.add_series("value", values)
    kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    gf = slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), data)
    chart = gf.chart
    chart.has_legend = False
    chart.has_title = False
    chart.font.name = FONT
    chart.font.size = Pt(axis_size)
    chart.font.color.rgb = BODY
    plot = chart.plots[0]
    plot.gap_width = gap
    plot.vary_by_categories = False
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = number_format
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(label_size)
    dl.font.bold = True
    dl.font.color.rgb = INK
    for pt_idx, color in enumerate(colors):
        point = plot.series[0].points[pt_idx]
        point.format.fill.solid()
        point.format.fill.fore_color.rgb = color
        point.format.line.fill.background()
    va = chart.value_axis
    va.has_major_gridlines = False
    va.visible = False
    va.minimum_scale = 0
    if value_max:
        va.maximum_scale = value_max
    ca = chart.category_axis
    ca.format.line.color.rgb = LINE
    ca.major_tick_mark = XL_TICK_MARK.NONE
    ca.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    ca.tick_labels.font.size = Pt(axis_size)
    ca.tick_labels.font.color.rgb = BODY
    if horizontal:
        ca.reverse_order = True  # first category on top
    return chart


def add_table(slide, x, y, w, col_widths, rows, row_h=0.42, size=14, header_fill=NAVY, highlight_row=None,
              first_col_bold=True):
    n_rows, n_cols = len(rows), len(rows[0])
    gf = slide.shapes.add_table(n_rows, n_cols, Inches(x), Inches(y), Inches(w), Inches(row_h * n_rows))
    table = gf.table
    table.first_row = True
    table.horz_banding = False
    for j, cw in enumerate(col_widths):
        table.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        table.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            header = i == 0
            if header:
                fill, color, bold = header_fill, WHITE, True
            elif highlight_row is not None and i == highlight_row:
                fill, color, bold = TEAL_TINT, INK, (j == 0 and first_col_bold) or True
            else:
                fill, color, bold = (WHITE if i % 2 else CARD), BODY, (j == 0 and first_col_bold)
            cell.fill.solid()
            cell.fill.fore_color.rgb = fill
            tf.paragraphs[0].text = ""
            fill_text(tf, [{"runs": [str(val)], "align": PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT}],
                      size=size, color=color, bold=bold, space_after=0)
    return table


def enforce_word_wrap(prs):
    """Set word_wrap=True on every text frame in the deck (shapes, tables, notes) and check it."""
    def frames(shapes):
        for sh in shapes:
            if sh.shape_type == 6:  # group
                yield from frames(sh.shapes)
            if getattr(sh, "has_text_frame", False) and sh.has_text_frame:
                yield sh.text_frame
            if getattr(sh, "has_table", False) and sh.has_table:
                for row in sh.table.rows:
                    for cell in row.cells:
                        yield cell.text_frame

    count = 0
    for slide in prs.slides:
        for tf in frames(slide.shapes):
            tf.word_wrap = True
            count += 1
        if slide.has_notes_slide:
            for tf in frames(slide.notes_slide.shapes):
                tf.word_wrap = True
                count += 1
    for slide in prs.slides:
        for tf in frames(slide.shapes):
            assert tf.word_wrap is True
    return count


# ---------------------------------------------------------------------------
# Feature groups (computed from the training script's feature lists)
# ---------------------------------------------------------------------------
FEATURE_GROUPS = [
    ("CGM", "current glucose, change over 15 and 30 min, 1 h mean and SD",
     lambda f: f.startswith("glucose")),
    ("Heart", "heart rate and HRV, now and 1 h means",
     lambda f: f.startswith("hr") or f.startswith("heart")),
    ("Activity", "steps in the last 15 min, 1 h and 2 h", lambda f: f.startswith("steps")),
    ("Meals, insulin", "carbohydrates and insulin units, now and 2 h sums",
     lambda f: f.startswith("meal") or f.startswith("carbs") or f.startswith("insulin")),
    ("Context", "hour of day, sleep stage", lambda f: f in ("hour_of_day", "sleep_stage")),
]


def feature_groups():
    groups, used = [], set()
    for name, desc, rule in FEATURE_GROUPS:
        cols = [f for f in WEARABLE_FEATURES if rule(f) and f not in used]
        used.update(cols)
        groups.append((name, desc, len(cols)))
    leftover = [f for f in WEARABLE_FEATURES if f not in used]
    if leftover:
        groups.append(("Other wearable", ", ".join(leftover), len(leftover)))
    groups.append(("EHR", "age, sex, BMI, diabetes duration, HbA1c, fasting glucose, BP, eGFR, diet, "
                          "medication, insulin regimen, risk tier, prior hypos, family history, TCF7L2",
                   len(EHR_FEATURES)))
    return groups


# ---------------------------------------------------------------------------
# The deck
# ---------------------------------------------------------------------------
def build_deck(m, cohort, team):
    H = m["holdout"]
    reg, pers, lin = H["regressor"], H["baseline_persistence"], H["baseline_linear_extrapolation"]
    hypo, hyper = H["classifier"]["hypo_lt70"], H["classifier"]["hyper_gt250"]
    inr_hypo = H["classifier_not_already_in_event"]["hypo_lt70"]
    inr_hyper = H["classifier_not_already_in_event"]["hyper_gt250"]
    rule_hypo = H["rule_baseline_events"]["hypo_lt70"]
    rule_hyper = H["rule_baseline_events"]["hyper_gt250"]
    ep_hypo, ep_hyper = H["episodes_hypo"], H["episodes_hyper"]
    swap = H["ehr_swap_test"]
    thr = H["alert_thresholds"]
    cv = m["grouped_cv_5fold"]
    abl = m["ablation_grouped_cv_5fold"]
    lat = m["latency"]
    data = m["data"]
    n_pat, n_test = data["patients"], H["test_patients"]
    n_train = n_pat - n_test
    feats = data["features"]
    labels = data["label_counts"]
    hypo_lo, hyper_hi = int(HYPO_MG_DL), int(HYPER_MG_DL)
    folds = cv["folds"]
    clarke = reg["clarke_pct"]
    ab = clarke["A"] + clarke["B"]

    prs = new_presentation()
    TOTAL = 12
    src = "models/metrics.json"

    # ------------------------------------------------------------------ 1 Title
    s = blank_slide(prs, dark=True)
    add_text(s, 0.7, 0.6, 7.0, 0.4,
             [{"runs": [("HAPPIEST HEALTH DIGITAL TWIN CHALLENGE 2026 · PHASE 1", {"bold": True})]}],
             size=14, color=TEAL_SOFT)
    add_text(s, 0.7, 1.05, 7.0, 1.0, ["GlycoTwin"], size=60, bold=True, color=WHITE, space_after=0)
    add_text(s, 0.7, 2.15, 6.6, 1.6,
             [f"A multimodal digital twin for Type 2 diabetes that forecasts hypoglycaemia "
              f"(<{hypo_lo}{NBSP}mg/dL) and marked hyperglycaemia (>{hyper_hi}{NBSP}mg/dL) 2{NBSP}hours ahead, "
              f"by fusing a synthetic EHR with synthetic CGM and wearable data"],
             size=20, color=ON_DARK, line_spacing=1.05)
    add_box(s, 0.7, 4.0, 6.6, 1.75, fill=NAVY_2, radius=0.05, margin=0.2, anchor=MSO_ANCHOR.MIDDLE,
            paras=[{"runs": [(f"Team {team['team']}", {"bold": True, "color": WHITE, "size": 20})],
                    "space_after": 2},
                   {"runs": [team["college"]], "color": ON_DARK, "space_after": 10},
                   {"runs": [(f"{team['leader']} (team leader)", {"bold": True, "color": WHITE})]},
                   {"runs": [" · ".join(team["members"])], "color": ON_DARK}],
            size=16)
    add_box(s, 0.7, 6.3, 12.0, 0.5, fill=None, line=TEAL_MID, radius=0.5, margin=0.12, anchor=MSO_ANCHOR.MIDDLE,
            paras=[{"runs": [SAFETY_LINE], "align": PP_ALIGN.CENTER}], size=14, color=TEAL_SOFT, space_after=0)
    hero, crop = shot("hero")
    if hero:
        # only the current-CGM and alert cards, so the text stays legible when projected
        hero_w = 2.95
        pic = add_picture(s, hero, 7.75 + (5.0 - hero_w) / 2, 0.62, w=hero_w, crop=crop, border=TEAL_MID)
        add_text(s, 7.75, 0.62 + pic.height / 914400 + 0.1, 5.0, 0.62,
                 [{"runs": ["Held-out patient: CGM in range now, hypoglycaemia alert for the next 2 h"],
                   "align": PP_ALIGN.CENTER}],
                 size=14, color=ON_DARK_MUTED, space_after=0)
    s.notes_slide.notes_text_frame.text = (
        "Introduce the team and the one-line idea: a per-patient model that combines the static record with the "
        "live CGM and wearable stream and warns about lows and highs two hours ahead. Say up front that all data is "
        "synthetic and this is a research prototype, not a medical device.")

    # ------------------------------------------------------------------ 2 Problem
    s = blank_slide(prs)
    slide_header(s, "The problem in India",
                 "Diabetes is common in India, and dangerous glucose swings happen between clinic visits")
    stats = [
        ("101 million", "people in India had diabetes in 2021 (11.4% of adults), counted by testing, "
                        "including undiagnosed cases [1]"),
        ("136 million", "had prediabetes (15.3%) [1]"),
        (f"BMI 23.9{NBSP}≈{NBSP}30", "South Asian adults reach the diabetes incidence of white adults "
                                          "with BMI 30 at a BMI of about 23.9 [2]"),
    ]
    y = 1.85
    for big, small in stats:
        add_box(s, 0.6, y, 5.6, 1.35, fill=TEAL_TINT, radius=0.08)
        add_text(s, 0.8, y + 0.08, 5.2, 0.62, [big], size=32, bold=True, color=TEAL, space_after=0)
        add_text(s, 0.8, y + 0.66, 5.2, 0.66, [small], size=14, color=BODY, space_after=0)
        y += 1.5
    add_box(s, 6.6, 1.85, 6.15, 4.35, fill=CARD, radius=0.04, margin=0.25,
            paras=[{"runs": [("What goes wrong between visits", {"bold": True, "color": NAVY, "size": 20})],
                    "space_after": 10},
                   {"runs": [("Hypoglycaemia: ", {"bold": True, "color": RED}),
                             f"Level 1 is <{hypo_lo}{NBSP}mg/dL, Level 2 is <54{NBSP}mg/dL [3]. In Type 2 diabetes "
                             f"the main drug causes are sulfonylureas and insulin; risk rises with age and lower "
                             f"kidney function."], "bullet": True, "bullet_color": RED, "space_after": 10},
                   {"runs": [("Marked hyperglycaemia: ", {"bold": True, "color": AMBER}),
                             f">{hyper_hi}{NBSP}mg/dL. The time-in-range consensus asks for more than 70% of "
                             f"readings in {rng(70, 180)}{NBSP}mg/dL and fewer than 4% below 70 [4]."],
                    "bullet": True, "bullet_color": AMBER, "space_after": 10},
                   {"runs": [("Care is episodic: ", {"bold": True, "color": TEAL}),
                             "clinic reviews happen every few months. A CGM shows the trend, but someone still has "
                             "to notice a trend heading out of range and act before it gets there."],
                    "bullet": True, "space_after": 10}],
            size=16)
    add_text(s, 0.6, 6.33, 12.1, 0.62,
             ["[1] Anjana RM et al., ICMR-INDIAB-17, Lancet Diabetes Endocrinol 2023  ·  [2] Caleyachetty R et al., "
              "Lancet Diabetes Endocrinol 2021",
              "[3] ADA Standards of Care in Diabetes 2025, section 6  ·  [4] Battelino T et al., Diabetes Care 2019"],
             size=12, color=MUTED, space_after=0)
    footer(s, 2, TOTAL)
    s.notes_slide.notes_text_frame.text = (
        "Scale from ICMR-INDIAB-17; the South Asian phenotype at lower BMI; the ADA hypoglycaemia levels and the "
        "time-in-range consensus. The gap we target: events build up over hours between visits.")

    # ------------------------------------------------------------------ 3 Solution
    s = blank_slide(prs)
    slide_header(s, "Our solution",
                 "Every 15 minutes, one fused feature set and two models answer two questions per patient")
    add_box(s, 0.6, 1.85, 3.5, 1.45, fill=CARD, radius=0.08,
            paras=[{"runs": [("Static EHR", {"bold": True, "color": NAVY, "size": 18})], "space_after": 2},
                   "Demographics, labs, medication, history, TCF7L2 genotype"], size=15)
    add_box(s, 0.6, 3.5, 3.5, 1.45, fill=CARD, radius=0.08,
            paras=[{"runs": [("Live wearable stream", {"bold": True, "color": NAVY, "size": 18})], "space_after": 2},
                   "CGM, heart rate, HRV, steps, sleep stage, carbs, insulin"], size=15)
    add_box(s, 4.75, 2.25, 3.0, 2.3, fill=TEAL, radius=0.08, anchor=MSO_ANCHOR.MIDDLE, margin=0.2,
            paras=[{"runs": [("GlycoTwin", {"bold": True, "size": 24})], "align": PP_ALIGN.CENTER, "space_after": 4},
                   {"runs": ["EHR joined to every 15-min window; two gradient-boosted tree models"],
                    "align": PP_ALIGN.CENTER}],
            size=15, color=WHITE)
    add_line(s, 4.1, 2.58, 4.72, 2.95, color=TEAL)
    add_line(s, 4.1, 4.22, 4.72, 3.85, color=TEAL)
    add_box(s, 8.4, 1.85, 4.35, 1.45, fill=RED_TINT, radius=0.08,
            paras=[{"runs": [("1. Event alert", {"bold": True, "color": RED, "size": 18})], "space_after": 2},
                   f"Will any reading in the next 2 h be <{hypo_lo} or >{hyper_hi}{NBSP}mg/dL? "
                   f"Risk score vs a set threshold"], size=15)
    add_box(s, 8.4, 3.5, 4.35, 1.45, fill=TEAL_TINT, radius=0.08,
            paras=[{"runs": [("2. Glucose forecast", {"bold": True, "color": TEAL, "size": 18})], "space_after": 2},
                   "What will the CGM read exactly 2 h from now (mg/dL)?"], size=15)
    add_line(s, 7.78, 2.95, 8.37, 2.58, color=TEAL)
    add_line(s, 7.78, 3.85, 8.37, 4.22, color=TEAL)
    add_text(s, 0.6, 5.2, 12.1, 0.4, [{"runs": [("How a doctor uses the twin", {"bold": True})]}],
             size=18, color=NAVY)
    uses = [
        ("Cohort risk board", "patients ranked by current 2-h risk score"),
        ("Live monitor", "EHR card, last 8 h of data, forecast and alert"),
        ("What-if sandbox", "rule-based counselling tool, separate from the model"),
        ("Model card", f"every number read from {src}"),
    ]
    for i, (h, d) in enumerate(uses):
        add_box(s, 0.6 + i * 3.08, 5.65, 2.9, 1.2, fill=CARD, radius=0.08, margin=0.14,
                paras=[{"runs": [(h, {"bold": True, "color": TEAL})], "space_after": 2}, d], size=15)
    footer(s, 3, TOTAL)
    s.notes_slide.notes_text_frame.text = (
        "Two outputs: an event alert for the next two hours and a point forecast of glucose at +2 h. The doctor "
        "stays in charge: the twin ranks who needs attention and explains why; the clinician decides the action.")

    # ------------------------------------------------------------------ 4 Data
    s = blank_slide(prs)
    slide_header(s, "Data: two synthetic streams",
                 f"Two synthetic streams for the same {n_pat} people, aligned every {data['sampling_minutes']} minutes")
    tiers = cohort["tiers"]
    add_box(s, 0.6, 1.8, 5.95, 3.15, fill=CARD, radius=0.05, margin=0.2,
            paras=[{"runs": [("Static EHR  ", {"bold": True, "color": NAVY, "size": 18}),
                             ("data/synthetic_ehr.csv", {"color": MUTED, "size": 14})], "space_after": 2},
                   {"runs": [(f"{cohort['n']} personas", {"bold": True, "color": TEAL, "size": 22})],
                    "space_after": 6},
                   {"runs": ["Age, sex, diet pattern, BMI; HbA1c, fasting glucose, BP, eGFR, LDL, triglycerides"],
                    "bullet": True},
                   {"runs": ["Medication, insulin regimen, risk tier, past diagnoses"], "bullet": True},
                   {"runs": ["Prior hypos (12 m), family history, TCF7L2 rs7903146 risk alleles"], "bullet": True},
                   {"runs": [f"HbA1c {cohort['hba1c_min']:.1f}–{cohort['hba1c_max']:.1f}% "
                             f"(mean {cohort['hba1c_mean']:.2f}%); {cohort['sulfonylurea']} on a sulfonylurea, "
                             f"{cohort['insulin']} on insulin"],
                    "color": MUTED, "space_before": 4}],
            size=15, space_after=3)
    add_box(s, 6.8, 1.8, 5.95, 3.15, fill=CARD, radius=0.05, margin=0.2,
            paras=[{"runs": [("Live stream  ", {"bold": True, "color": NAVY, "size": 18}),
                             ("data/synthetic_wearables.csv", {"color": MUTED, "size": 14})], "space_after": 2},
                   {"runs": [(f"{intc(data['windows'])} windows", {"bold": True, "color": TEAL, "size": 22}),
                             (f"  {n_pat} × {data['windows'] // n_pat} labelled windows "
                              f"({data['days_per_patient']} days of {data['sampling_minutes']} min data)",
                              {"color": MUTED, "size": 15})],
                    "space_after": 6},
                   {"runs": ["CGM glucose, heart rate, HRV, steps, sleep stage (Awake/Light/Deep/REM)"],
                    "bullet": True},
                   {"runs": ["Logged meal carbohydrates and insulin units"], "bullet": True},
                   {"runs": [f"Labels: any reading <{hypo_lo} or >{hyper_hi} in the next 2 h; glucose exactly "
                             f"2 h ahead"], "bullet": True},
                   {"runs": [f"Windows: {intc(labels['0'])} no event · {intc(labels['1'])} hypo · "
                             f"{intc(labels['2'])} hyper. {intc(data['cgm_readings_lt70'])} readings "
                             f"<{hypo_lo} ({intc(data['cgm_readings_lt54'])} <54) in "
                             f"{data['patients_with_any_lt70']} patients"],
                    "color": MUTED, "space_before": 4}],
            size=15, space_after=3)
    egfr_note = ("no persona is below 60, so not exercised" if cohort["egfr_lt60"] == 0
                 else f"{cohort['egfr_lt60']} personas below 60")
    add_box(s, 0.6, 5.15, 12.15, 1.75, fill=TEAL_TINT, radius=0.05, margin=0.2)
    add_text(s, 0.8, 5.25, 7.3, 1.6,
             [{"runs": [("The simulator couples the streams on purpose", {"bold": True, "color": TEAL})],
               "size": 16, "space_after": 4},
              {"runs": ["Sulfonylurea taken on a skipped breakfast causes lows; missed or extra insulin boluses"],
               "bullet": True},
              {"runs": ["Low glucose raises heart rate and lowers HRV; short sleep raises next-day insulin resistance"],
               "bullet": True},
              {"runs": [f"TCF7L2 risk alleles raise post-meal peaks; eGFR <60 slows insulin clearance ({egfr_note})"],
               "bullet": True}],
             size=14, space_after=2)
    add_text(s, 8.35, 5.25, 4.25, 1.6,
             [{"runs": [("Caveat", {"bold": True, "color": AMBER})], "size": 16, "space_after": 4},
              "Parameters are illustrative and not calibrated to any dataset. Population ranges are loosely "
              "informed by published Indian literature. Meal, walk and sleep times vary day to day."],
             size=14, space_after=2)
    footer(s, 4, TOTAL, source="data/synthetic_*.csv, " + src)
    s.notes_slide.notes_text_frame.text = (
        "Why a simulator: the task needs both streams for the same person, aligned in time, with a label for every "
        "window; Synthea has no CGM stream and MIMIC-IV cannot be redistributed. The cost is realism, so every result "
        "is labelled synthetic.")

    # ------------------------------------------------------------------ 5 Model
    s = blank_slide(prs)
    slide_header(s, "Model",
                 f"Two gradient-boosted tree models on {feats['total']} fused features")
    add_text(s, 0.6, 1.8, 6.0, 0.45,
             [{"runs": [(f"{feats['total']} features = {feats['wearable']} wearable + {feats['ehr']} EHR",
                         {"bold": True})]}], size=18, color=NAVY)
    add_text(s, 0.6, 2.22, 6.0, 0.4, ["All computed from the current and past windows only"], size=14, color=MUTED)
    y = 2.72
    for name, desc, n in feature_groups():
        is_ehr = name == "EHR"
        row_h = 0.98 if is_ehr else 0.62
        circle_badge(s, 0.65, y + 0.05, 0.5, str(n), fill=NAVY if is_ehr else TEAL, size=16)
        add_text(s, 1.3, y, 5.3, row_h,
                 [{"runs": [(name + "  ", {"bold": True, "color": INK}), (desc, {"color": BODY})]}],
                 size=15, space_after=0)
        y += row_h + 0.04
    cards = [
        ("Two scikit-learn HistGradientBoosting models", TEAL,
         [f"Classifier: 3 classes (no event / <{hypo_lo} / >{hyper_hi} in the next 2 h), "
          f"class_weight=\"balanced\" because events are rare",
          "Regressor: CGM glucose exactly 2 h ahead",
          f"Max depth {HGB_PARAMS['max_depth']}, learning rate {HGB_PARAMS['learning_rate']}, "
          f"up to {HGB_PARAMS['max_iter']} iterations, native categorical splits"]),
        ("Split by patient: tested on unseen people", NAVY,
         [f"{n_train} training / {n_test} held-out patients, none in both; held-out = "
          f"{intc(H['windows'])} windows never used for training or thresholds"]),
        ("Alert rule", RED,
         [f"Hypo alert if score ≥ {thr['hypo']:.2f}, else hyper alert if score ≥ {thr['hyper']:.2f}; "
          f"thresholds maximise F2 on {folds}-fold patient-grouped CV of the training patients",
          f"Scores are uncalibrated risk scores, not probabilities. About "
          f"{lat['single_window_ms_median']:.0f}{NBSP}ms per window on CPU"]),
    ]
    y = 1.8
    heights = [1.95, 1.05, 1.85]
    for (title, color, lines), hgt in zip(cards, heights):
        paras = [{"runs": [(title, {"bold": True, "color": color, "size": 17})], "space_after": 4}]
        paras += [{"runs": [ln], "bullet": True, "bullet_color": color} for ln in lines]
        add_box(s, 6.85, y, 5.9, hgt, fill=CARD, radius=0.05, margin=0.18, paras=paras, size=15, space_after=3)
        y += hgt + 0.12
    footer(s, 5, TOTAL, source="models/train_fusion_model.py, " + src)
    s.notes_slide.notes_text_frame.text = (
        "Why gradient-boosted trees: tabular mixed data, rare events, trains in under a minute on a CPU, "
        "deterministic with a fixed seed. The split is by patient, so held-out patients are genuinely unseen. "
        "Thresholds were chosen on training patients only.")

    # ------------------------------------------------------------------ 6 Results: forecast
    s = blank_slide(prs)
    ratio = pers["mae"] / reg["mae"]
    slide_header(s, "Results · 2-hour glucose forecast",
                 f"On {n_test} unseen patients the 2-hour forecast error is {mgdl(reg['mae'])}, "
                 f"{ratio:.1f}× lower than assuming glucose stays the same")
    add_text(s, 0.6, 1.8, 6.2, 0.45,
             [{"runs": [("Mean absolute error at +2 h (mg/dL, lower is better)", {"bold": True})]}],
             size=16, color=NAVY)
    bar_chart(s, 0.5, 2.25, 6.4, 3.3,
              ["GlycoTwin regressor", "Persistence (glucose stays the same)", "Linear extrapolation"],
              [reg["mae"], pers["mae"], lin["mae"]], [TEAL, GREY_BAR, GREY_BAR],
              value_max=max(lin["mae"], pers["mae"]) * 1.2)
    add_text(s, 0.6, 5.65, 6.2, 1.2,
             [f"R² {reg['r2']:.3f} vs {pers['r2']:.3f} for persistence. {folds}-fold patient-grouped CV over "
              f"all {n_pat} patients: MAE {num(cv['regressor']['mae'], 2)}{NBSP}mg/dL, "
              f"R² {cv['regressor']['r2']:.3f}."],
             size=15, color=BODY)
    add_text(s, 7.2, 1.8, 5.55, 0.45,
             [{"runs": [("Clarke error grid zones (% of forecasts)", {"bold": True})]}], size=16, color=NAVY)
    rows = [["Model", "A", "B", "C", "D", "E"]]
    for label, block in (("GlycoTwin", reg), ("Persistence", pers), ("Linear extrap.", lin)):
        z = block["clarke_pct"]
        rows.append([label] + [pctv(z[k]) for k in "ABCDE"])
    add_table(s, 7.2, 2.3, 5.55, [1.75, 0.8, 0.75, 0.75, 0.75, 0.75], rows, row_h=0.5, size=14,
              highlight_row=1)
    add_box(s, 7.2, 4.5, 5.55, 1.45, fill=TEAL_TINT, radius=0.06, margin=0.2, anchor=MSO_ANCHOR.MIDDLE,
            paras=[{"runs": [(f"{pctv(ab)} in zones A+B", {"bold": True, "color": TEAL, "size": 24})],
                    "space_after": 2},
                   f"{pctv(clarke['A'], 2)} A + {pctv(clarke['B'], 2)} B. Zone D {pctv(clarke['D'], 2)}; "
                   + ("none in C or E" if clarke["C"] == 0 and clarke["E"] == 0 else
                      f"zone C {pctv(clarke['C'], 2)}, zone E {pctv(clarke['E'], 2)}")],
            size=15)
    add_text(s, 7.2, 6.05, 5.55, 0.85,
             [f"Each forecast is compared with the CGM reading exactly 2 h later. Zones per Clarke et al., "
              f"Diabetes Care 1987."], size=14, color=MUTED)
    footer(s, 6, TOTAL, source=src)
    s.notes_slide.notes_text_frame.text = (
        "Two honest baselines: persistence (glucose stays the same) and linear extrapolation of the last 15-minute "
        "change. Held-out patients were never used for training or for choosing thresholds. Synthetic data, so this "
        "measures how well the model learns the simulator.")

    # ------------------------------------------------------------------ 7 Results: alerts
    s = blank_slide(prs)
    mostly_false = ep_hypo["false_alarm_fraction"] > 0.5 and ep_hyper["false_alarm_fraction"] > 0.5
    slide_header(s, "Results · 2-hour event alerts",
                 f"Alerts warned of {ep_hypo['warned_15_to_120_min_ahead']} of {ep_hypo['episodes']} lows and "
                 f"{ep_hyper['warned_15_to_120_min_ahead']} of {ep_hyper['episodes']} highs in advance"
                 + (", but most alarms are false" if mostly_false else ""))

    def event_card(x, title, color, tint, w, ep, rule, inr, short):
        add_box(s, x, 1.8, 5.95, 4.1, fill=tint, radius=0.04)
        add_text(s, x + 0.2, 1.9, 5.6, 0.45, [{"runs": [(title, {"bold": True})]}], size=18, color=color)
        stats = [
            (f"{ep['warned_15_to_120_min_ahead']}/{ep['episodes']}", f"episodes warned {rng(15, 120)}{NBSP}min ahead"),
            (pct(w["sensitivity"]), "window sensitivity"),
            (pct(w["precision"]), "window precision"),
        ]
        for i, (big, lab) in enumerate(stats):
            sx = x + 0.2 + i * 1.9
            add_text(s, sx, 2.4, 1.8, 0.65, [big], size=32, bold=True, color=color, space_after=0)
            add_text(s, sx, 3.05, 1.8, 0.7, [lab], size=14, color=BODY, space_after=0)
        add_text(s, x + 0.2, 3.85, 5.6, 2.0,
                 [{"runs": [f"{PR_AUC} {w['pr_auc']:.3f} · {ROC_AUC} {w['roc_auc']:.3f} · "
                            f"median lead time {ep['median_lead_time_min']:.0f} min"], "bullet": True,
                   "bullet_color": color},
                  {"runs": [f"If CGM is {hypo_lo}–{hyper_hi} now: sensitivity {pct(inr['sensitivity'])}, "
                            f"precision {pct(inr['precision'])}"], "bullet": True, "bullet_color": color},
                  {"runs": [f"Simple rule ({short}): sensitivity {pct(rule['sensitivity'])}, "
                            f"precision {pct(rule['precision'])}"], "bullet": True, "bullet_color": color},
                  {"runs": [(f"{ep['alarms_per_patient_day']:.2f} alarms per patient-day; "
                             f"{pct(ep['false_alarm_fraction'], 0)} not followed by the event within 2 h",
                             {"bold": True, "color": INK})], "bullet": True, "bullet_color": color}],
                 size=15, space_after=5)

    event_card(0.6, f"Hypoglycaemia (<{hypo_lo}{NBSP}mg/dL within 2{NBSP}h)", RED, RED_TINT,
               hypo, ep_hypo, rule_hypo, inr_hypo, "CGM <90 and falling")
    event_card(6.8, f"Marked hyperglycaemia (>{hyper_hi}{NBSP}mg/dL within 2{NBSP}h)", AMBER, AMBER_TINT,
               hyper, ep_hyper, rule_hyper, inr_hyper, "CGM >200 and rising")
    add_box(s, 0.6, 6.05, 12.15, 0.85, fill=CARD, radius=0.08, margin=0.2, anchor=MSO_ANCHOR.MIDDLE,
            paras=[{"runs": [("Honest reading: ", {"bold": True, "color": NAVY}),
                             f"thresholds favour sensitivity (F2), so alarm fatigue is the main weakness to fix with "
                             f"clinicians. {n_test} held-out patients, {intc(hypo['positives'])} hypo and "
                             f"{intc(hyper['positives'])} hyper windows; scores are uncalibrated."]}],
            size=15)
    footer(s, 7, TOTAL, source=src)
    s.notes_slide.notes_text_frame.text = (
        "Episode view: of the out-of-range episodes in held-out patients, how many got a warning 15 to 120 minutes "
        "before the first out-of-range reading. Window view: sensitivity and precision per 15-minute window. Be "
        "explicit about false alarms per patient-day and about the harder case where the patient is in range now.")

    # ------------------------------------------------------------------ 8 Fusion evidence
    s = blank_slide(prs)
    fused, wear = abl["fused_all_features"], abl["wearables_only"]
    fus_mae = fused["regressor_mae"] < wear["regressor_mae"]
    fus_hypo = fused["hypo_pr_auc"] > wear["hypo_pr_auc"]
    fus_hyper = fused["hyper_pr_auc"] > wear["hyper_pr_auc"]
    swap_worse = (swap["regressor_mae"] > reg["mae"] and swap["hypo_pr_auc"] < hypo["pr_auc"]
                  and swap["hyper_pr_auc"] < hyper["pr_auc"])
    # "modest" = under 5% lower MAE and under 0.05 PR-AUC gain vs wearables only (one CV run, no per-fold spread)
    modest = ((wear["regressor_mae"] - fused["regressor_mae"]) / wear["regressor_mae"] < 0.05
              and fused["hypo_pr_auc"] - wear["hypo_pr_auc"] < 0.05)
    helps = "helps modestly" if modest else "helps"
    if fus_mae and fus_hypo and not fus_hyper:
        title = f"Fusing the EHR {helps} for the forecast and lows; highs are driven by the wearables"
    elif fus_mae and fus_hypo and fus_hyper:
        title = f"Fusing the EHR {helps} for the forecast and both alerts"
    else:
        title = "Fusing the EHR gives mixed results: see the ablation"
    slide_header(s, "Fusion evidence", title)
    add_text(s, 0.6, 1.8, 6.6, 0.45,
             [{"runs": [(f"Forecast MAE by feature set ({folds}-fold patient-grouped CV, mg/dL)", {"bold": True})]}],
             size=16, color=NAVY)
    ab_rows = [
        ("All: EHR + wearables", "fused_all_features"),
        ("Wearables only", "wearables_only"),
        ("EHR only + hour of day", "ehr_only_plus_hour"),
        ("Without CGM", "without_cgm"),
        ("Without HR, HRV, sleep", "without_hr_hrv_sleep"),
        ("Without history, genetics", "without_history_and_genetics"),
        ("Without hour of day", "without_hour_of_day"),
    ]
    ab_rows = [(f"{lab} ({abl[k]['features']})", abl[k]["regressor_mae"]) for lab, k in ab_rows if k in abl]
    bar_chart(s, 0.45, 2.25, 6.8, 4.2, [r[0] for r in ab_rows], [r[1] for r in ab_rows],
              [TEAL] + [GREY_BAR] * (len(ab_rows) - 1), number_format="0.00", gap=40,
              value_max=max(r[1] for r in ab_rows) * 1.22)
    add_text(s, 0.6, 6.45, 6.6, 0.5, ["Lower is better; the shipped model uses all features."],
             size=14, color=MUTED)
    add_box(s, 7.55, 1.8, 5.2, 2.05, fill=NAVY, radius=0.05, margin=0.18,
            paras=[{"runs": [("EHR-swap test", {"bold": True, "color": WHITE, "size": 18})], "space_after": 2},
                   {"runs": ["Held-out patients scored with another patient's EHR:"], "color": ON_DARK,
                    "space_after": 6},
                   {"runs": [("Forecast MAE  ", {"color": ON_DARK}),
                             (f"{num(reg['mae'])} → {num(swap['regressor_mae'], 2)}{NBSP}mg/dL",
                              {"bold": True, "color": WHITE})]},
                   {"runs": [(f"Hypo {PR_AUC}  ", {"color": ON_DARK}),
                             (f"{hypo['pr_auc']:.3f} → {swap['hypo_pr_auc']:.3f}", {"bold": True, "color": WHITE})]},
                   {"runs": [(f"Hyper {PR_AUC}  ", {"color": ON_DARK}),
                             (f"{hyper['pr_auc']:.3f} → {swap['hyper_pr_auc']:.3f}",
                              {"bold": True, "color": WHITE})]}],
            size=16, space_after=2)
    take = []
    if swap_worse:
        take.append(("The model uses the EHR: ", "the wrong record makes every score worse."))
    take.append(("Small gains over wearables only: " if modest else "Fusion vs wearables only: ",
                 f"MAE {num(wear['regressor_mae'], 2)} → {num(fused['regressor_mae'], 2)}, hypo {PR_AUC} "
                 f"{wear['hypo_pr_auc']:.3f} → {fused['hypo_pr_auc']:.3f}"
                 + (f"; hyper {PR_AUC} is higher without the EHR ({wear['hyper_pr_auc']:.3f} vs "
                    f"{fused['hyper_pr_auc']:.3f})." if not fus_hyper else
                    f", hyper {PR_AUC} {wear['hyper_pr_auc']:.3f} → {fused['hyper_pr_auc']:.3f}.")))
    hrv, hist = abl.get("without_hr_hrv_sleep"), abl.get("without_history_and_genetics")
    no_signal = []
    if hrv and abs(hrv["regressor_mae"] - fused["regressor_mae"]) < 0.1 and hrv["hypo_pr_auc"] >= fused["hypo_pr_auc"] - 0.01:
        no_signal.append(f"HR, HRV and sleep (MAE {num(hrv['regressor_mae'], 2)} without them)")
    if hist and hist["hypo_pr_auc"] >= fused["hypo_pr_auc"]:
        no_signal.append(f"history and genetics (hypo {PR_AUC} {hist['hypo_pr_auc']:.3f} without them)")
    if no_signal:
        take.append(("No measurable signal ", "from " + " or from ".join(no_signal) + "."))
    take.append(("One CV run, ", "no per-fold spread reported: gaps this small may be noise."))
    add_text(s, 7.55, 4.0, 5.2, 2.95,
             [{"runs": [(a, {"bold": True, "color": INK}), b], "bullet": True} for a, b in take],
             size=15, space_after=5)
    footer(s, 8, TOTAL, source=src)
    s.notes_slide.notes_text_frame.text = (
        "Ablation retrains the whole model with feature groups removed, using patient-grouped CV. The EHR-swap test "
        "keeps the model fixed and gives each held-out patient someone else's record. The swap test is the stronger "
        "evidence; the fusion gains over wearables only are small and come from one CV run without per-fold "
        "spread. Be honest that HR, HRV, sleep, history and genetics add no measurable signal in this synthetic data.")

    # ------------------------------------------------------------------ 9 Dashboard: live monitor
    # Screenshots are cropped to the part being discussed so they stay legible when projected.
    s = blank_slide(prs)
    slide_header(s, "Doctor dashboard · Streamlit and hosted page",
                 "The live monitor scores a held-out patient window by window, on a laptop or a phone")
    pic_path, crop = shot("live")
    if pic_path:
        add_picture(s, pic_path, 0.6, 1.8, h=5.05, crop=crop)
    add_text(s, 7.8, 1.8, 4.95, 2.75,
             [{"runs": [("What the doctor sees", {"bold": True, "color": NAVY})], "size": 18, "space_after": 6},
              {"runs": ["Last 8 h of CGM, the +2 h forecast and the alert, with both uncalibrated risk scores "
                        "and their thresholds"], "bullet": True},
              {"runs": ["Held-out patients: the readings that followed (hidden from the model) shown for "
                        "comparison"], "bullet": True},
              {"runs": [f"Predicted low: “consider a snack / review medication timing with the care team”. "
                        f"Rescue-carb text only when CGM is <{hypo_lo}{NBSP}now"], "bullet": True},
              {"runs": ["Scored live on CPU for each selected window"], "bullet": True}],
             size=15, space_after=5)
    mob, crop = shot("phone")
    if mob:
        add_picture(s, mob, 10.55, 4.7, h=2.15, crop=crop)
    add_text(s, 7.8, 4.7, 2.55, 2.15,
             [{"runs": [("Hosted page on a phone", {"bold": True, "color": TEAL})], "space_after": 4},
              "Same patient, an earlier alert. Outputs precomputed in Python; no model runs in the browser."],
             size=15, space_after=0)
    footer(s, 9, TOTAL, source="dashboard/app.py, dashboard/index.html (screenshots)")
    s.notes_slide.notes_text_frame.text = (
        "Demo patient IND-PAT-196 is held out (unseen in training). Use the replay slider to step to the hours before "
        "the first low and show the alert appearing while the CGM is still in range, then reveal what actually "
        "happened. The phone view is the hosted static page for the same patient at an earlier alert.")

    # ------------------------------------------------------------------ 10 Dashboard: cohort and what-if
    s = blank_slide(prs)
    slide_header(s, "Doctor dashboard · cohort and sandbox",
                 "A cohort risk board shows who needs attention; a rule-based sandbox supports counselling")
    rb_w, wi_h = 7.0, 3.75
    rb, crop = shot("board")
    rb_bottom = 1.85
    if rb:
        pic = add_picture(s, rb, 0.6, 1.85, w=rb_w, crop=crop)
        rb_bottom = 1.85 + pic.height / 914400
    wi, crop = shot("whatif")
    wi_bottom, wi_x, wi_w = 1.85, 8.7, 4.05
    if wi:
        pic = add_picture(s, wi, 8.0, 1.85, h=wi_h, crop=crop)
        wi_w = pic.width / 914400
        wi_x = 12.75 - wi_w
        pic.left = Inches(wi_x)
        wi_bottom = 1.85 + wi_h
    add_text(s, 0.6, rb_bottom + 0.15, rb_w, 1.2,
             [{"runs": [("Cohort risk board", {"bold": True, "color": TEAL})], "space_after": 2},
              "Every patient scored in one batched call and ranked by the higher of the two risk scores "
              "(held-out patients shown)"], size=15, space_after=0)
    add_text(s, wi_x, wi_bottom + 0.12, wi_w, 6.95 - wi_bottom - 0.12,
             [{"runs": [("What-if sandbox", {"bold": True, "color": TEAL})], "space_after": 2},
              "Hand-written rules for counselling, not the ML model (its +2 h forecast is shown only for "
              "comparison); not a dosing tool"], size=15, space_after=0)
    footer(s, 10, TOTAL, source="dashboard/app.py (screenshots)")
    s.notes_slide.notes_text_frame.text = (
        "The risk board scores every patient at the replay time in one batched model call and ranks them. The what-if "
        "sandbox is deliberately rule-based and separate from the ML model; it is a counselling aid, not a forecast. "
        f"Both dashboards take every performance number from {src}: Streamlit at runtime, the hosted page at build "
        "time.")

    # ------------------------------------------------------------------ 11 Limitations and safety
    s = blank_slide(prs)
    slide_header(s, "Limitations and safety", "What this prototype does not show yet")
    ins = cohort["insulin"]
    ins_held = cohort["insulin_heldout"]
    ins_txt = f"{ins} of {cohort['n']} personas use insulin"
    if ins_held is not None:
        ins_txt += f" ({ins_held} in the held-out set)"
    egfr_txt = ("none has eGFR <60" if cohort["egfr_lt60"] == 0 else f"{cohort['egfr_lt60']} have eGFR <60")
    lims = [
        ("Synthetic data only. ", "The model learns our simulator, which is simpler than real physiology; high "
                                  f"{ROC_AUC} partly reflects that. No result transfers to real patients yet."),
        ("Uncalibrated simulator and scores. ", "Simulator parameters are hand-set; classifier outputs are risk "
                                                "scores, meaningful only against the thresholds."),
        ("False alarms. ", f"{ep_hypo['alarms_per_patient_day']:.2f} hypo and {ep_hyper['alarms_per_patient_day']:.2f}"
                           f" hyper alarms per patient-day; {pct(ep_hypo['false_alarm_fraction'], 0)} and "
                           f"{pct(ep_hyper['false_alarm_fraction'], 0)} are not followed by the event."),
        ("Harder case: in range now. ", f"If CGM is {rng(hypo_lo, hyper_hi)} at the time of the alert, "
                                        f"sensitivity is {pct(inr_hypo['sensitivity'])} (hypo) and "
                                        f"{pct(inr_hyper['sensitivity'])} (hyper)."),
        ("Thin coverage. ", f"{ins_txt}; {egfr_txt}; only {intc(data['cgm_readings_lt54'])} readings <54, too few "
                            f"to evaluate Level 2 hypoglycaemia."),
        ("Regular routines. ", f"Hour of day is a strong feature (MAE "
                               f"{num(abl['without_hour_of_day']['regressor_mae'], 2)} without it); real days vary more."),
    ]
    add_text(s, 0.6, 1.8, 6.6, 5.1,
             [{"runs": [(a, {"bold": True, "color": INK}), b], "bullet": True} for a, b in lims],
             size=15, space_after=7)
    add_box(s, 7.55, 1.8, 5.2, 5.1, fill=NAVY, radius=0.04, margin=0.25,
            paras=[{"runs": [("Safety statement", {"bold": True, "color": WHITE, "size": 20})], "space_after": 6},
                   {"runs": [("Synthetic data only. Research prototype, not a medical device. Not for diagnosis or "
                              "dosing decisions.", {"bold": True, "color": TEAL_SOFT})], "space_after": 10},
                   {"runs": ["Decision support for a clinician, who decides any action"], "bullet": True,
                    "bullet_color": TEAL_SOFT},
                   {"runs": ["A predicted low suggests a snack or a medication-timing review with the care team"],
                    "bullet": True, "bullet_color": TEAL_SOFT},
                   {"runs": [f"Rescue text (15 g fast carbs, recheck in 15 min) only when CGM is <{hypo_lo} now; "
                             f"never insulin doses"], "bullet": True, "bullet_color": TEAL_SOFT},
                   {"runs": ["Uses only synthetic data, so no personal health data is processed (designed with "
                             "DPDP Act 2023 principles in mind)"], "bullet": True, "bullet_color": TEAL_SOFT},
                   {"runs": ["Clinical use would need prospective validation and regulatory review"],
                    "bullet": True, "bullet_color": TEAL_SOFT}],
            size=15, color=ON_DARK, space_after=8)
    footer(s, 11, TOTAL, source="data/synthetic_ehr.csv, " + src)
    s.notes_slide.notes_text_frame.text = (
        "Read the safety statement aloud. Regulatory note: clinical use would fall under software as a medical "
        "device in India's Medical Devices Rules, 2017.")

    # ------------------------------------------------------------------ 12 Roadmap and close
    s = blank_slide(prs, dark=True)
    slide_header(s, "Next steps", "From a synthetic twin to a clinic pilot", dark=True)
    steps = [
        ("Real CGM data", "Validate on open Type 2 diabetes CGM datasets (for example ShanghaiT2DM, CGMacros on "
                          "PhysioNet) after licence checks"),
        ("Realistic EHR", "Synthea with an India-adapted module; MIMIC-IV only where its data use agreement allows"),
        ("Calibration and alarms", "Calibrate risk scores on grouped CV; tune thresholds with clinicians to cut "
                                   "false alarms"),
        ("Prospective pilot", "Consented pilot with an Indian diabetes clinic; measure outreach and out-of-range "
                              "time against usual care"),
    ]
    y = 1.85
    for i, (h, d) in enumerate(steps, 1):
        circle_badge(s, 0.65, y + 0.08, 0.55, str(i), fill=TEAL, size=18)
        add_text(s, 1.4, y, 5.9, 1.1,
                 [{"runs": [(h, {"bold": True, "color": WHITE})], "size": 18, "space_after": 2},
                  {"runs": [d], "color": ON_DARK}], size=15)
        if i < len(steps):
            add_line(s, 0.925, y + 0.66, 0.925, y + 1.2, color=TEAL_MID, width=1.5, arrow=False)
        y += 1.22
    repo_short = team["repo"].replace("https://", "")
    add_box(s, 7.8, 1.85, 4.95, 4.85, fill=NAVY_2, radius=0.04, margin=0.25,
            paras=[{"runs": [("Thank you", {"bold": True, "color": WHITE, "size": 26})], "space_after": 10},
                   {"runs": [("Code and docs", {"bold": True, "color": TEAL_SOFT})], "space_after": 0},
                   {"runs": [repo_short.rsplit("/", 1)[0] + "/"], "color": ON_DARK, "space_after": 0},
                   {"runs": [repo_short.rsplit("/", 1)[1]], "color": ON_DARK, "space_after": 10},
                   {"runs": [("Hosted page", {"bold": True, "color": TEAL_SOFT})], "space_after": 0},
                   {"runs": [LIVE_SITE], "color": ON_DARK, "space_after": 10},
                   {"runs": [("Video walkthrough (20+ min)", {"bold": True, "color": TEAL_SOFT})], "space_after": 0},
                   {"runs": [team["video"].replace("https://", "") if team["video"]
                             else "linked in README.md, section 5"], "color": ON_DARK, "space_after": 10},
                   {"runs": [f"Team {team['team']} · {team['college']}"], "color": ON_DARK_MUTED, "size": 14,
                    "space_after": 2},
                   {"runs": [" · ".join([team["leader"]] + team["members"])], "color": ON_DARK_MUTED,
                    "size": 14}],
            size=16)
    footer(s, 12, TOTAL, dark=True)
    s.notes_slide.notes_text_frame.text = (
        "Close with the roadmap and the links. Once the unlisted video link is pasted into README.md section 5, "
        "rerun generate_summit_docs.py and the link replaces the pointer to the README on this slide.")

    n = enforce_word_wrap(prs)
    assert len(prs.slides) == TOTAL, len(prs.slides)
    DECK_PPTX.parent.mkdir(parents=True, exist_ok=True)
    prs.save(DECK_PPTX)
    print(f"  wrote {DECK_PPTX.relative_to(ROOT)} ({len(prs.slides)} slides, word_wrap on {n} text frames)")


# ---------------------------------------------------------------------------
# Architecture diagram
# ---------------------------------------------------------------------------
def arch_node(slide, x, y, w, h, title, lines, kind="code", title_size=14, size=14, bullets=False):
    styles = {
        "code": dict(fill=WHITE, line=TEAL, tcolor=TEAL, color=BODY),
        "file": dict(fill=TEAL_TINT, line=None, tcolor=NAVY, color=BODY),
        "serve": dict(fill=WHITE, line=NAVY, tcolor=NAVY, color=BODY),
        "person": dict(fill=NAVY, line=None, tcolor=WHITE, color=ON_DARK),
        "roadmap": dict(fill=WHITE, line=MUTED, tcolor=MUTED, color=MUTED, dash=MSO_LINE_DASH_STYLE.DASH),
    }[kind]
    paras = [{"runs": [(title, {"bold": True, "color": styles["tcolor"], "size": title_size})], "space_after": 3}]
    paras += [{"runs": [ln], "bullet": bullets, "indent": 0.16, "bullet_color": styles["tcolor"]} for ln in lines]
    return add_box(slide, x, y, w, h, fill=styles["fill"], line=styles["line"], line_width=1.5, radius=0.07,
                   margin=0.1, anchor=MSO_ANCHOR.MIDDLE, paras=paras, size=size, color=styles["color"],
                   space_after=2 if bullets else 1, line_spacing=0.95, dash=styles.get("dash"))


def build_architecture(m, cohort):
    H = m["holdout"]
    data = m["data"]
    feats = data["features"]
    n_test = H["test_patients"]
    n_train = data["patients"] - n_test
    thr = H["alert_thresholds"]
    lat = m["latency"]

    prs = new_presentation()
    s = blank_slide(prs)
    add_text(s, 0.45, 0.22, 12.4, 0.6, ["GlycoTwin architecture: simulate, train, evaluate, serve"],
             size=26, bold=True, color=NAVY, space_after=0)
    add_text(s, 0.45, 0.8, 12.4, 0.36,
             ["Boxes are files in the repository; arrows show data flow. White = script, tinted = data or model "
              "file, navy = clinician. Slide 2: care pathway."],
             size=14, color=MUTED, space_after=0)

    # lanes
    lane_fill = RGBColor(0xF8, 0xFA, 0xFC)
    add_box(s, 0.3, 1.2, 12.73, 2.8, fill=lane_fill, line=LINE, radius=0.03)
    add_text(s, 0.45, 1.22, 7.0, 0.3, [{"runs": [("1  OFFLINE: SIMULATE, TRAIN, EVALUATE", {"bold": True})]}],
             size=12, color=TEAL)
    add_box(s, 0.3, 4.3, 12.73, 2.8, fill=lane_fill, line=LINE, radius=0.03)
    add_text(s, 0.45, 4.32, 7.0, 0.3,
             [{"runs": [("2  SERVING: TWO INFERENCE PATHS, DOCTOR IN THE LOOP", {"bold": True})]}],
             size=12, color=TEAL)

    top, bot = 1.55, 3.92  # lane 1 box band
    mid = (top + bot) / 2
    gap = 0.36
    gx, gw = 0.45, 1.75
    cx, cw = gx + gw + gap, 2.35
    tx, tw = cx + cw + gap, 2.95
    jx, jw = tx + tw + gap, 1.95
    ex = jx + jw + gap
    ew = 12.88 - ex
    arch_node(s, gx, mid - 0.85, gw, 1.7, f"generate_{ZW}datasets.py",
              ["data/ · seeded simulator, illustrative parameters"])
    ehr_y0, ehr_y1 = top, top + 1.0
    wear_y0, wear_y1 = bot - 1.0, bot
    arch_node(s, cx, ehr_y0, cw, ehr_y1 - ehr_y0, f"synthetic_{ZW}ehr.csv",
              [f"{cohort['n']} personas: labs, meds, history, TCF7L2"], kind="file")
    arch_node(s, cx, wear_y0, cw, wear_y1 - wear_y0, f"synthetic_{ZW}wearables.csv",
              [f"{intc(data['windows'])} windows of {data['sampling_minutes']}{NBSP}min"], kind="file")
    arch_node(s, tx, top, tw, bot - top, f"train_{ZW}fusion_{ZW}model.py",
              [f"{feats['total']} features: {feats['wearable']} wearable + {feats['ehr']} EHR, "
               f"EHR joined to every window",
               f"patient split {n_train} / {n_test}",
               "HGB classifier + regressor",
               f"alert thresholds {thr['hypo']:.2f} / {thr['hyper']:.2f} by F2 on grouped CV of training patients"],
              bullets=True)
    arch_node(s, jx, top, jw, bot - top, f"glycotwin_{ZW}model.joblib",
              ["classifier, regressor, thresholds, feature list, category levels, split"], kind="file")
    ev_y1 = top + 1.05
    me_y0 = bot - 1.0
    arch_node(s, ex, top, ew, ev_y1 - top, "evaluate.py",
              ["held-out, CV, baselines, ablation, latency"])
    arch_node(s, ex, me_y0, ew, bot - me_y0, "metrics.json",
              ["the only source of performance numbers"], kind="file")

    A = dict(color=MUTED, width=1.75)
    # generate -> csvs (fan-out)
    split_x = gx + gw + gap / 2
    ehr_mid, wear_mid = (ehr_y0 + ehr_y1) / 2, (wear_y0 + wear_y1) / 2
    add_line(s, gx + gw, mid, split_x, mid, arrow=False, **A)
    add_line(s, split_x, ehr_mid, split_x, wear_mid, arrow=False, **A)
    add_line(s, split_x, ehr_mid, cx, ehr_mid, **A)
    add_line(s, split_x, wear_mid, cx, wear_mid, **A)
    # csvs -> train
    add_line(s, cx + cw, ehr_mid, tx, ehr_mid, **A)
    add_line(s, cx + cw, wear_mid, tx, wear_mid, **A)
    # train -> joblib
    add_line(s, tx + tw, mid, jx, mid, **A)
    # joblib -> evaluate -> metrics
    ev_mid = (top + ev_y1) / 2
    add_line(s, jx + jw, ev_mid, ex, ev_mid, **A)
    add_line(s, ex + ew / 2, ev_y1, ex + ew / 2, me_y0, **A)

    # lane 2 nodes
    r1_0, r1_1 = 4.67, 5.77
    r2_0, r2_1 = 6.0, 6.98
    dx, dw = 0.45, 3.0
    ax, aw = dx + dw + 0.4, 3.85
    bx = ax + aw + 0.4
    bw = 12.88 - bx
    arch_node(s, dx, r1_0, dw, r2_1 - r1_0, "Doctor in the loop",
              ["Diabetologist or diabetes educator reviews the risk board, alerts and model card",
               "Decides the outreach: a call, a medication review",
               "Decision support only; no automated dosing"], kind="person", title_size=16, bullets=True)
    arch_node(s, ax, r1_0, aw, r1_1 - r1_0, "(a) dashboard/app.py · Streamlit",
              [f"live inference on the selected window (about {lat['single_window_ms_median']:.0f}{NBSP}ms on "
               f"CPU); Data Ingest runs 5 checks on uploads"], kind="serve")
    arch_node(s, bx, r1_0, bw, r1_1 - r1_0, f"(b) build_{ZW}standalone_{ZW}dashboard.py",
              ["scores the demo patients in Python at build time and embeds the outputs"], kind="serve")
    arch_node(s, bx, r2_0, bw, r2_1 - r2_0, "dashboard/index.html + plotly.min.js",
              ["static replay page; no model runs in the browser; uploads get the 5 ingest checks"], kind="file")
    arch_node(s, ax, r2_0, aw, r2_1 - r2_0, "public/ on Firebase Hosting",
              [f"{LIVE_SITE}, opens on a laptop or a phone"], kind="file")

    # bus: bundle + metrics feed both dashboards
    bus_y = 4.15
    a_in = ax + aw / 2
    b_in = bx + bw / 2
    j_out = jx + jw / 2
    m_out = ex + ew / 2
    add_line(s, j_out, bot, j_out, bus_y, arrow=False, **A)
    add_line(s, m_out, bot, m_out, bus_y, arrow=False, **A)
    add_line(s, a_in, bus_y, m_out, bus_y, arrow=False, **A)
    add_line(s, a_in, bus_y, a_in, r1_0, **A)
    add_line(s, b_in, bus_y, b_in, r1_0, **A)
    add_text(s, a_in - 3.1, bus_y - 0.17, 3.0, 0.34,
             [{"runs": ["bundle + metrics.json →"], "align": PP_ALIGN.RIGHT}], size=12, color=MUTED,
             space_after=0)
    # (b) -> index.html -> Firebase -> doctor ; (a) -> doctor
    add_line(s, b_in, r1_1, b_in, r2_0, **A)
    r2_mid = (r2_0 + r2_1) / 2
    r1_mid = (r1_0 + r1_1) / 2
    add_line(s, bx, r2_mid, ax + aw, r2_mid, **A)
    add_line(s, ax, r2_mid, dx + dw, r2_mid, **A)
    add_line(s, ax, r1_mid, dx + dw, r1_mid, **A)

    add_text(s, 0.45, 7.12, 12.4, 0.32, [SAFETY_LINE], size=11, color=MUTED, space_after=0)
    build_care_pathway(prs, m)
    n = enforce_word_wrap(prs)
    prs.save(ARCH_PPTX)
    print(f"  wrote {ARCH_PPTX.relative_to(ROOT)} (word_wrap on {n} text frames)")


def build_care_pathway(prs, m):
    """Slide 2 of the architecture file: the same system as a clinical loop, with roadmap items dashed."""
    data = m["data"]
    feats = data["features"]
    step = data["sampling_minutes"]
    hypo_lo, hyper_hi = int(HYPO_MG_DL), int(HYPER_MG_DL)

    s = blank_slide(prs)
    add_text(s, 0.45, 0.22, 12.4, 0.6, ["GlycoTwin in the care pathway: from device data to clinician action"],
             size=26, bold=True, color=NAVY, space_after=0)
    add_text(s, 0.45, 0.8, 12.4, 0.36,
             ["Conceptual view of slide 1. Solid boxes run in this prototype today, on synthetic data; dashed "
              "boxes are roadmap, not built; navy boxes are people."],
             size=14, color=MUTED, space_after=0)

    w, gap = 2.77, 0.45
    col = [0.45 + i * (w + gap) for i in range(4)]
    cmid = [c + w / 2 for c in col]
    t0, t1 = 1.3, 2.3      # top band: data sources and roadmap feeds
    m0, m1 = 2.75, 4.55    # middle row: ingest, fusion, models
    b0, b1 = 5.0, 6.8      # bottom row: board, clinician, outreach, patient
    tmid, mmid, bmid = (t0 + t1) / 2, (m0 + m1) / 2, (b0 + b1) / 2

    arch_node(s, col[0], t0, w, t1 - t0, "Roadmap: vendor exports, APIs",
              ["LibreView, Dexcom, Health Connect, HealthKit"], kind="roadmap")
    arch_node(s, col[1], t0, w, t1 - t0, "Roadmap: HL7 FHIR EHR feed",
              ["Patient, Observation, MedicationStatement, Condition"], kind="roadmap")
    arch_node(s, col[2], t0, w, t1 - t0, "EHR record",
              [f"labs, medication, history, TCF7L2; today synthetic_{ZW}ehr.csv"], kind="file")

    arch_node(s, col[0], m0, w, m1 - m0, "CGM and wearable data",
              ["glucose, heart rate, HRV, steps, sleep stage, carbs, insulin",
               f"today synthetic_{ZW}wearables.csv or an uploaded CSV / JSON"], kind="file", bullets=True)
    arch_node(s, col[1], m0, w, m1 - m0, "Data Ingest: 5 checks",
              ["schema · continuity · physiological bounds · rate of change · missingness",
               "in both dashboards; Streamlit also scores a full-schema file"], kind="serve", bullets=True)
    arch_node(s, col[2], m0, w, m1 - m0, "Feature fusion",
              [f"EHR joined to every {step}-min window",
               f"{feats['total']} features = {feats['wearable']} wearable + {feats['ehr']} EHR",
               "current and past data only"], kind="code", bullets=True)
    arch_node(s, col[3], m0, w, m1 - m0, "Twin models",
              [f"classifier: any reading <{hypo_lo} or >{hyper_hi}{NBSP}mg/dL in the next 2 h?",
               "regressor: glucose at +2 h",
               "uncalibrated risk scores vs fixed thresholds"], kind="code", bullets=True)

    arch_node(s, col[3], b0, w, b1 - b0, "Risk board and alerts",
              ["patients ranked by risk score",
               "Live Monitor: last 8 h of CGM, the +2 h forecast and the alert"], kind="serve", bullets=True)
    arch_node(s, col[2], b0, w, b1 - b0, "Clinician review",
              ["diabetologist or diabetes educator decides whether and how to act",
               "decision support only"], kind="person", bullets=True)
    arch_node(s, col[1], b0, w, b1 - b0, "Outreach by the care team",
              ["a call or a medication-timing review; predicted low: consider a snack",
               "no insulin dose advice"], kind="person", bullets=True)
    arch_node(s, col[0], b0, w, b1 - b0, "Patient",
              [f"keeps wearing the CGM and wearable; new readings every {step} min"], kind="person")

    A = dict(color=MUTED, width=1.75)
    D = dict(color=MUTED, width=1.5, dash=MSO_LINE_DASH_STYLE.DASH)
    add_line(s, cmid[0], t1, cmid[0], m0, **D)                 # vendor APIs -> device data
    add_line(s, col[1] + w, tmid, col[2], tmid, **D)           # FHIR -> EHR record
    add_line(s, cmid[2], t1, cmid[2], m0, **A)                 # EHR -> fusion
    for i in range(3):                                         # device data -> ingest -> fusion -> models
        add_line(s, col[i] + w, mmid, col[i + 1], mmid, **A)
    add_line(s, cmid[3], m1, cmid[3], b0, **A)                 # models -> risk board
    for i in (3, 2, 1):                                        # board -> clinician -> outreach -> patient
        add_line(s, col[i], bmid, col[i - 1] + w, bmid, **A)
    add_line(s, cmid[0], b0, cmid[0], m1, color=TEAL, width=2.0)  # patient -> new device data (the loop)

    # legend, top right
    lx, ly = col[3] + 0.1, t0 + 0.02
    for i, (label, kind) in enumerate((("runs in this prototype", "serve"), ("roadmap, not built", "roadmap"),
                                       ("people", "person"))):
        st = {"serve": (WHITE, NAVY, None), "roadmap": (WHITE, MUTED, MSO_LINE_DASH_STYLE.DASH),
              "person": (NAVY, None, None)}[kind]
        add_box(s, lx, ly + i * 0.33, 0.42, 0.24, fill=st[0], line=st[1], dash=st[2], line_width=1.5, radius=0.2)
        add_text(s, lx + 0.55, ly + i * 0.33 - 0.04, w - 0.65, 0.32, [label], size=14, color=BODY, space_after=0)

    add_text(s, 0.45, 7.12, 12.4, 0.32, [SAFETY_LINE], size=11, color=MUTED, space_after=0)


# ---------------------------------------------------------------------------
# PDF export (LibreOffice) and PNG of the diagram
# ---------------------------------------------------------------------------
def find_soffice():
    for name in ("soffice", "libreoffice"):
        path = shutil.which(name)
        if path:
            return path
    mac = Path("/Applications/LibreOffice.app/Contents/MacOS/soffice")
    return str(mac) if mac.exists() else None


def export_pdfs(paths):
    soffice = find_soffice()
    if not soffice:
        print("\nLibreOffice (soffice) not found, so the PDFs were not regenerated. Export them manually:")
        for p in paths:
            print(f"  soffice --headless --convert-to pdf --outdir docs {p.relative_to(ROOT)}")
        print("  (or open each .pptx in PowerPoint and use File > Save As > PDF)")
        return []
    env = os.environ.copy()
    if platform.system() == "Linux":
        env.setdefault("SAL_USE_VCLPLUGIN", "svp")
    out = []
    with tempfile.TemporaryDirectory(prefix="lo_profile_") as profile:
        for p in paths:
            cmd = [soffice, f"-env:UserInstallation={Path(profile).as_uri()}", "--headless",
                   "--convert-to", "pdf", "--outdir", str(p.parent), str(p)]
            res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
            pdf = p.with_suffix(".pdf")
            if res.returncode != 0 or not pdf.exists():
                print(f"  PDF export failed for {p.name}: {res.stderr.strip()[:400]}")
                continue
            print(f"  wrote {pdf.relative_to(ROOT)} (LibreOffice export of the .pptx)")
            out.append(pdf)
    return out


def export_arch_png(pdf):
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm or not pdf.exists():
        print("  pdftoppm not found; skipped docs/img/architecture.png")
        return
    ARCH_PNG.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([pdftoppm, "-png", "-r", "150", "-singlefile", str(pdf), str(ARCH_PNG.with_suffix(""))],
                   check=True, timeout=120)
    print(f"  wrote {ARCH_PNG.relative_to(ROOT)}")


def main():
    print("Generating GlycoTwin deck and architecture diagram")
    metrics, cohort, team = load_inputs()
    if not team["video"]:
        print("  WARNING: no YouTube or Google Drive link found in README.md section 5; slide 12 points to the "
              "README instead. Paste the unlisted video URL there and rerun before submitting.")
    build_deck(metrics, cohort, team)
    build_architecture(metrics, cohort)
    pdfs = export_pdfs([DECK_PPTX, ARCH_PPTX])
    arch_pdf = ARCH_PPTX.with_suffix(".pdf")
    if arch_pdf in pdfs:
        export_arch_png(arch_pdf)
    print("Done.")


if __name__ == "__main__":
    main()
