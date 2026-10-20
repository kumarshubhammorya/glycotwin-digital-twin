#!/usr/bin/env python3
"""
Converts docs/VIDEO_SCRIPT_20MIN.md into a high-definition, professionally styled
speaker PDF (docs/GlycoTwin_Video_Script_20Min.pdf) using headless Chrome.
"""

import os
import re
import subprocess

def parse_markdown_to_html(md_text):
    lines = md_text.splitlines()
    html_lines = []
    
    in_table = False
    table_rows = []
    
    in_blockquote = False
    blockquote_lines = []
    
    in_ul = False
    in_ol = False
    
    def format_inline(text):
        # Escape basic HTML
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        # Badges for stage directions
        text = re.sub(r"&lt;b&gt;\[SCREEN\]&lt;/b&gt;|\[SCREEN\]|\*\*\[SCREEN\]\*\*", 
                      r'<span class="badge badge-screen">🖥️ SCREEN</span>', text)
        text = re.sub(r"&lt;b&gt;\[DO\]&lt;/b&gt;|\[DO\]|\*\*\[DO\]\*\*", 
                      r'<span class="badge badge-do">⚡ ACTION</span>', text)
        text = re.sub(r"\*Expect\*|&lt;i&gt;Expect&lt;/i&gt;", 
                      r'<span class="badge badge-expect">👁️ EXPECT</span>', text)
        
        # Bold and Italic
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
        text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
        
        # Checkbox
        text = re.sub(r"\[ \]", r'<span class="checkbox">☐</span>', text)
        
        # Links
        text = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', text)
        
        return text

    def flush_table():
        nonlocal in_table, table_rows
        if not in_table or not table_rows:
            return
        out = ['<div class="table-container"><table>']
        header = True
        for row in table_rows:
            if re.match(r"^[\s\|:\-]+$", row):
                continue
            cells = [c.strip() for c in row.strip("|").split("|")]
            out.append("<tr>")
            tag = "th" if header else "td"
            for cell in cells:
                out.append(f"<{tag}>{format_inline(cell)}</{tag}>")
            out.append("</tr>")
            header = False
        out.append("</table></div>")
        html_lines.append("\n".join(out))
        in_table = False
        table_rows = []

    def flush_blockquote():
        nonlocal in_blockquote, blockquote_lines
        if not in_blockquote or not blockquote_lines:
            return
        content = " ".join([l.strip() for l in blockquote_lines])
        html_lines.append(f'<div class="speech-bubble"><div class="speech-icon">🎙️ <b>SPOKEN WORDS:</b></div><div class="speech-text">{format_inline(content)}</div></div>')
        in_blockquote = False
        blockquote_lines = []

    def flush_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            html_lines.append("</ul>")
            in_ul = False
        if in_ol:
            html_lines.append("</ol>")
            in_ol = False

    for line in lines:
        stripped = line.strip()

        # Handle Tables
        if stripped.startswith("|") and stripped.endswith("|"):
            flush_blockquote()
            flush_lists()
            in_table = True
            table_rows.append(stripped)
            continue
        elif in_table:
            flush_table()

        # Handle Blockquotes (Spoken lines)
        if stripped.startswith(">"):
            flush_lists()
            in_blockquote = True
            quote_text = stripped[1:].strip()
            if quote_text:
                blockquote_lines.append(quote_text)
            continue
        elif in_blockquote:
            flush_blockquote()

        # Horizontal Rule
        if re.match(r"^---+$", stripped):
            flush_lists()
            html_lines.append('<hr class="divider"/>')
            continue

        # Headers
        if stripped.startswith("#### "):
            flush_lists()
            title = stripped[5:]
            html_lines.append(f'<h4 class="sub-cue-title">{format_inline(title)}</h4>')
            continue
        elif stripped.startswith("### "):
            flush_lists()
            title = stripped[4:]
            html_lines.append(f'<h3 class="segment-title">{format_inline(title)}</h3>')
            continue
        elif stripped.startswith("## "):
            flush_lists()
            title = stripped[3:]
            html_lines.append(f'<h2 class="section-title">{format_inline(title)}</h2>')
            continue
        elif stripped.startswith("# "):
            flush_lists()
            title = stripped[2:]
            html_lines.append(f'<h1 class="main-title">{format_inline(title)}</h1>')
            continue

        # Lists
        if stripped.startswith("- ") or stripped.startswith("* "):
            if not in_ul:
                flush_lists()
                html_lines.append("<ul>")
                in_ul = True
            html_lines.append(f"<li>{format_inline(stripped[2:])}</li>")
            continue
        
        ol_match = re.match(r"^(\d+)\.\s+(.+)$", stripped)
        if ol_match:
            if not in_ol:
                flush_lists()
                html_lines.append("<ol>")
                in_ol = True
            html_lines.append(f"<li>{format_inline(ol_match.group(2))}</li>")
            continue

        # Regular empty line or paragraph
        if not stripped:
            flush_lists()
            continue

        flush_lists()
        # Stage directions or notes
        if "[SCREEN]" in stripped or "[DO]" in stripped or "Expect" in stripped:
            html_lines.append(f'<div class="stage-direction">{format_inline(stripped)}</div>')
        else:
            html_lines.append(f'<p>{format_inline(stripped)}</p>')

    flush_table()
    flush_blockquote()
    flush_lists()

    return "\n".join(html_lines)


def build_html_doc(body_content):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GlycoTwin: 20-Minute Video Script & Demonstration Guide</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

  @page {{
    size: A4;
    margin: 18mm 16mm 20mm 16mm;
    @top-right {{
      content: "GlycoTwin · 20-Min Video Script";
      font-family: 'Inter', sans-serif;
      font-size: 8pt;
      color: #64748B;
    }}
    @bottom-center {{
      content: "Page " counter(page) " of " counter(pages);
      font-family: 'Inter', sans-serif;
      font-size: 8pt;
      color: #64748B;
    }}
  }}

  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}

  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #0F172A;
    background: #FFFFFF;
    font-size: 10pt;
    line-height: 1.55;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }}

  /* Header Hero */
  .hero-box {{
    background: #0F172A;
    color: #FFFFFF;
    padding: 20px 24px;
    border-radius: 8px;
    margin-bottom: 22px;
    border-left: 6px solid #0D9488;
  }}

  .hero-tag {{
    font-size: 8.5pt;
    font-weight: 700;
    color: #2DD4BF;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
  }}

  .main-title {{
    font-size: 18pt;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.25;
    margin-bottom: 6px;
  }}

  .hero-sub {{
    font-size: 9.5pt;
    color: #94A3B8;
    margin-bottom: 12px;
  }}

  .hero-meta {{
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    font-size: 8.5pt;
    color: #CBD5E1;
    border-top: 1px solid #334155;
    padding-top: 10px;
  }}

  .hero-meta b {{
    color: #2DD4BF;
  }}

  /* Section Titles */
  .section-title {{
    font-size: 14pt;
    font-weight: 800;
    color: #0F172A;
    margin-top: 24px;
    margin-bottom: 12px;
    border-bottom: 2px solid #E2E8F0;
    padding-bottom: 6px;
    page-break-after: avoid;
  }}

  .segment-title {{
    font-size: 12pt;
    font-weight: 700;
    color: #0D9488;
    background: #F0FDFA;
    border-left: 4px solid #0D9488;
    padding: 6px 12px;
    border-radius: 0 6px 6px 0;
    margin-top: 18px;
    margin-bottom: 10px;
    page-break-after: avoid;
  }}

  .sub-cue-title {{
    font-size: 10.5pt;
    font-weight: 700;
    color: #1E293B;
    margin-top: 14px;
    margin-bottom: 8px;
    display: inline-block;
    background: #F1F5F9;
    padding: 3px 8px;
    border-radius: 4px;
    border: 1px solid #E2E8F0;
    page-break-after: avoid;
  }}

  p {{
    margin-bottom: 8px;
    color: #334155;
  }}

  ul, ol {{
    margin-left: 20px;
    margin-bottom: 12px;
    color: #334155;
  }}

  li {{
    margin-bottom: 4px;
  }}

  .checkbox {{
    font-size: 11pt;
    color: #0D9488;
    margin-right: 4px;
    font-weight: bold;
  }}

  /* Spoken Script Bubble */
  .speech-bubble {{
    background: #F8FAFC;
    border: 1.5px solid #CBD5E1;
    border-left: 5px solid #0D9488;
    border-radius: 6px;
    padding: 12px 14px;
    margin: 10px 0 14px 0;
    page-break-inside: avoid;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
  }}

  .speech-icon {{
    font-size: 8.5pt;
    font-weight: 800;
    color: #0D9488;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
  }}

  .speech-text {{
    font-size: 10.5pt;
    font-weight: 500;
    color: #0F172A;
    line-height: 1.6;
    letter-spacing: -0.01em;
  }}

  /* Stage Directions */
  .stage-direction {{
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 4px solid #F59E0B;
    border-radius: 4px;
    padding: 8px 12px;
    margin: 8px 0;
    font-size: 9pt;
    color: #78350F;
    page-break-inside: avoid;
    line-height: 1.5;
  }}

  /* Badges */
  .badge {{
    display: inline-block;
    padding: 1px 6px;
    border-radius: 3px;
    font-size: 7.5pt;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin-right: 4px;
  }}

  .badge-screen {{
    background: #DBEAFE;
    color: #1E40AF;
    border: 1px solid #BFDBFE;
  }}

  .badge-do {{
    background: #FEF3C7;
    color: #92400E;
    border: 1px solid #FDE68A;
  }}

  .badge-expect {{
    background: #EDE9FE;
    color: #5B21B6;
    border: 1px solid #DDD6FE;
  }}

  /* Tables */
  .table-container {{
    margin: 12px 0 18px 0;
    width: 100%;
    page-break-inside: avoid;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 8.5pt;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    overflow: hidden;
  }}

  th {{
    background: #1E293B;
    color: #FFFFFF;
    font-weight: 700;
    text-align: left;
    padding: 7px 10px;
    border: 1px solid #334155;
  }}

  td {{
    padding: 6px 10px;
    border: 1px solid #E2E8F0;
    color: #1E293B;
    vertical-align: top;
  }}

  tr:nth-child(even) td {{
    background: #F8FAFC;
  }}

  code {{
    font-family: 'JetBrains Mono', Courier, monospace;
    font-size: 8.5pt;
    background: #F1F5F9;
    padding: 1px 4px;
    border-radius: 3px;
    color: #0F172A;
    border: 1px solid #E2E8F0;
  }}

  .divider {{
    height: 1px;
    background: #E2E8F0;
    border: none;
    margin: 18px 0;
  }}

  a {{
    color: #0D9488;
    text-decoration: none;
    font-weight: 600;
  }}
</style>
</head>
<body>

  <div class="hero-box">
    <div class="hero-tag">Happiest Health Digital Twin Challenge 2026 · Bengaluru, India</div>
    <div class="main-title">GlycoTwin: 20-Minute Video Script & Recording Blueprint</div>
    <div class="hero-sub">Complete minute-by-minute teleprompter, stage directions, and demonstration guide for Team Classy Leopards</div>
    <div class="hero-meta">
      <div><b>Team:</b> Classy Leopards</div>
      <div><b>Institution:</b> Goa Institute of Management, Sanquelim</div>
      <div><b>Required Duration:</b> &ge; 20:00 (Planned: ~28:11)</div>
      <div><b>Live System:</b> glycotwin-app.web.app</div>
    </div>
  </div>

{body_content}

</body>
</html>
"""

def generate_pdf():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    md_path = os.path.join(script_dir, "docs", "VIDEO_SCRIPT_20MIN.md")
    temp_html_path = os.path.join(script_dir, "docs", "video_script_temp.html")
    pdf_out_path = os.path.join(script_dir, "docs", "GlycoTwin_Video_Script_20Min.pdf")

    if not os.path.exists(md_path):
        print(f"Error: {md_path} not found.")
        return

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    body_html = parse_markdown_to_html(md_text)
    full_html = build_html_doc(body_html)

    with open(temp_html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Generated intermediate HTML: {temp_html_path}")

    chrome_cmd = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_out_path}",
        temp_html_path
    ]

    print("Running Chrome print-to-pdf...")
    res = subprocess.run(chrome_cmd, capture_output=True, text=True)
    if res.returncode == 0:
        file_size = os.path.getsize(pdf_out_path)
        print(f"Successfully generated PDF: {pdf_out_path} ({file_size:,} bytes)")
    else:
        print(f"Error generating PDF: {res.stderr}")

    if os.path.exists(temp_html_path):
        os.remove(temp_html_path)

if __name__ == "__main__":
    generate_pdf()
