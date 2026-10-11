import pandas as pd
import json
import numpy as np
try:
    from fpdf import FPDF
except ImportError:
    FPDF = None


class _SafeEncoder(json.JSONEncoder):
    """Handle NumPy types, DataFrames, and other non-native JSON types."""

    def default(self, obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if obj is pd.NaT or (isinstance(obj, float) and np.isnan(obj)):
            return None
        return super().default(obj)


def generate_csv(matrix: pd.DataFrame) -> str:
    """Export the comparison matrix to a CSV string."""
    if matrix is None or matrix.empty:
        return ""
    return matrix.to_csv()


def generate_json(profiles: dict) -> str:
    """Export the full profiles dict (including DataFrames) to safe JSON."""
    if not profiles:
        return "{}"

    export_dict = {}
    for job_id, profile in profiles.items():
        entry = {
            "experience": profile.get("experience", {}),
            "education": profile.get("education", {}),
        }

        skills_df = profile.get("skills")
        if skills_df is not None and not skills_df.empty:
            entry["skills"] = skills_df.to_dict(orient="records")
        else:
            entry["skills"] = []

        export_dict[job_id] = entry

    return json.dumps(export_dict, indent=2, cls=_SafeEncoder)


def generate_per_job_csv(profiles: dict, job_id: str) -> str:
    """Export a single job's skill profile to CSV."""
    profile = profiles.get(job_id)
    if not profile:
        return ""
    skills_df = profile.get("skills")
    if skills_df is None or skills_df.empty:
        return ""
    return skills_df.to_csv(index=False)


def generate_pdf_profile(job_id: str, prof: dict) -> bytes:
    """Generate a static, beautifully formatted PDF image of the requirement profile."""
    if FPDF is None:
        return b""
        
    exp = prof.get("experience", {})
    edu = prof.get("education", {})
    exp_min = exp.get("min_years", 0)
    try: min_val = int(exp_min)
    except: min_val = 0
    exp_text = f"{min_val}+ years" if min_val > 0 else "Entry Level"
    
    edu_level = edu.get("level", "Not specified")
    if not edu_level: edu_level = "Not specified"
    edu_field = edu.get("field", "")
    edu_text = f"{edu_level}" + (f" in {edu_field}" if edu_field else "")
    
    skills_df = prof.get("skills", pd.DataFrame())
    
    tech_skills = []
    non_tech_skills = []
    if not skills_df.empty:
        for _, row in skills_df.iterrows():
            s = row.get("skill", "")
            if not s: continue
            req = row.get("requirement", "")
            cat = str(row.get("category", ""))
            safe_s = str(s).encode('ascii', 'replace').decode('ascii')
            if "Tech" not in cat:
                non_tech_skills.append((safe_s, req))
            else:
                tech_skills.append((safe_s, req))
                
    pdf = FPDF(format="A4")
    pdf.add_page()
    
    # Background
    pdf.set_fill_color(246, 243, 234)
    pdf.rect(0, 0, 210, 297, "F")
    
    # Shadow for Card
    pdf.set_fill_color(23, 23, 23)
    pdf.rect(17, 23, 176, 240, "F")
    
    # Card surface
    pdf.set_fill_color(255, 253, 247)
    pdf.set_draw_color(23, 23, 23)
    pdf.set_line_width(0.8)
    pdf.rect(15, 20, 176, 240, "DF")
    
    # Header
    pdf.set_font("Helvetica", style="B", size=20)
    pdf.set_text_color(23, 23, 23)
    pdf.set_xy(25, 30)
    safe_title = job_id.encode('ascii', 'replace').decode('ascii')
    pdf.cell(0, 10, safe_title, ln=True)
    
    # Separator
    pdf.line(15, 45, 191, 45)
    
    # Grid: Experience and Education Blocks
    pdf.set_fill_color(246, 243, 234)
    pdf.rect(25, 52, 75, 20, "DF")
    pdf.set_font("Helvetica", style="B", size=9)
    pdf.set_text_color(107, 107, 107)
    pdf.set_xy(28, 55)
    pdf.cell(70, 5, "EXPERIENCE REQUIRED")
    pdf.set_text_color(23, 23, 23)
    pdf.set_font("Helvetica", style="B", size=11)
    pdf.set_xy(28, 62)
    pdf.cell(70, 5, exp_text)
    
    pdf.set_fill_color(246, 243, 234)
    pdf.rect(106, 52, 75, 20, "DF")
    pdf.set_font("Helvetica", style="B", size=9)
    pdf.set_text_color(107, 107, 107)
    pdf.set_xy(109, 55)
    pdf.cell(70, 5, "EDUCATION REQUIRED")
    pdf.set_text_color(23, 23, 23)
    pdf.set_font("Helvetica", style="B", size=11)
    pdf.set_xy(109, 62)
    pdf.cell(70, 5, edu_text[:35] + ("..." if len(edu_text)>35 else ""))
    
    # ── Technical Skills ──
    y_pos = 85
    pdf.set_line_width(0.5)
    pdf.line(25, y_pos, 181, y_pos)
    
    y_pos += 8
    pdf.set_font("Helvetica", style="B", size=11)
    pdf.set_text_color(23, 23, 23)
    pdf.set_xy(25, y_pos)
    pdf.cell(0, 5, "TECHNICAL SKILLS", ln=True)
    
    y_pos += 8
    
    def draw_chips(pdf, x, y, max_x, skills_data):
        cx, cy, lh = x, y, 8
        pdf.set_font("Helvetica", style="B", size=9)
        pdf.set_text_color(23, 23, 23)
        if not skills_data:
            pdf.set_xy(cx, cy)
            pdf.cell(50, lh, "None identified")
            return cy + lh
            
        for skill, req in skills_data:
            if req == "Mandatory": pdf.set_fill_color(216, 240, 213)
            elif req == "Preferred": pdf.set_fill_color(255, 228, 163)
            else: pdf.set_fill_color(247, 211, 204)
            pdf.set_draw_color(23, 23, 23)
            pdf.set_line_width(0.3)
            
            w = pdf.get_string_width(skill) + 6
            if cx + w > max_x:
                cx = x
                cy += lh + 3
            pdf.set_xy(cx, cy)
            pdf.cell(w, lh, skill, border=1, fill=True, align="C")
            cx += w + 3
        return cy + lh

    y_pos = draw_chips(pdf, 25, y_pos, 181, tech_skills)
    
    # ── Non-Technical Skills ──
    y_pos += 12
    pdf.set_line_width(0.5)
    pdf.line(25, y_pos, 181, y_pos)
    
    y_pos += 8
    pdf.set_font("Helvetica", style="B", size=11)
    pdf.set_xy(25, y_pos)
    pdf.cell(0, 5, "NON-TECHNICAL / SOFT SKILLS", ln=True)
    
    y_pos += 8
    y_pos = draw_chips(pdf, 25, y_pos, 181, non_tech_skills)
    
    return bytes(pdf.output())
