# SkillLens AI — AI-Based Skill Extraction from Job Descriptions
import streamlit as st
import pandas as pd
from pathlib import Path
from dataclasses import asdict
from exporter import generate_csv, generate_json, generate_per_job_csv, generate_pdf_profile
from modules.pipeline import analyze_job

# ─────────────────────── Constants ───────────────────────────

SUPPORTED_FILE_TYPES = ["pdf", "docx", "txt", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"]
MAX_JOBS = 5
REQUIREMENT_OPTIONS = ["Mandatory", "Preferred", "Uncertain"]
STATUS_OPTIONS = ["pending", "approved", "rejected", "edited"]

# ─────────────────────── Helpers ─────────────────────────────


def _load_css():
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def _init_state():
    for key, default in {"jobs": [], "texts": {}, "profiles": {}}.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _safe(d: dict, key: str, fallback=""):
    v = d.get(key)
    return v if v is not None else fallback



def _extract_uploaded_text(uploaded_file):
    """Extract text from an uploaded document or image, using OCR when needed."""
    from modules.document_extractor import extract_text

    result = extract_text(uploaded_file)
    text = getattr(result, "text", result if isinstance(result, str) else "")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("No readable text was extracted from this file.")
    method = getattr(result, "method", "unknown")
    return text.strip(), method


def _profile_to_state(text: str, job_id: str, title: str | None = None, extraction_method: str = "paste", source_filename: str = "") -> dict:
    """Run the extraction/classification pipeline and adapt its result to the UI tables."""
    profile = analyze_job(text, job_id, title or job_id)
    data = asdict(profile)
    skill_rows = []
    for skill in data.get("skills", []):
        skill_rows.append({
            "skill_id": skill.get("skill_id", ""),
            "original_phrase": skill.get("original_phrase", ""),
            "skill": skill.get("skill", ""),
            "category": skill.get("category", "Technical"),
            "requirement": skill.get("requirement", "Uncertain"),
            "basis": skill.get("requirement_basis", "none"),
            "evidence": skill.get("evidence", ""),
            "needs_review": bool(skill.get("needs_review", True)),
            "status": skill.get("user_status", "pending"),
            "review_reason": skill.get("review_reason", ""),
        })

    columns = ["skill_id", "original_phrase", "skill", "category", "requirement", "basis", "evidence", "needs_review", "status", "review_reason"]
    return {
        "skills": pd.DataFrame(skill_rows, columns=columns),
        "experience": data.get("experience", {}),
        "education": data.get("education", {}),
        "certifications": data.get("certifications", []),
        "warnings": data.get("warnings", []),
        "title": title or job_id,
        "raw_text": text,
        "extraction_method": extraction_method,
        "source_filename": source_filename,
    }


def _label(text: str, icon: str = "◆"):
    """Render a monospace section label with an icon."""
    st.markdown(f'<p class="section-label"><span class="shape-icon">{icon}</span>{text}</p>', unsafe_allow_html=True)


def _empty(message: str):
    st.markdown(
        f'<div class="empty-state">'
        f'<div class="empty-label">No data</div>'
        f'<p>{message}</p></div>',
        unsafe_allow_html=True,
    )


def _badge(text: str, variant: str) -> str:
    return f'<span class="badge badge-{variant}">{text}</span>'



# ───────────── Comparison Matrix Logic ───────────────────────


def _build_matrix(profiles: dict):
    rows = []
    for job_id, prof in profiles.items():
        df = prof.get("skills")
        if df is None or df.empty:
            continue
        valid = df[df["status"] != "rejected"]
        for _, r in valid.iterrows():
            rows.append({"job_id": job_id, "skill": r["skill"], "requirement": r["requirement"]})

    if not rows:
        return pd.DataFrame(), [], [], []

    flat = pd.DataFrame(rows)
    matrix = flat.pivot_table(index="skill", columns="job_id", values="requirement", aggfunc="first")
    matrix = matrix.fillna("Not mentioned")

    num_jobs = len(profiles)
    common, unique, differing = [], [], []
    for skill, row in matrix.iterrows():
        mentions = [v for v in row if v != "Not mentioned"]
        if not mentions:
            continue
        elif len(mentions) == 1:
            unique.append(skill)
        elif len(mentions) == num_jobs and len(set(mentions)) == 1:
            common.append(skill)
        else:
            differing.append(skill)

    return matrix, common, unique, differing


def _render_matrix_html(matrix: pd.DataFrame) -> str:
    """Renders the comparison matrix as a clean HTML table to avoid default dark UI styling."""
    html = '<table class="compare-table"><thead><tr><th>Skill</th>'
    for col in matrix.columns:
        html += f'<th>{col}</th>'
    html += '</tr></thead><tbody>'
    
    for skill, row in matrix.iterrows():
        html += f'<tr><td class="skill-name">{skill}</td>'
        for _, val in row.items():
            cls_name = val.lower().replace(" ", "-")
            html += f'<td class="cell-{cls_name}">{val}</td>'
        html += '</tr>'
    html += '</tbody></table>'
    return html


def _style_cell(val):
    colors = {
        "Mandatory":     f"background-color:#D8F0D5;color:#1a5c14;font-weight:600;font-family:'IBM Plex Mono',monospace;font-size:0.78rem;",
        "Preferred":     f"background-color:#FFE4A3;color:#6b4c00;font-weight:600;font-family:'IBM Plex Mono',monospace;font-size:0.78rem;",
        "Uncertain":     f"background-color:#F7D3CC;color:#8b2a1a;font-weight:600;font-family:'IBM Plex Mono',monospace;font-size:0.78rem;",
        "Not mentioned": f"background-color:#E6E3DA;color:#6b6b6b;font-family:'IBM Plex Mono',monospace;font-size:0.78rem;",
    }
    return colors.get(val, "")


def _count_pending(profiles: dict) -> int:
    total = 0
    for prof in profiles.values():
        df = prof.get("skills")
        if df is not None and not df.empty:
            total += int(((df["needs_review"] == True) | (df["status"] == "pending")).sum())
        if _safe(prof.get("experience", {}), "needs_review", False):
            total += 1
        if _safe(prof.get("education", {}), "needs_review", False):
            total += 1
    return total


def _count_approved(profiles: dict) -> int:
    total = 0
    for prof in profiles.values():
        df = prof.get("skills")
        if df is not None and not df.empty:
            total += int((df["status"] == "approved").sum())
    return total


def _count_skills(profiles: dict) -> int:
    return sum(len(p.get("skills", pd.DataFrame())) for p in profiles.values())


# ═══════════════════════ Sidebar ═════════════════════════════


def _sidebar():
    with st.sidebar:
        st.markdown(
            '<div class="app-header__brand">'
            '<span class="app-header__name">SkillLens AI</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        n_jobs = len(st.session_state.profiles)
        n_skills = _count_skills(st.session_state.profiles)
        n_pending = _count_pending(st.session_state.profiles)
        n_approved = _count_approved(st.session_state.profiles)

        _label("Status")
        c1, c2 = st.columns(2)
        c1.metric("Jobs", n_jobs)
        c2.metric("Skills", n_skills)
        c3, c4 = st.columns(2)
        c3.metric("Pending", n_pending)
        c4.metric("Approved", n_approved)


# ═══════════════════ Tab 1: Input ════════════════════════════


def _tab_input():
    st.markdown("### Input")
    st.caption("ADD JOB DESCRIPTIONS — PASTE TEXT OR UPLOAD PDF / DOCX / TXT / IMAGE FILES (OCR SUPPORTED). UP TO 5 JOBS.")

    col_upload, col_text = st.columns([1, 2], gap="large")
    uploaded = []

    with col_upload:
        _label("Upload files")
        uploaded = st.file_uploader(
            "Upload files",
            type=SUPPORTED_FILE_TYPES,
            accept_multiple_files=True,
            key="file_uploader",
            label_visibility="collapsed",
            help="Images are read with OCR. Scanned PDFs are rendered and passed through OCR when selectable text is not available.",
        ) or []
        if uploaded:
            for f in uploaded:
                st.success(f"{f.name} — {f.size / 1024:.1f} KB")

    with col_text:
        _label("Raw text input", icon="▶")
        for i in range(MAX_JOBS):
            job_label = f"Job {chr(65 + i)}"
            existing = ""
            for j in st.session_state.jobs:
                if j.startswith(job_label):
                    existing = st.session_state.texts.get(j, "")
                    break
            with st.expander(f"{job_label} Input", expanded=(i < len(st.session_state.jobs) or i == 0)):
                st.text_area(
                    f"Paste {job_label} description",
                    value=existing,
                    height=200,
                    key=f"text_input_{i}",
                    placeholder="Paste a job description here…",
                    label_visibility="collapsed",
                )

    if st.button("Analyze job descriptions", type="primary", use_container_width=True):
        # Pasted descriptions and uploaded files share a single limit of five jobs.
        inputs = []
        for i in range(MAX_JOBS):
            text = str(st.session_state.get(f"text_input_{i}", "") or "").strip()
            if text:
                job_id = f"Job {chr(65 + i)}"
                inputs.append({"job_id": job_id, "text": text, "method": "paste", "filename": ""})

        seen_ids = {item["job_id"].casefold() for item in inputs}
        if len(inputs) < MAX_JOBS:
            for f in uploaded:
                if len(inputs) >= MAX_JOBS:
                    st.warning(f"The app supports at most {MAX_JOBS} job descriptions per analysis. Remaining uploads were skipped.")
                    break
                try:
                    extracted_text, extraction_method = _extract_uploaded_text(f)
                except Exception as exc:
                    st.error(f"Could not read **{f.name}**: {exc}")
                    continue

                stem = Path(f.name).stem.strip() or "Uploaded job"
                base_id = f"Upload — {stem}"
                job_id = base_id
                duplicate = 2
                while job_id.casefold() in seen_ids:
                    job_id = f"{base_id} ({duplicate})"
                    duplicate += 1
                seen_ids.add(job_id.casefold())
                inputs.append({"job_id": job_id, "text": extracted_text, "method": extraction_method, "filename": f.name})
        elif uploaded:
            st.warning(f"You already entered {MAX_JOBS} text descriptions; uploads were skipped to respect the {MAX_JOBS}-job limit.")

        successes = {}
        for item in inputs:
            try:
                successes[item["job_id"]] = _profile_to_state(
                    item["text"],
                    item["job_id"],
                    title=item["job_id"],
                    extraction_method=item["method"],
                    source_filename=item["filename"],
                )
            except Exception as exc:
                st.error(f"Analysis failed for **{item['job_id']}**: {exc}")

        if successes:
            st.session_state.jobs = list(successes.keys())
            st.session_state.texts = {job_id: prof["raw_text"] for job_id, prof in successes.items()}
            st.session_state.profiles = successes
            st.success(f"Analyzed {len(successes)} job description(s). Open **Review Profiles** to inspect and correct the results.")
            for job_id, prof in successes.items():
                if prof.get("extraction_method") in {"image_ocr", "pdf_ocr"}:
                    st.info(f"OCR succeeded for **{job_id}** using `{prof['extraction_method']}`. Open Review Profiles → Extracted source text to verify what was read.")
                for warning in prof.get("warnings", []):
                    st.warning(f"{job_id}: {warning}")
        elif not uploaded and not any(str(st.session_state.get(f"text_input_{i}", "") or "").strip() for i in range(MAX_JOBS)):
            st.warning("Paste at least one job description or upload a PDF, DOCX, TXT, or image file first.")
        elif not inputs:
            st.warning("No files could be read and no pasted job description was available. Check the OCR setup and try again.")

    if st.session_state.jobs:
        st.markdown("---")
        _label("Loaded jobs")
        for idx, job_id in enumerate(st.session_state.jobs):
            prof = st.session_state.profiles.get(job_id, {})
            skills_count = len(prof.get("skills", pd.DataFrame()))
            method = prof.get("extraction_method", "paste")
            st.markdown(
                f'<div class="job-row">'
                f'<span class="job-row__name">{job_id}</span>'
                f'<span class="job-row__meta">{skills_count} skills extracted &mdash; {method}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ═══════════════ Tab 2: Requirement Profiles ═════════════════


def _generate_req_card_html(job_id: str, prof: dict) -> str:
    exp = prof.get("experience", {})
    edu = prof.get("education", {})
    
    exp_min = _safe(exp, "min_years", 0)
    try:
        min_years_val = int(exp_min)
    except:
        min_years_val = 0
        
    exp_text = f"{min_years_val}+ years" if min_years_val > 0 else "Entry Level / Not specified"
    
    edu_level = _safe(edu, "level", "Not specified")
    if not edu_level: edu_level = "Not specified"
    
    edu_field = _safe(edu, "field", "")
    edu_text = f"{edu_level}"
    if edu_field:
        edu_text += f" in {edu_field}"
    if _safe(edu, "equivalent_allowed", False):
        edu_text += " (or equivalent)"
        
    skills_df = prof.get("skills", pd.DataFrame())
    
    # Split skills by category safely
    if not skills_df.empty and "category" in skills_df.columns:
        tech_df = skills_df[skills_df["category"].str.contains("Tech", case=False, na=False)]
        non_tech_df = skills_df[~skills_df["category"].str.contains("Tech", case=False, na=False)]
    else:
        tech_df = skills_df
        non_tech_df = pd.DataFrame()

    def _get_chips_html(df):
        if df.empty:
            return '<div class="req-card__label" style="text-transform: none;">None identified</div>'
        chips_html = []
        for _, row in df.iterrows():
            skill = row["skill"]
            req = row["requirement"]
            cls_name = "common" if req == "Mandatory" else "unique" if req == "Preferred" else "differing"
            chips_html.append(f'<span class="skill-chip {cls_name}" style="margin: 0;">{skill}</span>')
        return f'<div class="req-card__chips">{" ".join(chips_html)}</div>'
    
    html = f"""<div class="req-card">
    <div class="req-card__header">{job_id}</div>
    <div class="req-card__grid" style="grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));">
        <div class="req-card__block">
            <div class="req-card__label">Experience Required</div>
            <div class="req-card__value">{exp_text}</div>
        </div>
        <div class="req-card__block">
            <div class="req-card__label">Education Required</div>
            <div class="req-card__value">{edu_text}</div>
        </div>
    </div>
    <div class="req-card__section">
        <div class="req-card__section-title">Technical Skills</div>
        {_get_chips_html(tech_df)}
    </div>
    <div class="req-card__section">
        <div class="req-card__section-title">Non-Technical / Soft Skills</div>
        {_get_chips_html(non_tech_df)}
    </div>
</div>"""
    return html


def _tab_req_profiles():
    st.markdown("### Requirement Profiles")
    st.caption("HIGH-LEVEL SUMMARY OF EXTRACTED REQUIREMENTS")

    if not st.session_state.profiles:
        _empty("No profiles loaded. Go to the Input tab and upload a file or paste a job description.")
        return

    for job_id, prof in st.session_state.profiles.items():
        st.markdown(_generate_req_card_html(job_id, prof), unsafe_allow_html=True)


# ═══════════════ Tab 3: Profile & Review ═════════════════════


def _tab_profile():
    st.markdown("### Profile Review")

    if not st.session_state.profiles:
        _empty("No profiles loaded. Go to the Input tab and upload a file or paste a job description.")
        return

    job_ids = list(st.session_state.profiles.keys())

    # Control row: selector + stats
    col_sel, col_s1, col_s2, col_s3, col_s4 = st.columns([3, 1, 1, 1, 1])
    with col_sel:
        selected = st.selectbox("Job", job_ids, key="profile_job_select", label_visibility="collapsed")

    profile = st.session_state.profiles[selected]

    extraction_method = profile.get("extraction_method", "paste")
    source_filename = profile.get("source_filename", "")
    if source_filename:
        st.caption(f"Source file: {source_filename} · Extraction method: {extraction_method}")
    else:
        st.caption(f"Extraction method: {extraction_method}")
    raw_text = profile.get("raw_text", "")
    with st.expander("Extracted source text (verify OCR output)", expanded=extraction_method in {"image_ocr", "pdf_ocr"}):
        if raw_text:
            st.text_area("Extracted text", value=raw_text, height=220, disabled=True, key=f"source_text_{selected}")
        else:
            st.caption("No source text is stored for this profile.")
    if profile.get("warnings"):
        with st.expander("Pipeline warnings", expanded=False):
            for warning in profile["warnings"]:
                st.warning(warning)

    skills_df = profile.get("skills", pd.DataFrame())

    if not skills_df.empty:
        with col_s1:
            st.metric("Total", len(skills_df))
        with col_s2:
            st.metric("Mandatory", int((skills_df["requirement"] == "Mandatory").sum()))
        with col_s3:
            st.metric("Preferred", int((skills_df["requirement"] == "Preferred").sum()))
        with col_s4:
            n_rev = int(skills_df["needs_review"].sum()) if "needs_review" in skills_df.columns else 0
            st.metric("To review", n_rev)

    st.markdown("---")

    # Skills table
    _label("Extracted skills")

    if skills_df.empty:
        _empty("No skills extracted for this job.")
    else:
        edited = st.data_editor(
            skills_df,
            column_config={
                "original_phrase": st.column_config.TextColumn("Original Phrase", width="medium"),
                "skill":          st.column_config.TextColumn("Normalized", width="small"),
                "category":       st.column_config.TextColumn("Category", width="small"),
                "requirement":    st.column_config.SelectboxColumn("Requirement", options=REQUIREMENT_OPTIONS, width="small"),
                "basis":          st.column_config.TextColumn("Basis", width="medium"),
                "evidence":       st.column_config.TextColumn("Evidence", width="large"),
                "needs_review":   st.column_config.CheckboxColumn("Review", width="small"),
                "status":         st.column_config.SelectboxColumn("Status", options=STATUS_OPTIONS, width="small"),
            },
            hide_index=True,
            num_rows="dynamic",
            use_container_width=True,
            key=f"editor_{selected}",
        )
        st.session_state.profiles[selected]["skills"] = edited

    st.markdown("---")

    # Experience & Education — structured form sections
    col_exp, col_edu = st.columns(2, gap="large")

    with col_exp:
        _label("Experience")
        exp = profile.get("experience", {})
        exp_min = st.number_input("Minimum years", value=int(_safe(exp, "min_years", 0)), min_value=0, key=f"exp_min_{selected}")
        exp_evid = st.text_area("Evidence", value=_safe(exp, "evidence"), height=72, key=f"exp_evid_{selected}")
        exp_rev = st.checkbox("Flag for review", value=bool(_safe(exp, "needs_review", False)), key=f"exp_rev_{selected}")
        st.session_state.profiles[selected]["experience"] = {**exp, "min_years": exp_min, "evidence": exp_evid, "needs_review": exp_rev}

    with col_edu:
        _label("Education")
        edu = profile.get("education", {})
        edu_level = st.text_input("Level", value=_safe(edu, "level"), key=f"edu_level_{selected}")
        edu_field = st.text_input("Field", value=_safe(edu, "field"), key=f"edu_field_{selected}")
        edu_evid = st.text_area("Evidence", value=_safe(edu, "evidence"), height=72, key=f"edu_evid_{selected}")
        edu_eq = st.checkbox("Equivalent accepted", value=bool(_safe(edu, "equivalent_allowed", False)), key=f"edu_eq_{selected}")
        edu_rev = st.checkbox("Flag for review", value=bool(_safe(edu, "needs_review", False)), key=f"edu_rev_{selected}")
        st.session_state.profiles[selected]["education"] = {**edu, "level": edu_level, "field": edu_field, "evidence": edu_evid, "equivalent_allowed": edu_eq, "needs_review": edu_rev}


# ═══════════════════ Tab 3: Compare ══════════════════════════


def _tab_compare():
    st.markdown("### Compare")

    if not st.session_state.profiles:
        _empty("Add at least two jobs to generate a comparison matrix.")
        return

    if len(st.session_state.profiles) < 2:
        st.info("Add at least two jobs to compare. Currently only one job is loaded.")
        return

    pending = _count_pending(st.session_state.profiles)
    if pending > 0:
        st.warning(f"{pending} items still awaiting human review across all jobs.")
    else:
        st.success("All items have been reviewed.")

    matrix, common, unique, differing = _build_matrix(st.session_state.profiles)

    if matrix.empty:
        _empty("No non-rejected skills to compare.")
        return

    # Legend
    st.markdown(
        '<div class="legend">'
        '<div class="legend__item"><span class="legend__swatch" style="background:#D8F0D5;"></span>Mandatory</div>'
        '<div class="legend__item"><span class="legend__swatch" style="background:#FFE4A3;"></span>Preferred</div>'
        '<div class="legend__item"><span class="legend__swatch" style="background:#F7D3CC;"></span>Uncertain</div>'
        '<div class="legend__item"><span class="legend__swatch" style="background:#E6E3DA;"></span>Not mentioned</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Matrix HTML Render
    html_matrix = _render_matrix_html(matrix)
    st.markdown(html_matrix, unsafe_allow_html=True)

    st.markdown("---")

    # Skill groups
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown(
            '<div class="skills-group">'
            '<div class="skills-group__header">Common</div>'
            '<div class="skills-group__desc">Same requirement in every job</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if common:
            chips = " ".join(f'<span class="skill-chip common">{s}</span>' for s in common)
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.markdown('<span class="skills-group__empty">None</span>', unsafe_allow_html=True)

    with col2:
        st.markdown(
            '<div class="skills-group">'
            '<div class="skills-group__header">Unique</div>'
            '<div class="skills-group__desc">Found in only one job</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if unique:
            chips = " ".join(f'<span class="skill-chip unique">{s}</span>' for s in unique)
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.markdown('<span class="skills-group__empty">None</span>', unsafe_allow_html=True)

    with col3:
        st.markdown(
            '<div class="skills-group">'
            '<div class="skills-group__header">Differing</div>'
            '<div class="skills-group__desc">Different labels or partially missing</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if differing:
            chips = " ".join(f'<span class="skill-chip differing">{s}</span>' for s in differing)
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.markdown('<span class="skills-group__empty">None</span>', unsafe_allow_html=True)


# ═══════════════ Tab 4: Summary & Export ═════════════════════


def _tab_export():
    st.markdown("### Summary & Export")

    if not st.session_state.profiles:
        _empty("Nothing to export. Load or enter job descriptions first.")
        return


    profiles = st.session_state.profiles
    n_jobs = len(profiles)
    n_skills = _count_skills(profiles)
    n_approved = _count_approved(profiles)
    n_pending = _count_pending(profiles)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jobs", n_jobs)
    c2.metric("Skills", n_skills)
    c3.metric("Approved", n_approved)
    c4.metric("Pending", n_pending)

    st.markdown("---")

    _label("Main Export: Requirement Profiles (PDF)", icon="★")
    st.caption("DOWNLOAD A STATIC, BEAUTIFULLY FORMATTED PDF IMAGE OF EACH JOB'S REQUIREMENTS PROFILE.")
    
    # Generate PDFs
    cols = st.columns(3)
    for idx, (job_id, prof) in enumerate(profiles.items()):
        pdf_bytes = generate_pdf_profile(job_id, prof)
        
        with cols[idx % 3]:
            safe_name = job_id.replace(" ", "_").replace("—", "-")[:40]
            st.download_button(
                label=f"Download {job_id[:20]} PDF",
                data=pdf_bytes,
                file_name=f"{safe_name}_profile.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
                key=f"dl_pdf_{idx}"
            )

    st.markdown("---")
    
    with st.expander("Secondary Exports (CSV & JSON Data)", expanded=False):
        col_matrix, col_json, col_jobs = st.columns(3, gap="medium")

        with col_matrix:
            _label("Comparison matrix — CSV")
            st.caption("SKILL-VS-JOB MATRIX.")
            matrix, _, _, _ = _build_matrix(profiles)
            if matrix.empty:
                st.caption("No matrix data.")
            else:
                csv_data = generate_csv(matrix)
                st.download_button(
                    "Download Matrix CSV",
                    data=csv_data,
                    file_name="skilllens_comparison_matrix.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_csv_matrix",
                )

        with col_json:
            _label("Full profiles — JSON")
            st.caption("ALL PROFILES DATA AS JSON.")
            json_data = generate_json(profiles)
            st.download_button(
                "Download Profiles JSON",
                data=json_data,
                file_name="skilllens_profiles.json",
                mime="application/json",
                use_container_width=True,
                key="dl_json",
            )
            
        with col_jobs:
            _label("Per-job skill CSVs")
            st.caption("INDIVIDUAL SKILL TABLES.")
            for job_id in profiles:
                csv_single = generate_per_job_csv(profiles, job_id)
                if csv_single:
                    safe_name = job_id.replace(" ", "_").replace("—", "-")[:40]
                    st.download_button(
                        f"{job_id[:20]} CSV",
                        data=csv_single,
                        file_name=f"skilllens_{safe_name}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key=f"dl_csv_{job_id}",
                    )


# ═══════════════════════ Main ════════════════════════════════


def main():
    st.set_page_config(
        page_title="SkillLens AI",
        page_icon="S",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _load_css()
    _init_state()
    _sidebar()

    # App header
    st.markdown(
        '<div class="app-header">'
        '<div class="app-header__brand">'
        '<span class="app-header__name">SkillLens AI</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # Clean splits: Navigate via Sidebar to avoid clustered tabs
    with st.sidebar:
        st.markdown("---")
        _label("Navigation", icon="❖")
        view_mode = st.radio(
            "Select View",
            ["Input Data", "Requirement Profiles", "Review Profiles", "Compare Skills", "Export & Summary"],
            label_visibility="collapsed"
        )
        st.markdown("---")

    if view_mode == "Input Data":
        _tab_input()
    elif view_mode == "Requirement Profiles":
        _tab_req_profiles()
    elif view_mode == "Review Profiles":
        _tab_profile()
    elif view_mode == "Compare Skills":
        _tab_compare()
    elif view_mode == "Export & Summary":
        _tab_export()


if __name__ == "__main__":
    main()
