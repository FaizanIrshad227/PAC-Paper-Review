import os, json, base64, re, time, uuid, threading
from flask import Flask, render_template, request, jsonify, send_file, session
import fitz
import anthropic
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

app = Flask(__name__)
app.secret_key = "pac-review-secret-2026"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

UPLOAD_FOLDER  = "/tmp/pac_uploads"
REPORT_FOLDER  = "/tmp/pac_reports"
BATCH_SIZE     = 5     # pages per Claude call
MAX_RETRIES    = 3
MAX_FILE_SIZE  = 50 * 1024 * 1024
IMAGE_DPI      = 72    # lower = smaller files = faster upload to Claude

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Job store: job_id -> {"status": ..., "log": [...], "report_path": ...}
jobs = {}
jobs_lock = threading.Lock()

# ══════════════════════════════════════════════════════════════════════════════
#  GRADING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert examiner reviewing a student's handwritten answer sheet.

EXAM: {exam_title}

MODEL SOLUTION (this is the ONLY reference for what is correct or incorrect):
{solution_text}

QUESTION PAPER REFERENCE:
{mark_scheme}

STRICT RULES:
1. Compare the student's answer ONLY against the MODEL SOLUTION above.
2. Do NOT use your own knowledge to judge correctness — only the solution counts.
3. Do NOT include any marks or scores.
4. For each question, list what matches the solution and what is missing/wrong vs the solution.
5. If a question is not attempted, note "Not attempted".
6. Be specific — reference actual numbers, steps, and concepts from both the student's
   answer and the model solution.

Return ONLY a JSON object — no markdown, no extra text:
{{
  "questions": {{
    "<question_key>": {{
      "status": "correct" | "partial" | "incorrect" | "not_attempted",
      "correct_points": ["points the student got right per the solution"],
      "missing_or_wrong": ["what is missing or wrong compared to the solution"],
      "comments": "specific comparison between student answer and model solution"
    }}
  }},
  "overall_feedback": "2-3 sentence summary comparing student performance to the model solution"
}}
"""

def pdfs_to_text(paths):
    out = []
    for path in paths:
        doc = fitz.open(path)
        for page in doc:
            out.append(page.get_text())
        doc.close()
    return "\n".join(out).strip()

def pdfs_to_images(paths, dpi=IMAGE_DPI):
    imgs = []
    mat  = fitz.Matrix(dpi / 72, dpi / 72)
    for path in paths:
        doc = fitz.open(path)
        for page in doc:
            pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            data = base64.standard_b64encode(pix.tobytes("jpeg", jpg_quality=80)).decode()
            imgs.append(data)
        doc.close()
    return imgs

def extract_mark_scheme(paths):
    text  = pdfs_to_text(paths)
    lines = [l.strip() for l in text.splitlines() if re.search(r'\(\d+\)', l)]
    return "\n".join(lines) if lines else "(See question paper)"

def parse_json(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())

def call_claude(client, system_prompt, images_b64, instruction):
    content = [
        {"type": "image",
         "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}
        for b64 in images_b64
    ]
    content.append({"type": "text", "text": instruction})
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=8192,
                system=system_prompt,
                messages=[{"role": "user", "content": content}]
            )
            return resp.content[0].text
        except anthropic.APIConnectionError:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(5 * attempt)

def merge_batches(batches):
    merged    = {"questions": {}, "overall_feedback": ""}
    feedbacks = []
    for b in batches:
        merged["questions"].update(b.get("questions", {}))
        feedbacks.append(b.get("overall_feedback", ""))
    merged["overall_feedback"] = " | ".join(f for f in feedbacks if f)
    return merged

def grade_student(q_paths, s_paths, student_paths, log_fn):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    log_fn("Agent is extracting solution text…")
    solution_text = pdfs_to_text(s_paths)[:14000]
    log_fn("Agent is extracting mark scheme…")
    mark_scheme   = extract_mark_scheme(q_paths)
    exam_title    = ", ".join(os.path.splitext(os.path.basename(p))[0] for p in q_paths)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        exam_title=exam_title, solution_text=solution_text, mark_scheme=mark_scheme)

    log_fn("Agent is converting answer sheet to images…")
    all_imgs  = pdfs_to_images(student_paths)
    n_pages   = len(all_imgs)
    n_batches = (n_pages + BATCH_SIZE - 1) // BATCH_SIZE
    log_fn(f"Agent detected {n_pages} pages → {n_batches} batch(es)")

    results = []
    for i in range(n_batches):
        batch = all_imgs[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        p1, p2 = i * BATCH_SIZE + 1, i * BATCH_SIZE + len(batch)
        log_fn(f"Agent is reviewing batch {i+1}/{n_batches} (pages {p1}–{p2})…")
        instr = (f"Batch {i+1}/{n_batches} (pages {p1}–{p2}). "
                 "Grade every question visible. Return ONLY the JSON response."
                 if n_batches > 1 else
                 "Grade every question. Return ONLY the JSON response.")
        raw = call_claude(client, system_prompt, batch, instr)
        results.append(parse_json(raw))
        log_fn(f"Agent completed batch {i+1} ✓")

    return results[0] if n_batches == 1 else merge_batches(results)

# ══════════════════════════════════════════════════════════════════════════════
#  REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

STATUS_COLOUR = {"correct":"#1a6b1a","partial":"#d4870e",
                 "incorrect":"#b00000","not_attempted":"#555555"}
STATUS_LABEL  = {"correct":"✓ Correct","partial":"◑ Partial",
                 "incorrect":"✗ Incorrect","not_attempted":"— Not Attempted"}

def safe(text):
    if not isinstance(text, str): text = str(text)
    return text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def generate_report(results, student_label, exam_title, out_path):
    PAC_BLUE  = "#003087"
    PAC_LIGHT = "#e8f4fd"
    PAC_GREEN = "#1a6b1a"
    PAC_RED   = "#b00000"
    ACCENT    = "#005baa"

    styles  = getSampleStyleSheet()
    s_title = ParagraphStyle("T",  parent=styles["Title"],   fontSize=14, spaceAfter=4)
    s_sub   = ParagraphStyle("S",  parent=styles["Normal"],  fontSize=9,  spaceAfter=2,
                             textColor=colors.HexColor("#555555"))
    s_h1    = ParagraphStyle("H1", parent=styles["Heading1"],fontSize=11, spaceAfter=3,
                             textColor=colors.HexColor(PAC_BLUE))
    s_h2    = ParagraphStyle("H2", parent=styles["Heading2"],fontSize=10, spaceAfter=3,
                             textColor=colors.HexColor(ACCENT))
    s_body  = ParagraphStyle("B",  parent=styles["Normal"],  fontSize=9, leading=13, spaceAfter=2)
    s_ok    = ParagraphStyle("OK", parent=s_body, textColor=colors.HexColor(PAC_GREEN))
    s_err   = ParagraphStyle("Er", parent=s_body, textColor=colors.HexColor(PAC_RED))
    s_note  = ParagraphStyle("N",  parent=s_body, fontSize=8,
                             textColor=colors.HexColor("#7a5c00"))

    doc   = SimpleDocTemplate(out_path, pagesize=A4,
                              leftMargin=2*cm, rightMargin=2*cm,
                              topMargin=2*cm,  bottomMargin=2*cm)
    story = []
    story.append(Paragraph("The Professionals' Academy of Commerce (PAC)", s_title))
    story.append(Paragraph(safe(exam_title), s_h1))
    story.append(Paragraph(f"File: <b>{safe(student_label)}</b>", s_sub))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor(PAC_BLUE)))
    story.append(Spacer(1, 0.3*cm))

    questions = results.get("questions", {})
    if questions:
        rows = [["Question", "Status"]]
        for qk, qv in questions.items():
            rows.append([safe(qk), STATUS_LABEL.get(qv.get("status",""), "")])
        qt = Table(rows, colWidths=[11*cm, 6*cm])
        tst = [
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor(PAC_BLUE)),
            ("TEXTCOLOR", (0,0),(-1,0),colors.white),
            ("FONTNAME",  (0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",  (0,0),(-1,-1),9),
            ("ALIGN",     (1,0),(-1,-1),"CENTER"),
            ("GRID",      (0,0),(-1,-1),0.3,colors.lightgrey),
            ("ROWHEIGHT", (0,0),(-1,-1),16),
        ]
        for ri, (_, qv) in enumerate(questions.items(), 1):
            st  = qv.get("status","")
            col = STATUS_COLOUR.get(st,"#000000")
            bg  = ("#f0fff0" if st=="correct" else "#fffbe6" if st=="partial"
                   else "#fff0f0" if st=="incorrect" else "#f8f8f8")
            tst += [("TEXTCOLOR",(1,ri),(1,ri),colors.HexColor(col)),
                    ("FONTNAME", (1,ri),(1,ri),"Helvetica-Bold"),
                    ("BACKGROUND",(0,ri),(-1,ri),colors.HexColor(bg))]
        qt.setStyle(TableStyle(tst))
        story.append(qt)
        story.append(Spacer(1, 0.4*cm))

    if results.get("overall_feedback"):
        story.append(Paragraph("Overall Feedback", s_h1))
        story.append(Paragraph(safe(results["overall_feedback"]), s_body))
        story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Detailed Question Review", s_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor(ACCENT)))

    for qk, qv in questions.items():
        st  = qv.get("status","")
        col = STATUS_COLOUR.get(st,"#000000")
        story.append(Spacer(1, 0.25*cm))
        story.append(Paragraph(
            f"<b>{safe(qk)}</b>  <font color='{col}'><b>{safe(STATUS_LABEL.get(st,st))}</b></font>",
            s_h2))
        for pt in qv.get("correct_points",[]):
            story.append(Paragraph(f"  ✓  {safe(pt)}", s_ok))
        for er in qv.get("missing_or_wrong",[]):
            story.append(Paragraph(f"  ✗  {safe(er)}", s_err))
        if qv.get("comments"):
            story.append(Paragraph(safe(qv["comments"]), s_note))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))

    doc.build(story)
    return out_path

# ══════════════════════════════════════════════════════════════════════════════
#  BACKGROUND JOB RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_job(job_id, q_paths, s_paths, students):
    def log(msg):
        with jobs_lock:
            jobs[job_id]["log"].append(msg)

    reports = []
    exam_title = ", ".join(os.path.splitext(os.path.basename(p))[0] for p in q_paths)

    try:
        for i, (name, paths) in enumerate(students, 1):
            log(f"[{i}/{len(students)}] Agent is reviewing: {name}")
            try:
                results  = grade_student(q_paths, s_paths, paths, log)
                out_path = os.path.join(REPORT_FOLDER, f"{job_id}_{i}.pdf")
                generate_report(results, name, exam_title, out_path)
                reports.append((name, out_path))
                log(f"✓ Agent has completed the review for {name}")
            except Exception as e:
                log(f"✗ Error for {name}: {e}")

        # If multiple students, zip them
        if len(reports) == 1:
            final_path = reports[0][1]
        else:
            import zipfile
            final_path = os.path.join(REPORT_FOLDER, f"{job_id}_all_reports.zip")
            with zipfile.ZipFile(final_path, "w") as zf:
                for name, rpath in reports:
                    zf.write(rpath, f"{name}_Review.pdf")

        with jobs_lock:
            jobs[job_id]["status"]      = "done"
            jobs[job_id]["report_path"] = final_path
            jobs[job_id]["report_name"] = (
                f"{reports[0][0]}_Review.pdf" if len(reports)==1
                else "PAC_Review_Reports.zip"
            )
        log("All done!")

    except Exception as e:
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"]  = str(e)
        log(f"Fatal error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════════

def save_uploads(files, prefix, job_id):
    saved = []
    for f in files:
        if f and f.filename:
            fname = f"{job_id}_{prefix}_{uuid.uuid4().hex[:6]}_{f.filename}"
            path  = os.path.join(UPLOAD_FOLDER, fname)
            f.save(path)
            saved.append(path)
    return saved

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/review", methods=["POST"])
def review():
    job_id  = uuid.uuid4().hex
    q_paths = save_uploads(request.files.getlist("question_paper"), "q", job_id)
    s_paths = save_uploads(request.files.getlist("solution"),       "s", job_id)

    if not q_paths:
        return jsonify({"error": "Please upload at least one question paper PDF"}), 400
    if not s_paths:
        return jsonify({"error": "Please upload at least one model solution PDF"}), 400

    # Collect students: name_1, sheets_1[], name_2, sheets_2[], ...
    students = []
    i = 1
    while True:
        name  = request.form.get(f"student_name_{i}")
        files = request.files.getlist(f"student_files_{i}")
        if name is None:
            break
        paths = save_uploads(files, f"stu{i}", job_id)
        if paths:
            students.append((name or f"Student_{i}", paths))
        i += 1

    if not students:
        return jsonify({"error": "Please add at least one student answer sheet"}), 400

    with jobs_lock:
        jobs[job_id] = {"status": "running", "log": [], "report_path": None}

    threading.Thread(target=run_job,
                     args=(job_id, q_paths, s_paths, students),
                     daemon=True).start()

    return jsonify({"job_id": job_id})

@app.route("/status/<job_id>")
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "status":      job["status"],
        "log":         job["log"],
        "report_name": job.get("report_name"),
        "error":       job.get("error"),
    })

@app.route("/download/<job_id>")
def download(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return "Report not ready", 404
    return send_file(job["report_path"],
                     as_attachment=True,
                     download_name=job.get("report_name","review_report.pdf"))

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
