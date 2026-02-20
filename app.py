import base64
import io
import os
import re
import json
from datetime import datetime, date
from decimal import Decimal

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, render_template, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename

import pdfplumber
from sqlalchemy import create_engine, Column, Integer, String, Date, Numeric, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from dateutil import parser as dateparser

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ---------- Config ----------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}
# Default to data/ directory, can be overridden via DB_URL environment variable
DB_URL = os.environ.get("DB_URL", "sqlite:///data/delivery_orders.db")
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB limit

# Gemini API Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
USE_GEMINI = GEMINI_AVAILABLE and GEMINI_API_KEY is not None
if USE_GEMINI:
    genai.configure(api_key=GEMINI_API_KEY)

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("data", exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# Add cache control headers to prevent browser caching
@app.after_request
def add_cache_control(response):
    if request.endpoint and 'static' not in request.endpoint:
        response.cache_control.no_cache = True
        response.cache_control.no_store = True
        response.cache_control.must_revalidate = True
        response.cache_control.max_age = 0
    return response
# In production, use: app.secret_key = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

# ---------- Database (SQLAlchemy) ----------
engine = create_engine(DB_URL, echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)


class DeliveryOrder(Base):
    __tablename__ = "delivery_orders"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    delivery_to = Column(String(500))
    delivery_order_no = Column(String(100))
    date = Column(Date)
    description_size = Column(Text)  # multi-line Description & Size
    total_weight = Column(Numeric(14, 3))
    raw_text = Column(Text)


Base.metadata.create_all(bind=engine)


# ---------- Helpers ----------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(filepath):
    """Extracts text from PDF using pdfplumber and returns a single string."""
    texts = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                texts.append(txt)
    return "\n".join(texts)


def extract_text_from_pdf_with_gemini_vision(filepath):
    """
    Fallback for image-based or scanned PDFs: convert pages to images and use
    Gemini Vision to extract text. Returns (extracted_text, error_message).
    error_message is None on success, or a string describing the failure.
    """
    if not USE_GEMINI:
        return "", "GEMINI_API_KEY is not set."
    try:
        from pdf2image import convert_from_path
    except ImportError:
        return "", "pdf2image not installed; install poppler-utils in Docker/OS."
    try:
        images = convert_from_path(filepath, dpi=150)
    except Exception as e:
        return "", f"Could not convert PDF to images: {e!s}"
    if not images:
        return "", "PDF produced no pages."
    # Use a vision-capable model (current API model IDs; 1.5-flash is deprecated)
    vision_models = ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-flash-lite", "gemini-3-flash-preview")
    model = None
    for name in vision_models:
        try:
            model = genai.GenerativeModel(name)
            break
        except Exception:
            continue
    if model is None:
        return "", "No supported Gemini vision model available. Try updating google-generativeai."
    prompt = "Extract all text from this document page. Return only the raw text, preserving line breaks. No explanation."
    texts = []
    last_error = None
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        part = {"inline_data": {"mime_type": "image/png", "data": b64}}
        try:
            response = model.generate_content([part, prompt])
            if response and response.text:
                texts.append(response.text.strip())
        except Exception as e:
            last_error = str(e)
            continue
    if texts:
        return "\n".join(texts), None
    return "", last_error or "Gemini Vision returned no text for any page."


def parse_delivery_order_with_gemini(text):
    """
    Uses Gemini API to parse delivery order text and extract structured data.
    Returns dict with keys: delivery_to, delivery_order_no, date, description_size, total_weight.
    """
    if not USE_GEMINI:
        return None

    model = None
    for name in ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-pro"):
        try:
            model = genai.GenerativeModel(name)
            break
        except Exception:
            continue
    if model is None:
        return None

    try:
        prompt = f"""You are an expert at extracting structured data from delivery order / DO documents.
Analyze the following text and extract ONLY these fields. Return ONLY a valid JSON object.
Use null for missing values. Dates in YYYY-MM-DD. Total weight as numeric (e.g. 23471.880).

Fields to extract:
- delivery_to: The delivery destination/site. When the document has "GOODS TO SITE" use that line and the following phone line (e.g. "GOODS TO SITE - J.BAHRU" and "TEL: 3542616"). Join with newline. Do NOT use the company name/address under "Deliver To" for this field.
- delivery_order_no: Delivery Order number (e.g. 10113848)
- date: Delivery order date (the date next to "Delivery Order No" or "Date" on the DO)
- description_size: All line items under "Description & Size" - include full product descriptions and sizes (e.g. RECTANGULAR HOLLOW SECTION...). Join multiple lines with newlines. Include any handwritten codes if visible in text.
- total_weight: Total weight in kg (numeric, e.g. 23471.880)

Document text:
{text[:30000]}

Return ONLY the JSON object, no other text:"""

        response = model.generate_content(prompt)
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        parsed_data = json.loads(response_text)
        result = {
            "delivery_to": parsed_data.get("delivery_to"),
            "delivery_order_no": parsed_data.get("delivery_order_no"),
            "date": find_date(str(parsed_data.get("date"))) if parsed_data.get("date") else None,
            "description_size": parsed_data.get("description_size"),
            "total_weight": safe_decimal_from_str(str(parsed_data.get("total_weight"))) if parsed_data.get("total_weight") is not None else None,
        }
        return result
    except (json.JSONDecodeError, Exception) as e:
        print(f"Gemini delivery order parse error: {e}")
        return None


def safe_decimal_from_str(s):
    """Try to get Decimal from string like 'RM 123.45' or '123.45'"""
    if not s:
        return None
    # Preserve negative sign, remove currency symbols and commas
    is_negative = s.strip().startswith("-")
    s = re.sub(r"[^\d\.]", "", s)
    if not s:
        return None
    try:
        value = Decimal(s)
        return -value if is_negative else value
    except Exception:
        return None


def find_date(text):
    """Try parsing a date-like string using dateutil. Returns date or None."""
    try:
        dt = dateparser.parse(text, dayfirst=True, fuzzy=True)
        if dt:
            return dt.date()
    except Exception:
        return None


def parse_delivery_order(text):
    """
    Heuristic parsing of delivery order text.
    Extracts: delivery_to, delivery_order_no, date, description_size, total_weight.
    """
    result = {
        "delivery_to": None,
        "delivery_order_no": None,
        "date": None,
        "description_size": None,
        "total_weight": None,
    }
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    t = re.sub(r"\s+", " ", text)

    # Delivery Order No (e.g. "Delivery Order No." / "Delivery Order No" then number)
    m = re.search(r"Delivery\s+Order\s+No\.?\s*[:\-]?\s*(\d+)", t, re.IGNORECASE)
    if m:
        result["delivery_order_no"] = m.group(1).strip()

    # Date next to DO (often right after "Date" near Delivery Order)
    m = re.search(r"(?:Delivery\s+Order\s+No\.?[^\d]*\d+[^\d]*)?Date\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", t, re.IGNORECASE)
    if m:
        result["date"] = find_date(m.group(1))
    if not result["date"]:
        m = re.search(r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", t)
        if m:
            result["date"] = find_date(m.group(1))

    # Delivery To: prefer "GOODS TO SITE" + TEL (destination/site), else "Deliver To" address block
    goods_to_site = re.search(r"GOODS\s+TO\s+SITE\s*[:\-]?\s*([^\n]+)(?:\s*\n\s*TEL\s*[:\-]?\s*([^\n]+))?", text, re.IGNORECASE)
    if goods_to_site:
        site_line = goods_to_site.group(1).strip()
        tel_part = goods_to_site.group(2).strip() if goods_to_site.group(2) else ""
        if tel_part:
            tel_line = tel_part if re.match(r"^TEL\s*:", tel_part, re.IGNORECASE) else "TEL: " + tel_part
            result["delivery_to"] = (site_line + "\n" + tel_line)[:500]
        else:
            result["delivery_to"] = site_line[:500]
    if not result["delivery_to"]:
        deliver_to_start = re.search(r"Deliver(?:y)?\s+To\s*[:\-]?\s*", text, re.IGNORECASE)
        if deliver_to_start:
            rest = text[deliver_to_start.end():]
            end = re.search(r"\n\s*(?:Delivery\s+Order\s+No|Goods\s+to\s+Site|Delivery\s+Instruction|Ref\.|Order\s+Date)", rest, re.IGNORECASE)
            addr = rest[:end.start()].strip() if end else rest[:600].strip()
            addr = re.sub(r"\s+", " ", addr).strip()
            if addr:
                result["delivery_to"] = addr[:500]

    # Total Weight (label "Total Weight" with numeric value, often at bottom)
    m = re.search(r"Total\s+Weight\s*[:\-]?\s*([0-9,]+(?:\.[0-9]+)?)", t, re.IGNORECASE)
    if m:
        result["total_weight"] = safe_decimal_from_str(m.group(1))

    # Description & Size: look for table header "Description & Size" then capture content
    desc_match = re.search(r"Description\s*&\s*Size\s*([\s\S]*?)(?=Pieces|Weight\s*\(kgs\)|Item\s+Number|$)", text, re.IGNORECASE)
    if desc_match:
        block = desc_match.group(1).strip()
        # Clean and limit size; keep newlines for multi-line items
        block = re.sub(r"\s+", " ", block)
        result["description_size"] = block[:4000].strip() if block else None
    # Fallback: lines containing "RECTANGULAR HOLLOW" or "SECTION" as product lines
    if not result["description_size"]:
        product_lines = [ln for ln in lines if re.search(r"(?:RECTANGULAR|HOLLOW|SECTION|MM\s*X\s*\d)", ln, re.IGNORECASE)]
        if product_lines:
            result["description_size"] = "\n".join(product_lines)[:4000]

    return result


# ---------- Routes ----------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


def _wants_json():
    return request.headers.get("X-Requested-With") == "XMLHttpRequest"


@app.route("/result", methods=["GET"])
def result_page():
    """Show the result page from session (used after AJAX upload)."""
    parsed = session.get("parsed_data")
    if not parsed:
        flash("No data to review. Please upload a PDF first.")
        return redirect(url_for("index"))
    filename = session.get("filename", "")
    raw_text_preview = session.get("raw_text_preview", "")
    # display_parsed: ensure all values are strings for the form
    display_parsed = {k: (v if v is not None else "") for k, v in parsed.items()}
    return render_template("result.html", parsed=display_parsed, filename=filename, raw_text_preview=raw_text_preview)


@app.route("/upload", methods=["POST"])
def upload():
    ajax = _wants_json()
    if "file" not in request.files:
        flash("No file part")
        if ajax:
            return jsonify({"success": False, "message": "No file part"})
        return redirect(url_for("index"))
    file = request.files["file"]
    if file.filename == "":
        flash("No selected file")
        if ajax:
            return jsonify({"success": False, "message": "No selected file"})
        return redirect(url_for("index"))
    if not allowed_file(file.filename):
        flash("Allowed file types: pdf")
        if ajax:
            return jsonify({"success": False, "message": "Allowed file types: pdf"})
        return redirect(url_for("index"))
    
    try:
        filename = secure_filename(file.filename)
        # Handle duplicate filenames by adding timestamp
        base, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base}_{timestamp}{ext}"
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)

        # Extract text (pdfplumber first; if empty, try Gemini Vision for image-based PDFs)
        try:
            extracted_text = extract_text_from_pdf(path)
            if not extracted_text or len(extracted_text.strip()) == 0:
                gemini_error = None  # set by Gemini Vision path if it runs and fails
                if USE_GEMINI:
                    extracted_text, gemini_error = extract_text_from_pdf_with_gemini_vision(path)
                    if extracted_text and len(extracted_text.strip()) > 0:
                        flash("Text extracted from image-based PDF using Gemini Vision.", "info")
                if not extracted_text or len(extracted_text.strip()) == 0:
                    msg = "Could not extract text from PDF. The file might be corrupted or image-based."
                    if gemini_error:
                        msg += f" Gemini Vision: {gemini_error}"
                    elif not USE_GEMINI:
                        msg += " For scanned PDFs, set GEMINI_API_KEY in .env and restart the app."
                    flash(msg)
                    os.remove(path)  # Clean up
                    if ajax:
                        return jsonify({"success": False, "message": msg})
                    return redirect(url_for("index"))
        except Exception as e:
            flash(f"Error extracting text from PDF: {str(e)}")
            if os.path.exists(path):
                os.remove(path)  # Clean up
            if ajax:
                return jsonify({"success": False, "message": str(e)})
            return redirect(url_for("index"))

        # Parse fields - try Gemini API first, fallback to regex parsing
        parsed = None
        if USE_GEMINI:
            try:
                parsed = parse_delivery_order_with_gemini(extracted_text)
                if parsed:
                    flash("Data extracted using Gemini AI", "info")
            except Exception as e:
                flash(f"Gemini API error: {str(e)}. Falling back to regex parsing.", "warning")
                parsed = None

        if not parsed:
            parsed = parse_delivery_order(extracted_text)
            if USE_GEMINI:
                flash("Used regex parsing (Gemini unavailable or failed)", "info")

        # Store parsed data and filename in session for review before saving
        # Convert dates and decimals to strings for JSON serialization
        session_data = {}
        for key, value in parsed.items():
            if value is None:
                session_data[key] = None
            elif isinstance(value, date):
                session_data[key] = value.isoformat() if value else None
            elif isinstance(value, Decimal):
                session_data[key] = str(value)
            else:
                session_data[key] = value
        
        session['parsed_data'] = session_data
        session['filename'] = filename
        # Store raw_text in session but limit size to avoid cookie size limits
        # Flask sessions use cookies by default (max ~4KB), so we'll store a smaller preview
        session['raw_text'] = extracted_text[:5000]  # Reduced size to avoid session cookie limits
        session['raw_text_preview'] = extracted_text[:2000]

        # Create display-friendly version for template (convert dates and decimals to strings)
        display_parsed = {}
        for key, value in parsed.items():
            if value is None:
                display_parsed[key] = ''
            elif isinstance(value, date):
                display_parsed[key] = value.isoformat() if value else ''
            elif isinstance(value, Decimal):
                display_parsed[key] = str(value)
            else:
                display_parsed[key] = str(value) if value else ''

        # Show to user for review before saving
        if ajax:
            return jsonify({"success": True, "redirect": url_for("result_page")})
        return render_template("result.html", parsed=display_parsed, filename=filename, raw_text_preview=extracted_text[:2000])
    
    except Exception as e:
        flash(f"Unexpected error: {str(e)}")
        if ajax:
            return jsonify({"success": False, "message": str(e)})
        return redirect(url_for("index"))


@app.route("/admin")
def admin_page():
    db = SessionLocal()
    try:
        orders = db.query(DeliveryOrder).order_by(DeliveryOrder.id.desc()).all()
    except Exception as e:
        flash(f"Error loading orders: {str(e)}")
        orders = []
    finally:
        db.close()

    return render_template("admin.html", orders=orders)


@app.route("/save", methods=["POST"])
def save_bill():
    """Save the reviewed parsed delivery order data to the database."""
    if 'parsed_data' not in session or 'filename' not in session:
        flash("No data to save. Please upload a PDF first.")
        return redirect(url_for("index"))

    parsed = session.get('parsed_data', {})
    filename = session.get('filename')
    raw_text = session.get('raw_text', '')

    def get_form_value(key, default=None):
        value = request.form.get(key, '').strip()
        if not value or value == '—':
            return default
        return value

    def get_form_decimal(key):
        value = get_form_value(key)
        if not value:
            return None
        return safe_decimal_from_str(value)

    def get_form_date(key):
        value = get_form_value(key)
        if not value:
            return None
        return find_date(value)

    parsed = dict(parsed)
    parsed['delivery_to'] = get_form_value('delivery_to') or parsed.get('delivery_to')
    parsed['delivery_order_no'] = get_form_value('delivery_order_no') or parsed.get('delivery_order_no')
    parsed['description_size'] = get_form_value('description_size') or parsed.get('description_size')

    form_date = get_form_date('date')
    if form_date:
        parsed['date'] = form_date
    elif parsed.get('date'):
        v = parsed['date']
        if isinstance(v, str):
            try:
                parsed['date'] = datetime.fromisoformat(v).date()
            except Exception:
                parsed['date'] = find_date(v)
        else:
            parsed['date'] = v
    else:
        parsed['date'] = None

    form_weight = get_form_decimal('total_weight')
    parsed['total_weight'] = form_weight if form_weight is not None else None
    if parsed['total_weight'] is None and parsed.get('total_weight'):
        try:
            parsed['total_weight'] = Decimal(str(parsed['total_weight']))
        except Exception:
            parsed['total_weight'] = None

    db = SessionLocal()
    try:
        order = DeliveryOrder(
            filename=filename,
            delivery_to=parsed.get("delivery_to"),
            delivery_order_no=parsed.get("delivery_order_no"),
            date=parsed.get("date"),
            description_size=parsed.get("description_size"),
            total_weight=parsed.get("total_weight"),
            raw_text=raw_text,
        )
        db.add(order)
        db.commit()
        db.refresh(order)
        order_id = order.id

        session.pop('parsed_data', None)
        session.pop('filename', None)
        session.pop('raw_text', None)
        session.pop('raw_text_preview', None)

        flash(f"Delivery order saved successfully! (ID: {order_id})")
        return redirect(url_for("admin_page"))
    except Exception as e:
        db.rollback()
        flash(f"Database error: {str(e)}")
        return redirect(url_for("index"))
    finally:
        db.close()


@app.route("/admin/delete", methods=["POST"])
def delete_bills():
    order_ids = request.form.getlist("order_ids")

    if not order_ids:
        flash("No orders selected for deletion")
        return redirect(url_for("admin_page"))

    db = SessionLocal()
    try:
        deleted_count = 0
        for order_id in order_ids:
            try:
                oid = int(order_id)
                order = db.query(DeliveryOrder).filter(DeliveryOrder.id == oid).first()
                if order:
                    file_path = os.path.join(app.config["UPLOAD_FOLDER"], order.filename)
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except Exception:
                            pass
                    db.delete(order)
                    deleted_count += 1
            except (ValueError, Exception):
                continue
        db.commit()
        flash(f"Successfully deleted {deleted_count} order(s)")
    except Exception as e:
        db.rollback()
        flash(f"Error deleting orders: {str(e)}")
    finally:
        db.close()

    return redirect(url_for("admin_page"))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
