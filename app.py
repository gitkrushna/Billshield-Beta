import base64
import io
import json
import os
import re

import gradio as gr
import pypdfium2
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from PIL import Image, ImageOps

import classify
import store


# ---------- Setup ----------
load_dotenv()

app = FastAPI()

DEEPSEEK_MODEL = "deepseek-flash"
EMPTY_VALUES = ["", "none", "null", "not stated", "n/a", "na", "-"]


# Connect to DeepSeek only when we need it, so a missing key shows a message instead of crashing the app
def get_deepseek_client():
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        raise Exception("The DeepSeek API key is not set (DEEPSEEK_API_KEY).")
    return OpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com",
        timeout=45,         # give up after 45 seconds instead of hanging
        max_retries=1,
    )


# Turn messy values (None, "not stated", numbers, lists) into clean text. Empty means "not found".
def clean_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        value = ", ".join(clean_text(item) for item in value if clean_text(item))
    value = str(value).strip()
    if value.lower() in EMPTY_VALUES:
        return ""
    return value


# ---------- File helpers ----------

# Why files arrive as text: on Vercel, every click can be handled by a different server copy,
# and each copy has its own /tmp. So we never save uploads on the server. The browser reads the
# file (photos are shrunk first), turns it into text, and sends it in the SAME request that reads it.
MAX_FILE_MB = 4     # Vercel refuses requests over 4.5 MB. The browser checks this first; this is the safety net.
MAX_PDF_MB = 3      # PDFs are sent as they are, and turning them into text makes them 1/3 bigger (3 MB -> 4 MB)


# The browser sends a chosen file as {"name": "label.png", "data": "data:image/jpeg;base64,...."}.
# Turn it back into the file's bytes.
def file_from_browser_to_bytes(file_from_browser):
    file_as_text = file_from_browser.get("data") or ""
    file_as_text = file_as_text.split(",", 1)[-1]           # drop the "data:image/jpeg;base64," start
    if len(file_as_text) * 3 / 4 > MAX_FILE_MB * 1024 * 1024:
        raise Exception("The file is over " + str(MAX_FILE_MB) + " MB.")
    return base64.b64decode(file_as_text)


# A photo opens directly. A PDF: take page 1 and turn it into an image.
def open_file_as_image(file_bytes):
    if file_bytes.startswith(b"%PDF"):
        pdf = pypdfium2.PdfDocument(file_bytes)
        if len(pdf) == 0:
            raise Exception("This PDF has no pages.")
        image = pdf[0].render(scale=150 / 72).to_pil()
        pdf.close()
        return image
    image = Image.open(io.BytesIO(file_bytes))
    return ImageOps.exif_transpose(image)       # turn phone photos the right way up


# Shrink the image and turn it into text that DeepSeek can receive
def convert_file_to_text(file_bytes):
    image = open_file_as_image(file_bytes)
    image = image.convert("RGB")
    image.thumbnail((1600, 1600))

    image_buffer = io.BytesIO()
    image.save(image_buffer, format="JPEG", quality=85)
    image_as_text = base64.b64encode(image_buffer.getvalue()).decode("utf-8")
    return "data:image/jpeg;base64," + image_as_text


# Send messages to DeepSeek and get its answer back as a Python dictionary.
# Call A uses this now; Call B (classify) will use it too.
def ask_deepseek_for_json(messages):
    deepseek_response = get_deepseek_client().chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=0,      # same input -> same answer, as far as possible
    )
    reply_text = deepseek_response.choices[0].message.content or ""

    # DeepSeek sometimes adds words around the JSON. Find the first {...} that is valid JSON.
    decoder = json.JSONDecoder()
    position = reply_text.find("{")
    while position != -1:
        try:
            answer, _ = decoder.raw_decode(reply_text[position:])
            if isinstance(answer, dict):
                return answer
        except ValueError:
            pass
        position = reply_text.find("{", position + 1)
    raise Exception("DeepSeek's answer was not in the expected format.")


def ask_deepseek_about_file(file_bytes, instruction):
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": instruction},
            {"type": "image_url", "image_url": {"url": convert_file_to_text(file_bytes)}},
        ],
    }
    return ask_deepseek_for_json([message])


# Remove anything that looks like a tax code or tax rate from product facts.
# The classifier must never see what code someone else chose (blind check).
def remove_tax_codes(text):
    text = re.sub(r"(?i)\b(HSN|SAC|tariff|heading)\b(\s*(code|no\.?|number))?\W*\d[\d .]{3,}", "[removed]", text)
    text = re.sub(r"(?i)\b[ICS]?GST\b\W*\d+(\.\d+)?\s*%", "[removed]", text)
    return text


# ---------- Call A: read the label and the bill ----------

LABEL_INSTRUCTION = """
Read this product label. Reply with JSON only, using exactly these keys:
{
  "product_name": "",
  "ingredients": ["name and % for each, e.g. Seaweed extract 30%"],
  "form": "liquid, powder, granules, machine, etc.",
  "intended_use": "",
  "registration": "any registration or licence mentioned",
  "other_claims": ["each other important claim as a short phrase, e.g. 99% purity"]
}
Write "not stated" if something is not on the label.
ingredients: ONLY what the label prints as an ingredient or composition list, copied exactly. A purity or quality
claim (e.g. "99% purity") is NOT an ingredient list: put it in other_claims and write "not stated" for ingredients.
Do not include any HSN code, tax rate or price.
"""

BILL_INSTRUCTION = """
Read this bill or invoice. Reply with JSON only, using exactly these keys:
{
  "product_name": "the first product line on the bill",
  "hsn_code": "digits only",
  "cgst_percent": "number only",
  "sgst_percent": "number only",
  "igst_percent": "number only",
  "bill_date": "as printed",
  "number_of_products": "how many product lines the bill has"
}
Write "not stated" if something is not on the bill.
"""

LABEL_SECTION_HEADER = "FROM THE LABEL PHOTO (check and fix if anything is wrong)"


# Label photo -> what goes in the Product details box, and a message.
# (It returns a list, so both answers can wait together in one hidden box until they go on the page.)
def read_label(label_photo, current_details):
    current_details = current_details or ""
    if not label_photo:
        return [current_details, ""]
    try:
        label = ask_deepseek_about_file(file_from_browser_to_bytes(label_photo), LABEL_INSTRUCTION)

        # One fact per line, so the user can check and fix it at a glance. Claims become a bullet list.
        raw_claims = label.get("other_claims")
        if not isinstance(raw_claims, list):
            raw_claims = re.split(r";|\n", clean_text(raw_claims))
        claims = [clean_text(claim).strip(" .") for claim in raw_claims if clean_text(claim).strip(" .")]
        label_lines = [
            LABEL_SECTION_HEADER,
            "Product: " + (clean_text(label.get("product_name")) or "not stated"),
            "Ingredients: " + (clean_text(label.get("ingredients")) or "not stated on the label"),
            "Form: " + (clean_text(label.get("form")) or "not stated"),
            "Used for: " + (clean_text(label.get("intended_use")) or "not stated"),
            "Registration: " + (clean_text(label.get("registration")) or "not stated"),
            "Label claims:" + ("" if claims else " none"),
        ] + ["  • " + claim for claim in claims[:10]]
        details_from_label = remove_tax_codes("\n".join(label_lines))

        # Keep what the user typed. If an earlier label was read, replace that part instead of adding it twice.
        typed_by_user = current_details.split(LABEL_SECTION_HEADER)[0].strip()
        if typed_by_user:
            new_details = typed_by_user + "\n\n" + details_from_label
        else:
            new_details = details_from_label

        return [new_details, "✅ Product details filled in from the label. Check them, then click **Check GST code**."]
    except Exception as error:
        return [current_details, "❌ Couldn't read that label. Try a clearer JPG or PNG, or type the details yourself. (" + str(error) + ")"]


# Bill photo or PDF -> what goes in the HSN code and GST % boxes, and a message.
# It never writes into Product details, so nothing from the bill reaches the classifier.
def read_bill(bill_photo, current_hsn_code, current_gst_rate):
    if not bill_photo:
        return [current_hsn_code, current_gst_rate, ""]
    try:
        bill = ask_deepseek_about_file(file_from_browser_to_bytes(bill_photo), BILL_INSTRUCTION)

        hsn_code = re.sub(r"\D", "", clean_text(bill.get("hsn_code")))     # keep digits only

        gst_rate = ""
        try:
            igst = float(clean_text(bill.get("igst_percent")) or 0)
            cgst = float(clean_text(bill.get("cgst_percent")) or 0)
            sgst = float(clean_text(bill.get("sgst_percent")) or 0)
            total = igst if igst > 0 else cgst + sgst                   # add the parts in code, not in the LLM
            if total > 0:
                gst_rate = str(int(total)) if total == int(total) else str(total)
        except ValueError:
            pass

        if not hsn_code and not gst_rate:
            return [current_hsn_code, current_gst_rate, "⚠️ Couldn't find an HSN code or GST % on that bill. Type them yourself if you have them."]

        notes = []
        product_on_bill = clean_text(bill.get("product_name"))
        if product_on_bill:
            notes.append("Product on the bill: **" + remove_tax_codes(product_on_bill) + "**.")
        if clean_text(bill.get("number_of_products")) not in ["", "1"]:
            notes.append("This bill has several products; we read the first one. Fix the boxes if that's the wrong one.")
        bill_date = clean_text(bill.get("bill_date"))
        if bill_date:
            notes.append("Bill date: " + bill_date + ". Rates changed on 22 Sep 2025; older bills may follow the old rates.")

        message = "✅ HSN code and GST % filled in from the bill. Check them, then click **Check GST code**."
        if notes:
            message = message + "\n\n" + " ".join(notes)

        # Keep what the user typed if the bill didn't show it
        return [hsn_code or current_hsn_code, gst_rate or current_gst_rate, message]
    except Exception as error:
        return [current_hsn_code, current_gst_rate, "❌ Couldn't read that bill. Try a clearer JPG, PNG or PDF, or type the HSN code and GST % yourself. (" + str(error) + ")"]


# Each photo box has its own "busy" flag, so two reads at the same moment can't overwrite each other's count.
# Check is greyed out while either box is busy, and only comes back when both are done.
def start_reading():
    return gr.update(interactive=False), "⏳ Reading…", True


# The read remembers the "Clear all" count from when it started, so its answer can be thrown away if
# "Clear all" was clicked while it was reading.
def read_label_now(label_photo, current_details, clear_count):
    return read_label(label_photo, current_details) + [clear_count]


def read_bill_now(bill_photo, current_hsn_code, current_gst_rate, clear_count):
    return read_bill(bill_photo, current_hsn_code, current_gst_rate) + [clear_count]


def answer_is_old(answer, clear_count):
    return not answer or answer[-1] != clear_count


# After a read: put the answer on the page, mark this box done, and switch Check back on if the other box is done too
def finish_label_reading(label_answer, bill_busy, clear_count):
    if answer_is_old(label_answer, clear_count):
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    new_details, message, _ = label_answer
    return new_details, message, gr.update(interactive=not bill_busy), False


def finish_bill_reading(bill_answer, label_busy, clear_count):
    if answer_is_old(bill_answer, clear_count):
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
    hsn_code, gst_rate, message, _ = bill_answer
    return hsn_code, gst_rate, message, gr.update(interactive=not label_busy), False


# The reading request itself failed (for example the server took too long): say so and switch Check back on
def reading_failed(this_box_busy, other_box_busy):
    if not this_box_busy:      # "Clear all" was clicked meanwhile: stay quiet
        return gr.skip(), gr.skip(), gr.skip()
    message = "❌ Couldn't read that file. Please try again, or type the details yourself."
    return message, gr.update(interactive=not other_box_busy), False


# Empties a photo box: no file, no problem message
def empty_file_box():
    return gr.update(value=None, problem="")


RESULT_PLACEHOLDER = ("*The correct HSN code and GST rate appear here, with the reason and what to do next. "
                      "If you added a bill, you'll also see whether it matches.*")


# Fill the form with a ready-made example so first-time users can try it instantly
def load_example():
    example_details = (
        "Product: SeaGrow seaweed bio-stimulant, 1 litre\n"
        "Ingredients: Seaweed extract (Ascophyllum nodosum) 30%, potassium 5%\n"
        "Form: liquid\n"
        "Use: sprayed on crops to improve growth and stress tolerance\n"
        "Registration: Bio-stimulant, FCO Schedule VI\n"
        "Other: Pre-packed and labelled 1 litre bottle. Contains no plant growth regulators"
    )
    return (empty_file_box(), empty_file_box(), example_details, "3808", "18", RESULT_PLACEHOLDER,
            "Example filled in. Click **Check GST code**.", "")


# Clear every box and switch Check back on. Counting the clears throws away answers still on their way.
def clear_all(clear_count):
    return (empty_file_box(), empty_file_box(), "", "", None, RESULT_PLACEHOLDER, "", "",
            gr.update(interactive=True), False, False, (clear_count or 0) + 1)


# ---------- Check button ----------
def check_bill(product_details, bill_hsn_code, bill_gst_rate, user_name=""):
    product_details = (product_details or "").strip()
    if not product_details:
        return "Add the product details first. Type them, or upload a label photo to fill them in."

    # The classifier will only ever see the product details, with any tax codes removed (blind check)
    facts_for_classifier = remove_tax_codes(product_details)
    record = classify.run_check(facts_for_classifier, bill_hsn_code, bill_gst_rate, ask_deepseek_for_json, DEEPSEEK_MODEL)
    record["user_name"] = (user_name or "").strip() or None
    not_saved = store.save_check(record)
    return classify.render(record) + ("\n\n_(" + not_saved + ")_" if not_saved else "")


# "Show my checks": the last 20 checks saved under the typed name
def show_my_checks(user_name):
    rows, message = store.my_checks(user_name)
    return gr.update(value=rows, visible=bool(rows)), message


# ---------- Photo boxes (label photo, bill photo or PDF) ----------
# These boxes never send the file to the server by itself. When a file is chosen, a little JavaScript in the
# browser checks it, shrinks photos (longest side 1600 px, JPEG), turns the file into text and keeps it
# in the box. Then it fires the box's "upload" event, and the Python code below gets the file inside
# the same request that reads it. Nothing is saved on the server, so it works on Vercel.
FILE_BOX_HTML = """
<div class="file-box">
  <span class="file-box-title">${box_title}</span>
  <label class="file-box-drop">
    <input type="file" accept="${accept}">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
         stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
         <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
    <span>${value && value.name ? "📄 " + value.name : "Drop File Here"}</span>
    <span class="file-box-small">${value && value.name ? "Click to choose another file" : "- or -"}</span>
    <span>${value && value.name ? "" : "Click to Upload"}</span>
    <span class="file-box-problem">${problem}</span>
  </label>
</div>
"""

FILE_BOX_STYLE = """
.file-box {display: flex; flex-direction: column; height: 150px; border: 1px dashed var(--block-border-color);
           border-radius: var(--block-radius); background: var(--block-background-fill); overflow: hidden;}
.file-box-title {align-self: flex-start; padding: var(--block-label-padding); font-size: var(--block-label-text-size);
                 color: var(--block-label-text-color); border-right: 1px solid var(--block-border-color);
                 border-bottom: 1px solid var(--block-border-color); border-bottom-right-radius: var(--block-radius);}
.file-box-drop {flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 2px;
                padding: 6px; text-align: center; cursor: pointer; overflow-wrap: anywhere;
                color: var(--body-text-color-subdued); font-size: var(--text-lg); line-height: 1.45;}
.file-box-drop input {position: absolute; width: 1px; height: 1px; opacity: 0;}
.file-box-small {font-size: var(--text-md); opacity: 0.8;}
.file-box-problem {font-size: 13px; color: var(--error-text-color, #dc2626);}
"""

FILE_BOX_JS = """
const MB = 1024 * 1024;

// Read a file as text: "data:application/pdf;base64,...."
function readAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

// Shrink a photo so its longest side is at most 1600 px, and turn it into JPEG text
async function shrinkPhoto(file) {
  const photo = await createImageBitmap(file);      // the browser turns phone photos the right way up
  const scale = Math.min(1, 1600 / Math.max(photo.width, photo.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(photo.width * scale);
  canvas.height = Math.round(photo.height * scale);
  const pen = canvas.getContext("2d");
  pen.fillStyle = "white";                           // see-through PNGs get a white background
  pen.fillRect(0, 0, canvas.width, canvas.height);
  pen.drawImage(photo, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.85);
}

async function useFile(file) {
  if (!file) return;
  props.problem = "";
  const name = file.name.toLowerCase();
  const isPdf = name.endsWith(".pdf");
  if (!props.accept.split(",").some(ending => name.endsWith(ending))) {
    props.problem = "Please choose a " + props.accept.replaceAll(",", " ") + " file.";
    return;
  }
  const limitMb = isPdf ? props.max_pdf_mb : props.max_photo_mb;
  if (file.size > limitMb * MB) {
    props.problem = "This file is over " + limitMb + " MB. Please choose a smaller one.";
    return;
  }
  try {
    const data = isPdf ? await readAsText(file) : await shrinkPhoto(file);
    props.value = {name: file.name.replace(/[<>&"']/g, ""), data: data};
    trigger("upload");                               // runs the Python code for this box
  } catch (error) {
    props.problem = "Couldn't open that file. Try a JPG or PNG.";
  }
}

element.addEventListener("change", event => {
  if (event.target.type !== "file") return;
  useFile(event.target.files[0]);
  event.target.value = "";                           // so choosing the same file again still works
});
element.addEventListener("dragover", event => event.preventDefault());
element.addEventListener("drop", event => {
  event.preventDefault();
  useFile(event.dataTransfer.files[0]);
});
"""


def file_box(box_title, file_types):
    return gr.HTML(
        value=None,
        html_template=FILE_BOX_HTML,
        css_template=FILE_BOX_STYLE,
        js_on_load=FILE_BOX_JS,
        box_title=box_title,
        accept=",".join(file_types),
        problem="",
        max_photo_mb=MAX_FILE_MB,
        max_pdf_mb=MAX_PDF_MB,
        elem_classes="file-box-holder",
    )


# ---------- The page ----------
PAGE_STYLE = """
.gradio-container {max-width: 1100px !important; margin: auto !important;}
.file-box-holder .html-container {padding: 0 !important;}
#upload-hint, #upload-hint p {font-size: 13px; opacity: 0.75; margin-top: -4px;}
#answer table {width: 100%;}
.page-title {font-size: 2em; font-weight: 700; margin: 0;}
.gst-word {color: #E50914;}
@media (max-width: 700px) { #answer {min-height: 260px !important;} }
"""

with gr.Blocks(title="BillShield: GST Check") as page:
    # Hidden boxes. They live in the browser (not on the server), so every request carries them.
    label_busy = gr.Checkbox(False, visible=False)     # the label photo is being read
    bill_busy = gr.Checkbox(False, visible=False)      # the bill is being read
    clear_count = gr.Number(0, precision=0, visible=False)   # how often "Clear all" was clicked
    label_answer = gr.JSON(visible=False)    # what the label reading found, until it goes on the page
    bill_answer = gr.JSON(visible=False)     # what the bill reading found, until it goes on the page

    # HTML (not Markdown) so "GST" can have its own colour: Netflix red, #E50914
    gr.HTML('<h1 class="page-title">BillShield: <span class="gst-word">GST</span> Check</h1>')
    gr.Markdown(
        "Find the right HSN code and GST rate for a product, and see if a bill got it right. "
        "**All you need is the product details.** Photos just fill in the boxes for you."
    )

    with gr.Row(equal_height=True):
        # Left: what the user fills in
        with gr.Column(scale=3):
            with gr.Row(equal_height=True):
                label_photo = file_box("Label photo", [".jpg", ".jpeg", ".png", ".webp"])
                bill_photo = file_box("Bill photo or PDF", [".jpg", ".jpeg", ".png", ".webp", ".pdf"])
            gr.Markdown(
                "Label photo fills in the product details. Bill fills in the HSN code and GST %. "
                "JPG, PNG or PDF. Photos under 4 MB, PDFs under 3 MB.",
                elem_id="upload-hint",
            )

            label_status = gr.Markdown("")
            bill_status = gr.Markdown("")

            product_details_box = gr.Textbox(
                label="Product details (needed)",
                info="Type them, or upload a label photo. Say what it is, ingredients with %, form and use.",
                placeholder="e.g. Seaweed bio-stimulant, liquid. Seaweed extract 30%, potassium 5%. No plant growth regulators.",
                lines=6,
            )

            with gr.Row(equal_height=True):
                bill_hsn_box = gr.Textbox(
                    label="HSN code on the bill",
                    info="Only if you have a bill",
                    placeholder="e.g. 3808",
                )
                bill_gst_box = gr.Dropdown(
                    label="GST % on the bill",
                    info="CGST + SGST, e.g. 9 + 9 = 18",
                    choices=["0", "5", "18", "40"],
                    value=None,
                    allow_custom_value=True,
                )

            user_name_box = gr.Textbox(
                label="Your name (optional)",
                info="Only to find your checks again under My checks",
                placeholder="e.g. Anand",
            )

            check_button = gr.Button("Check GST code", variant="primary", size="lg")
            with gr.Row():
                example_button = gr.Button("Fill in an example", variant="secondary", min_width=0)
                clear_all_button = gr.Button("Clear all", variant="secondary", min_width=0)

        # Right: the answer
        with gr.Column(scale=2):
            # Markdown (not a text box), so the result can use bold, headings, bullets and a small table
            answer_box = gr.Markdown(
                RESULT_PLACEHOLDER, label="Result", show_label=True, container=True, padding=True,
                min_height=560, elem_id="answer",
            )

    # My checks: what was saved under the name above
    with gr.Accordion("My checks", open=False):
        my_checks_button = gr.Button("Show my checks", variant="secondary", size="sm")
        my_checks_message = gr.Markdown("")
        my_checks_table = gr.Dataframe(
            headers=["When (UTC)", "Product", "Bill", "Expected", "Verdict"],
            interactive=False, wrap=True, visible=False,
        )

    # Every event below is ONE plain request (queue=False): no waiting line, no live connection.
    # On Vercel each request may land on a different server copy, so nothing may depend on the
    # server remembering anything between two requests.

    # Label chosen: grey out Check -> read the label -> put the answer on the page and switch Check back on.
    # If the read request fails (e.g. Vercel time limit), reading_failed says so and switches Check back on.
    label_step_1 = label_photo.upload(
        start_reading, outputs=[check_button, label_status, label_busy], queue=False,
    )
    label_step_2 = label_step_1.then(
        read_label_now, inputs=[label_photo, product_details_box, clear_count], outputs=label_answer, queue=False,
    )
    label_step_2.success(
        finish_label_reading,
        inputs=[label_answer, bill_busy, clear_count],
        outputs=[product_details_box, label_status, check_button, label_busy],
        queue=False,
    )
    label_step_2.failure(
        reading_failed, inputs=[label_busy, bill_busy], outputs=[label_status, check_button, label_busy], queue=False,
    )

    # Bill chosen: grey out Check -> read the bill -> put the answer on the page and switch Check back on
    bill_step_1 = bill_photo.upload(
        start_reading, outputs=[check_button, bill_status, bill_busy], queue=False,
    )
    bill_step_2 = bill_step_1.then(
        read_bill_now, inputs=[bill_photo, bill_hsn_box, bill_gst_box, clear_count], outputs=bill_answer, queue=False,
    )
    bill_step_2.success(
        finish_bill_reading,
        inputs=[bill_answer, label_busy, clear_count],
        outputs=[bill_hsn_box, bill_gst_box, bill_status, check_button, bill_busy],
        queue=False,
    )
    bill_step_2.failure(
        reading_failed, inputs=[bill_busy, label_busy], outputs=[bill_status, check_button, bill_busy], queue=False,
    )

    example_button.click(
        load_example,
        outputs=[label_photo, bill_photo, product_details_box, bill_hsn_box, bill_gst_box, answer_box, label_status, bill_status],
        queue=False,
    )

    # Clear all also counts up clear_count, so answers still on their way are thrown away
    clear_all_button.click(
        clear_all,
        inputs=clear_count,
        outputs=[label_photo, bill_photo, product_details_box, bill_hsn_box, bill_gst_box,
                 answer_box, label_status, bill_status, check_button, label_busy, bill_busy, clear_count],
        queue=False,
    )

    check_button.click(
        check_bill,
        inputs=[product_details_box, bill_hsn_box, bill_gst_box, user_name_box],
        outputs=answer_box,
        queue=False,
    )

    my_checks_button.click(
        show_my_checks, inputs=user_name_box, outputs=[my_checks_table, my_checks_message], queue=False,
    )


# Put the page on the website at "/"
app = gr.mount_gradio_app(
    app,
    page,
    path="/",
    theme=gr.themes.Default(primary_hue="indigo", neutral_hue="slate"),
    css=PAGE_STYLE,
    max_file_size="4mb",
)
