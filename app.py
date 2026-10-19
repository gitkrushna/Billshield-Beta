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


# ---------- Setup ----------
load_dotenv()

app = FastAPI()

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

# A photo opens directly. A PDF: take page 1 and turn it into an image.
def open_file_as_image(file_path):
    if file_path.lower().endswith(".pdf"):
        pdf = pypdfium2.PdfDocument(file_path)
        if len(pdf) == 0:
            raise Exception("This PDF has no pages.")
        image = pdf[0].render(scale=150 / 72).to_pil()
        pdf.close()
        return image
    image = Image.open(file_path)
    return ImageOps.exif_transpose(image)       # turn phone photos the right way up


# Shrink the image and turn it into text that DeepSeek can receive
def convert_file_to_text(file_path):
    image = open_file_as_image(file_path)
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
        model="deepseek-flash",
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


def ask_deepseek_about_file(file_path, instruction):
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": instruction},
            {"type": "image_url", "image_url": {"url": convert_file_to_text(file_path)}},
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
  "other_claims": "anything else important on the label"
}
Write "not stated" if something is not on the label.
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

LABEL_SECTION_HEADER = "--- From the label photo ---"


# Label photo -> fills the Product details box
def read_label(label_file_path, current_details):
    current_details = current_details or ""
    if label_file_path is None:
        return current_details, ""
    try:
        label = ask_deepseek_about_file(label_file_path, LABEL_INSTRUCTION)

        label_lines = [
            LABEL_SECTION_HEADER,
            "Product: " + (clean_text(label.get("product_name")) or "not stated"),
            "Ingredients: " + (clean_text(label.get("ingredients")) or "not stated"),
            "Form: " + (clean_text(label.get("form")) or "not stated"),
            "Use: " + (clean_text(label.get("intended_use")) or "not stated"),
            "Registration: " + (clean_text(label.get("registration")) or "not stated"),
            "Other: " + (clean_text(label.get("other_claims")) or "not stated"),
        ]
        details_from_label = remove_tax_codes("\n".join(label_lines))

        # Keep what the user typed. If an earlier label was read, replace that part instead of adding it twice.
        typed_by_user = current_details.split(LABEL_SECTION_HEADER)[0].strip()
        if typed_by_user:
            new_details = typed_by_user + "\n\n" + details_from_label
        else:
            new_details = details_from_label

        return new_details, "✅ Product details filled in from the label. Check them, then click **Check GST code**."
    except Exception as error:
        return current_details, "❌ Couldn't read that label. Try a clearer JPG or PNG, or type the details yourself. (" + str(error) + ")"


# Bill photo or PDF -> fills the HSN code and GST % boxes.
# It never writes into Product details, so nothing from the bill reaches the classifier.
def read_bill(bill_file_path, current_hsn_code, current_gst_rate):
    if bill_file_path is None:
        return current_hsn_code, current_gst_rate, ""
    try:
        bill = ask_deepseek_about_file(bill_file_path, BILL_INSTRUCTION)

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
            return current_hsn_code, current_gst_rate, "⚠️ Couldn't find an HSN code or GST % on that bill. Type them yourself if you have them."

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
        return hsn_code or current_hsn_code, gst_rate or current_gst_rate, message
    except Exception as error:
        return current_hsn_code, current_gst_rate, "❌ Couldn't read that bill. Try a clearer JPG, PNG or PDF, or type the HSN code and GST % yourself. (" + str(error) + ")"


# Grey out Check while anything is being read. Count the reads, so Check only comes back when all are done.
def start_reading(reads_in_progress):
    return gr.update(interactive=False), "⏳ Reading…", reads_in_progress + 1


def finish_reading(reads_in_progress):
    reads_left = max(reads_in_progress - 1, 0)
    return gr.update(interactive=(reads_left == 0)), reads_left


# Fill the form with a ready-made example so first-time users can try it instantly
def load_example():
    example_details = (
        "Product: SeaGrow seaweed bio-stimulant, 1 litre\n"
        "Ingredients: Seaweed extract (Ascophyllum nodosum) 30%, potassium 5%\n"
        "Form: liquid\n"
        "Use: sprayed on crops to improve growth and stress tolerance\n"
        "Registration: Bio-stimulant, FCO Schedule VI\n"
        "Other: Contains no plant growth regulators"
    )
    return None, None, example_details, "3808", "18", "", "Example filled in. Click **Check GST code**."


# Clear every box and switch Check back on
def clear_all():
    return None, None, "", "", None, "", "", gr.update(interactive=True), 0


# ---------- Check button ----------
def check_bill(product_details, bill_hsn_code, bill_gst_rate):
    product_details = (product_details or "").strip()
    if not product_details:
        return "Add the product details first. Type them, or upload a label photo to fill them in."

    # The classifier will only ever see the product details, with any tax codes removed
    facts_for_classifier = remove_tax_codes(product_details)

    # NEXT STEP: Call B (classify) with facts_for_classifier + the rules file,
    # then verify, confidence checklist, compare with the bill, verdict, save.
    return (
        "Ready to classify.\n\n"
        + "Product facts the classifier will see:\n" + facts_for_classifier + "\n\n"
        + "Bill HSN code (kept hidden from the classifier): " + (bill_hsn_code or "not given") + "\n"
        + "Bill GST %: " + (bill_gst_rate or "not given")
    )


# ---------- The page ----------
PAGE_STYLE = """
.gradio-container {max-width: 1100px !important; margin: auto !important;}
#upload-hint, #upload-hint p {font-size: 13px; opacity: 0.75; margin-top: -4px;}
#answer textarea {min-height: 560px;}
@media (max-width: 700px) { #answer textarea {min-height: 260px;} }
"""

with gr.Blocks(title="BillShield") as page:
    reads_in_progress = gr.State(0)

    gr.Markdown(
        "# BillShield\n"
        "Find the right HSN code and GST rate for a product, and see if a bill got it right. "
        "**All you need is the product details.** Photos just fill in the boxes for you."
    )

    with gr.Row(equal_height=True):
        # Left: what the user fills in
        with gr.Column(scale=3):
            with gr.Row(equal_height=True):
                label_upload = gr.File(
                    label="Label photo",
                    file_types=[".jpg", ".jpeg", ".png", ".webp"],
                    type="filepath",
                    height=150,
                )
                bill_upload = gr.File(
                    label="Bill photo or PDF",
                    file_types=[".jpg", ".jpeg", ".png", ".webp", ".pdf"],
                    type="filepath",
                    height=150,
                )
            gr.Markdown(
                "Label photo fills in the product details. Bill fills in the HSN code and GST %. "
                "JPG, PNG or PDF, under 4 MB.",
                elem_id="upload-hint",
            )

            status_message = gr.Markdown("")

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

            check_button = gr.Button("Check GST code", variant="primary", size="lg")
            with gr.Row():
                example_button = gr.Button("Fill in an example", variant="secondary", min_width=0)
                clear_all_button = gr.Button("Clear all", variant="secondary", min_width=0)

        # Right: the answer
        with gr.Column(scale=2):
            answer_box = gr.Textbox(
                label="Result",
                placeholder="The correct HSN code and GST rate appear here, with the reason and what to do next. "
                            "If you added a bill, you'll also see whether it matches.",
                lines=24,
                max_lines=40,
                elem_id="answer",
            )

    # Label uploaded: grey out Check -> read the label -> switch Check back on
    label_step_1 = label_upload.upload(
        start_reading, inputs=reads_in_progress, outputs=[check_button, status_message, reads_in_progress]
    )
    label_step_2 = label_step_1.then(
        read_label,
        inputs=[label_upload, product_details_box],
        outputs=[product_details_box, status_message],
    )
    label_step_3 = label_step_2.then(
        finish_reading, inputs=reads_in_progress, outputs=[check_button, reads_in_progress]
    )

    # Bill uploaded: grey out Check -> read the bill -> switch Check back on
    bill_step_1 = bill_upload.upload(
        start_reading, inputs=reads_in_progress, outputs=[check_button, status_message, reads_in_progress]
    )
    bill_step_2 = bill_step_1.then(
        read_bill,
        inputs=[bill_upload, bill_hsn_box, bill_gst_box],
        outputs=[bill_hsn_box, bill_gst_box, status_message],
    )
    bill_step_3 = bill_step_2.then(
        finish_reading, inputs=reads_in_progress, outputs=[check_button, reads_in_progress]
    )

    example_button.click(
        load_example,
        outputs=[label_upload, bill_upload, product_details_box, bill_hsn_box, bill_gst_box, answer_box, status_message],
    )

    clear_all_button.click(
        clear_all,
        outputs=[label_upload, bill_upload, product_details_box, bill_hsn_box, bill_gst_box,
                 answer_box, status_message, check_button, reads_in_progress],
        cancels=[label_step_1, label_step_2, label_step_3, bill_step_1, bill_step_2, bill_step_3],
    )

    check_button.click(
        check_bill,
        inputs=[product_details_box, bill_hsn_box, bill_gst_box],
        outputs=answer_box,
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