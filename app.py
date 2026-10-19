import gradio as gr
from fastapi import FastAPI

app = FastAPI()


def check_bill(description, bill_hsn, bill_rate):
    # Placeholder until item 2 (DeepSeek + rules)
    return f"Got it: HSN {bill_hsn} at {bill_rate}%. Check coming soon."


with gr.Blocks(title="BillShield") as demo:
    gr.Markdown("# BillShield\nDoes this product's GST code fit what it actually is?")

    description = gr.Textbox(
        label="Product description",
        placeholder="What it is, ingredients with %, form, use",
        lines=4,
    )
    bill_hsn = gr.Textbox(label="HSN code on the bill", placeholder="e.g. 2106")
    bill_rate = gr.Dropdown(
        label="GST rate on the bill (%)",
        choices=["0", "5", "18", "40"],
        allow_custom_value=True,
    )

    check_btn = gr.Button("Check")
    result = gr.Textbox(label="Result")

    check_btn.click(check_bill, inputs=[description, bill_hsn, bill_rate], outputs=result)

app = gr.mount_gradio_app(app, demo, path="/")


