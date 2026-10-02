import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
    ).to(device)

    prompt = """
در 6 خط و با تیترهای کوتاه توضیح بده:

1) LLM چیست؟
2) Local (اجرای محلی) یعنی چه؟
3) API/Cloud یعنی چه؟
4) 3 مزیت Local و 3 مزیت API را بولت‌پوینت کن.
5) یک مثال خیلی کوتاه از کاربرد (مثلاً چت‌بات یا تولید متن).

مهم: اطلاعات غلط نده. پاسخ را فقط به فارسی بنویس.
""".strip()

    messages = [
        {"role": "system", "content": "تو یک دستیار آموزشی دقیق هستی. پاسخ کوتاه، ساختارمند و بدون اطلاعات ساختگی می‌دهی."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=220,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)

    # تلاش برای جدا کردن فقط جواب assistant (اگر قالب چت داشته باشد)
    if "assistant" in decoded:
        decoded = decoded.split("assistant", 1)[-1].strip()

    print("\n--- Assistant Output ---\n")
    print(decoded)

    # ذخیره خروجی در فایل UTF-8 (برای ارسال به استاد)
    with open("llm_output.txt", "w", encoding="utf-8") as f:
        f.write(decoded + "\n")

    print("\nSaved to: llm_output.txt")

if __name__ == "__main__":
    main()