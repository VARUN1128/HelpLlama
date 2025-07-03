# HelpLlama

# 🦙 HelpLlama: Fine-Tuned Customer Support Chatbot with TinyLlama + QLoRA + Gradio

> **Your private AI support assistant — fine-tuned, memory-efficient, and ready to serve.**

**HelpLlama** is an open-source, production-ready chatbot built using [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0), fine-tuned with **QLoRA** on real-world customer support FAQs. It runs efficiently even on **Google Colab**, and features a **Gradio-powered web interface** for instant interaction.

---

## ✨ Features

✅ Fine-tunes TinyLlama with QLoRA (4-bit quantization + LoRA adapters)
✅ Runs on consumer hardware (T4, 8GB VRAM)
✅ Instruction-tuned for real-world support conversations
✅ Full training pipeline in 14 simple steps
✅ Gradio web UI included
✅ Built with Hugging Face 🤗 + BitsAndBytes + PEFT

---

## 📸 Preview

![HelpLlama Gradio UI](https://user-images.githubusercontent.com/demo/helpllama-ui.png)
*Ask questions like a real customer. Get answers like a real support agent.*

---

## 📁 Repository Structure

```
HelpLlama/
├── train_helpllama.ipynb        # Notebook for fine-tuning with QLoRA
├── gradio_chat.py               # Gradio interface script
├── requirements.txt             # Required libraries
├── README.md
└── /tinyllama-qlora-support-bot # Saved fine-tuned model & tokenizer
```

---

## 🚀 Quick Start

### ✅ 1. Install Dependencies

```bash
pip install -q bitsandbytes accelerate datasets loralib peft transformers trl gradio
```

---

### 📦 2. Load & Prepare Dataset

```python
from datasets import load_dataset

dataset = load_dataset("MakTek/Customer_support_faqs_dataset", split="train")
```

Convert to instruction-style format:

```python
def format_instruction(example):
    return f"### Instruction:\n{example['question']}\n\n### Response:\n{example['answer']}"
dataset = dataset.map(lambda x: {"text": format_instruction(x)})
```

---

### 🧠 3. Load & Quantize Base Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
```

---

### 🪄 4. Apply LoRA (QLoRA)

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

---

### 🧷 5. Tokenize Dataset

```python
def tokenize(example):
    tokenized = tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize, batched=True)
```

---

### ⚙️ 6. Training Arguments & Trainer

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./tinyllama-qlora-support-bot",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    bf16=True,
    optim="paged_adamw_8bit",
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

---

### 💾 7. Save the Model

```python
model.save_pretrained("tinyllama-qlora-support-bot")
tokenizer.save_pretrained("tinyllama-qlora-support-bot")
```

---

### 💬 8. Test Locally

```python
from transformers import pipeline

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
prompt = "### Instruction:\nhow can I request a refund?\n\n### Response:\n"
output = pipe(prompt, max_new_tokens=100)
print(output[0]["generated_text"])
```

---

### 🌐 9. Launch Gradio UI

```python
import gradio as gr

def generate_response(instruction):
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    output = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
    return output[0]['generated_text'].replace(prompt, "").strip()

gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=3, placeholder="Ask your customer support question here..."),
    outputs=gr.Textbox(lines=6),
    title="HelpLlama – Customer Support Chatbot",
    description="Fine-tuned with QLoRA on support FAQs using TinyLlama-1.1B."
).launch()
```

---

## 🌟 Highlights

| Technique              | Purpose                             |
| ---------------------- | ----------------------------------- |
| ✅ TinyLlama 1.1B       | Small, instruction-tuned base model |
| ✅ QLoRA (4-bit)        | Memory-efficient fine-tuning        |
| ✅ LoRA                 | Parameter-efficient updates         |
| ✅ Gradio UI            | Chatbot frontend in minutes         |
| ✅ Hugging Face Trainer | Easy training orchestration         |

---

## 📈 Benchmarks (Colab T4)

| Metric            | Value          |
| ----------------- | -------------- |
| GPU Used          | T4 (16 GB)     |
| Epochs            | 3              |
| Batch Size        | 2 (+grad\_acc) |
| Training Time     | \~45 min       |
| RAM Usage (QLoRA) | \~6 GB         |

---

## 🔮 Roadmap

* [ ] Add memory for long context support
* [ ] Add Telegram/WhatsApp integration
* [ ] Convert to Hugging Face Space demo
* [ ] Fine-tune on multilingual datasets
* [ ] Add speech-to-text + TTS interface

---

## 🙋‍♂️ Author

**Varun Haridas**
📧 [varun.haridas321@gmail.com](mailto:varun.haridas321@gmail.com)

> Built with 🤍 for devs, startups, and AI builders who want control over their models.

---


