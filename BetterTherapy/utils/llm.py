import torch
import json


def generate_response(prompt: str, model, tokenizer, type="validator") -> str:
    tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    input_length = input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1000,
            temperature=1.5,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(
        output_ids[0][input_length:], skip_special_tokens=True
    )
    if type == "validator":
        return parse_response(generated_text)
    return generated_text


def parse_response(text: str) -> dict:
    json_str = text.strip()

    if not json_str.startswith("{"):
        json_str = "{" + json_str
    if not json_str.endswith("}"):
        json_str = json_str + "}"

    try:
        parsed = json.loads(json_str)
        return parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
