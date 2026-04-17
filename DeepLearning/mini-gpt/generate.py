import torch
import torch.nn.functional as F


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=1.0,
    top_k=50,
    device="cpu",
):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    max_len = model.pos_enc.pe.size(1)

    for _ in range(max_new_tokens):
        context = input_ids[:, -max_len:]
        logits = model(context)
        next_logits = logits[:, -1, :] / temperature

        if top_k > 0:
            k = min(top_k, next_logits.size(-1))
            top_values, _ = torch.topk(next_logits, k)
            threshold = top_values[:, -1].unsqueeze(-1)
            next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids.squeeze(0).tolist(), skip_special_tokens=True)
