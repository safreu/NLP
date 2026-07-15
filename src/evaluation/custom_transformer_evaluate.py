import torch


def generate_prediction(model, src_tensor, tokenizer, device, max_length=256):
    model.eval()

    outputs = [tokenizer.cls_token_id]

    with torch.no_grad():
        for _ in range(max_length):
            trg_input = torch.tensor([outputs], dtype=torch.long, device=device)

            src_attention_mask = (src_tensor != tokenizer.pad_token_id).long()

            out = model(src_tensor, src_attention_mask, trg_input)

            best_next_item = out.argmax(dim=2)[:, -1].item()

            outputs.append(best_next_item)

            if best_next_item == tokenizer.sep_token_id:
                break

    return tokenizer.decode(outputs, skip_special_tokens=True)


def build_custom_transformer_predict_fn(model, tokenizer, device, max_length: int = 256):
    def predict_fn(sources: list[str]) -> list[str]:
        predictions = []

        for source in sources:
            src_indices = tokenizer(
                source,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )["input_ids"]

            src_tensor = torch.tensor([src_indices], dtype=torch.long, device=device)

            prediction = generate_prediction(
                model=model,
                src_tensor=src_tensor,
                tokenizer=tokenizer,
                device=device,
                max_length=max_length,
            )

            predictions.append(prediction)

        return predictions

    return predict_fn
