import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from stage3_sequence.encoder import HieroglyphTransformerEncoder
from stage3_sequence.utils import load_config, load_code_to_idx

class HieroglyphEncoderDataset(Dataset):
    """
    Dataset for training Stage 3 Transformer Encoder.
    Expects preprocessed sequences of Gardiner sign IDs, category IDs, and spatial features.
    """
    def __init__(self, data_samples, max_seq_len=256):
        self.samples = data_samples
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # إرجاع التنسيق المطلوب للـ Encoder forward method
        return {
            "sign_ids": torch.tensor(sample["sign_ids"], dtype=torch.long),
            "category_ids": torch.tensor(sample["category_ids"], dtype=torch.long),
            "spatials": torch.tensor(sample["spatials"], dtype=torch.float),
            "sign_types": torch.tensor(sample["sign_types"], dtype=torch.long),
            "word_boundaries": torch.tensor(sample["word_boundaries"], dtype=torch.long),
        }

def collate_fn(batch):
    """Pad sequences to the maximum length in the current batch."""
    max_len = max(len(b["sign_ids"]) for b in batch)
    
    sign_ids, category_ids, spatials, sign_types, word_boundaries, masks = [], [], [], [], [], []
    
    for b in batch:
        seq_len = len(b["sign_ids"])
        pad_len = max_len - seq_len
        
        # Padding with zeros (or appropriate index)
        sign_ids.append(torch.cat([b["sign_ids"], torch.zeros(pad_len, dtype=torch.long)]))
        category_ids.append(torch.cat([b["category_ids"], torch.zeros(pad_len, dtype=torch.long)]))
        spatials.append(torch.cat([b["spatials"], torch.zeros(pad_len, b["spatials"].shape[1], dtype=torch.float)]))
        sign_types.append(torch.cat([b["sign_types"], torch.zeros(pad_len, dtype=torch.long)]))
        word_boundaries.append(torch.cat([b["word_boundaries"], torch.zeros(pad_len, dtype=torch.long)]))
        
        # Padding mask: True for padded positions
        masks.append(torch.cat([torch.zeros(seq_len, dtype=torch.bool), torch.ones(pad_len, dtype=torch.bool)]))
        
    return {
        "sign_ids": torch.stack(sign_ids),
        "category_ids": torch.stack(category_ids),
        "spatials": torch.stack(spatials),
        "sign_types": torch.stack(sign_types),
        "word_boundaries": torch.stack(word_boundaries),
        "key_padding_mask": torch.stack(masks),
    }

def train_encoder():
    cfg = load_config()
    c3 = cfg["sequence"]
    
    class_map_path = cfg["paths"]["class_map"]
    if not os.path.exists(class_map_path):
        raise FileNotFoundError(f"Class map not found at {class_map_path}. Run training of classifier first.")
        
    code_to_idx = load_code_to_idx(class_map_path)
    vocab_size = len(code_to_idx)

    # تهيئة نموذج الـ Transformer Encoder
    model = HieroglyphTransformerEncoder(
        vocab_size=vocab_size,
        d_model=c3["d_model"],
        nhead=c3["nhead"],
        num_layers=c3["num_layers"],
        dim_feedforward=c3.get("dim_feedforward", 1024),
        dropout=c3.get("dropout", 0.1),
        max_seq_len=c3.get("max_seq_len", 256),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # بيانات تجريبية (Mock Data) أو استبدالها ببيانات التدريب الفعلية من ملفات الـ Dataset
    # يمكنك تحميل عينات حقيقية هنا من ملفات الـ JSON أو الـ Annotations الخاصة بمشروعك
    dummy_samples = [
        {
            "sign_ids": [1, 2, 3, 4],
            "category_ids": [0, 0, 1, 1],
            "spatials": [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]],
            "sign_types": [0, 1, 0, 2],
            "word_boundaries": [1, 0, 1, 0],
        }
    ] * 50  # تكرار للمحاكاة

    dataset = HieroglyphEncoderDataset(dummy_samples)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion_type = nn.CrossEntropyLoss()
    criterion_boundary = nn.CrossEntropyLoss()

    print(f"[Training] Starting Stage 3 Transformer Encoder training on {device}...")

    model.train()
    epochs = 5
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            sign_ids = batch["sign_ids"].to(device)
            category_ids = batch["category_ids"].to(device)
            spatials = batch["spatials"].to(device)
            sign_types = batch["sign_types"].to(device)
            word_boundaries = batch["word_boundaries"].to(device)
            key_padding_mask = batch["key_padding_mask"].to(device)

            optimizer.zero_grad()

            out = model(
                sign_ids=sign_ids,
                category_ids=category_ids,
                spatials=spatials,
                key_padding_mask=key_padding_mask,
            )

            # حساب الخسارة (Loss) لرأسي التصنيف وحدود الكلمات
            loss_type = criterion_type(out["sign_type"].view(-1, 4), sign_types.view(-1))
            loss_boundary = criterion_boundary(out["word_boundary"].view(-1, 2), word_boundaries.view(-1))
            
            loss = loss_type + loss_boundary
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

    # حفظ الأوزان بالشكل المطلوب تماماً ليتعرف عليها SequenceEncoderInference
    output_dir = "outputs/sequence"
    os.makedirs(output_dir, exist_ok=True)
    weights_save_path = os.path.join(output_dir, "encoder_best.pt")

    checkpoint = {
        "model_state": model.state_dict(),
        "vocab_size": vocab_size,
    }
    torch.save(checkpoint, weights_save_path)
    print(f"[Training] Success! Encoder weights saved to {weights_save_path}")

if __name__ == "__main__":
    train_encoder()