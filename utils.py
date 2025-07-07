# utils/train_utils.py

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import json
from models.vision_language_model import VisionLanguageModel, contrastive_loss
from datasets.medical_dataset import MedicalReportDataset
import torch.optim as optim
import gc

def train_model(images_dir, reports_dir, save_dir='./saved_model',
                batch_size=16, learning_rate=1e-4, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    dataset = MedicalReportDataset(images_dir, reports_dir, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = VisionLanguageModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            image_feats, text_feats = model(images, input_ids, attention_mask)
            loss = contrastive_loss(image_feats, text_feats, model.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            torch.cuda.empty_cache()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(loader):.4f}")
        scheduler.step()

        torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch+1}.pth"))

        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump({'text_model_name': 'distilbert-base-uncased', 'embed_dim': 512}, f)

    gc.collect()
    print("Training complete.")
