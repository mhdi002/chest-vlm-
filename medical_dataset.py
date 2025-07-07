# datasets/medical_dataset.py

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import re
import torchvision.transforms as transforms

class MedicalReportDataset(Dataset):
    def __init__(self, images_dir, reports_dir, tokenizer, max_length=256, img_size=224):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.data = []
        images = list(Path(images_dir).glob("*.jpg"))
        for img_path in images:
            txt_path = Path(reports_dir) / (img_path.stem + ".txt")
            if txt_path.exists():
                findings = self.extract_findings(txt_path)
                if findings:
                    self.data.append({'image_path': img_path, 'findings': findings})

    def extract_findings(self, path):
        text = path.read_text(encoding='utf-8')
        match = re.search(r'Findings?\s*:(.*?)(?=\n\s*[A-Z][a-z]*\s*:|$)', text, re.DOTALL | re.IGNORECASE)
        findings = match.group(1).strip() if match else text.strip().split("Impression:")[0].strip()
        return findings if findings else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image = self.transform(Image.open(entry['image_path']).convert("RGB"))
        tokens = self.tokenizer(entry['findings'], truncation=True, padding='max_length',
                                max_length=self.max_length, return_tensors='pt')
        return {
            'image': image,
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze()
        }
