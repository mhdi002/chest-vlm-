# models/vision_language_model.py

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from transformers import AutoModel

class VisionLanguageModel(nn.Module):
    def __init__(self, text_model_name='distilbert-base-uncased', embed_dim=512):
        super().__init__()
        self.vision_encoder = models.resnet50(pretrained=True)
        self.vision_encoder.fc = nn.Identity()
        vision_dim = 2048
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size

        self.vision_projection = nn.Sequential(
            nn.Linear(vision_dim, embed_dim), nn.ReLU(), nn.Dropout(0.1)
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, embed_dim), nn.ReLU(), nn.Dropout(0.1)
        )
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, images):
        features = self.vision_encoder(images)
        return self.vision_projection(features)

    def encode_text(self, input_ids, attention_mask):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = text_outputs.last_hidden_state[:, 0]
        return self.text_projection(cls)

    def forward(self, images, input_ids, attention_mask):
        image_features = nn.functional.normalize(self.encode_image(images), dim=-1)
        text_features = nn.functional.normalize(self.encode_text(input_ids, attention_mask), dim=-1)
        return image_features, text_features

def contrastive_loss(image_features, text_features, temperature):
    logits = torch.matmul(image_features, text_features.T) * temperature.exp()
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = nn.functional.cross_entropy(logits, labels)
    loss_t2i = nn.functional.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2
