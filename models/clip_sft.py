import torch
from transformers import CLIPProcessor, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "openai/clip-vit-base-patch16"
processor = CLIPProcessor.from_pretrained(model_name)
vision_model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(device)
text_model = CLIPTextModelWithProjection.from_pretrained(model_name).to(device)

class CLIPDataset(Dataset):
    def __init__(self, image_dir, json_path, processor, image_extensions=["jpg", "png", "jpeg"]):
        self.processor = processor
        
        with open(json_path, 'r') as f:
            self.text_mapping = json.load(f)
        

        self.samples = []
        for ext in image_extensions:
            image_paths = glob.glob(os.path.join(image_dir, f"*.{ext}"))
            for path in image_paths:
                filename = os.path.basename(path)
                if filename in self.text_mapping:
                    self.samples.append(path)
                else:
                    print(f"lose  {filename}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        filename = os.path.basename(img_path)
        

        image = Image.open(img_path).convert("RGB")
        text = self.text_mapping[filename]  
        
        return image, text


dataset = CLIPDataset(
    image_dir="../pheme_imagess_consistency",          
    json_path="../pheme_images_texts.json", 
    processor=processor
)

def collate_fn(batch):
    images, texts = zip(*batch)
    inputs = processor(
        text=list(texts), 
        images=list(images), 
        return_tensors="pt", 
        padding=True,
        truncation=True
    )
    return inputs.to(device)

dataloader = DataLoader(
    dataset, 
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)


def collate_fn(batch):
    images, texts = zip(*batch)
    inputs = processor(
        text=list(texts), 
        images=list(images), 
        return_tensors="pt", 
        padding=True,
        truncation=True
    )
    return inputs.to(device)

dataloader = DataLoader(
    dataset, 
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)

logit_scale = torch.nn.Parameter(torch.ones([]) * torch.tensor(1/0.07).log())


optimizer = torch.optim.AdamW(
    [
        {"params": vision_model.parameters()},
        {"params": text_model.parameters()},
        {"params": [logit_scale]}
    ],
    lr=1e-5,
    weight_decay=0.01
)


num_epochs = 2

for epoch in range(num_epochs):
    vision_model.train()
    text_model.train()
    total_loss = 0.0
    
    for batch in dataloader:

        vision_outputs = vision_model(**batch["pixel_values"])
        text_outputs = text_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        
        image_embeds = vision_outputs.image_embeds
        text_embeds = text_outputs.text_embeds
        
        logit_scale_value = logit_scale.exp()
        logits_per_image = logit_scale_value * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()
        
        batch_size = image_embeds.size(0)
        labels = torch.arange(batch_size, device=device)
        
        loss = (
            torch.nn.functional.cross_entropy(logits_per_image, labels) +
            torch.nn.functional.cross_entropy(logits_per_text, labels)
        ) / 2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

vision_model.save_pretrained("./fine_tuned_clip_vision")
text_model.save_pretrained("./fine_tuned_clip_text")
processor.save_pretrained("./fine_tuned_clip_processor")

