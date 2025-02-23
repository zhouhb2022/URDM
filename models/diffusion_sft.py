import torch,torchvision,json
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, DDPMScheduler
from torch.optim import AdamW
from PIL import Image
import torch.nn.functional as F
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4'
model_name = "./models/stable-diffusion"
batch_size = 10
num_epochs = 100
learning_rate = 1e-6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = StableDiffusionPipeline.from_pretrained(model_name)
model = pipe.unet
text_encoder = pipe.text_encoder
vae = pipe.vae
noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

for param in text_encoder.parameters():
    param.requires_grad = False
for param in vae.parameters():
    param.requires_grad = False

class TextImageDataset(Dataset):
    def __init__(self, image_paths, texts, tokenizer, transform=None):  
        self.image_paths = image_paths
        self.texts = texts
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        text = self.texts[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return {"pixel_values": image, "input_ids": pipe.tokenizer(
            text, 
            max_length=pipe.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)}

def train(model, dataloader, optimizer):
    model.train()
    
    for epoch in range(num_epochs):
        for batch in dataloader:

            images = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            

            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * 0.18215

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, 
                (latents.shape[0],), device=device
            ).long()
            

            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            

            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]
            

            noise_pred = model(
                noisy_latents, 
                timesteps, 
                encoder_hidden_states
            ).sample
            

            loss = F.mse_loss(noise_pred, noise)
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch}, Loss: {loss.item()}")


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(512),
    torchvision.transforms.CenterCrop(512),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5], [0.5])  
])



image_dir = "../pheme_images_consistency"
json_path = "../pheme_images_texts.json"

with open(json_path, 'r') as f:
    text_mapping = json.load(f)

image_paths = []
texts = []


for img_name in os.listdir(image_dir):

    base_name = os.path.basename(img_name)
    
    if base_name in text_mapping:
        image_paths.append(os.path.join(image_dir, img_name))
        texts.append(text_mapping[base_name])
    else:
        print(f"Warning: {img_name} has no corresponding text description")

assert len(image_paths) == len(texts), "nums not consistency"


dataset = TextImageDataset(
    image_paths, 
    texts, 
    tokenizer=pipe.tokenizer,  
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

optimizer = AdamW(model.parameters(), lr=learning_rate)


model.to(device)
text_encoder.to(device)
vae.to(device)

train(model, dataloader, optimizer)

model.save_pretrained("models/stable-diffusion/new_unet")

