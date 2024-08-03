import os
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

# Load the model and necessary components
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_description(image_path):
    # Open the image
    image = Image.open(image_path).convert("RGB")
    
    # Preprocess the image
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    
    # Generate the caption
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    description = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return description

def main():
    output_folder = 'extracted_frames'
    frames = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.jpg')]

    for frame in frames:
        description = generate_description(frame)
        print(f"Description for {frame}:\n{description}\n")

if __name__ == "__main__":
    main()
