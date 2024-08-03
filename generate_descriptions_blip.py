from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

# Load pre-trained BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_description(image_path):
    # Open and process the image
    raw_image = Image.open(image_path).convert("RGB")

    # Process the image and generate caption
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)

    return description

def main():
    output_folder = 'extracted_frames'  # Folder where the extracted frames are saved
    frames = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.png')]

    for frame in frames:
        description = generate_description(frame)
        print(f"Description for {frame}:\n{description}\n")

if __name__ == "__main__":
    main()
