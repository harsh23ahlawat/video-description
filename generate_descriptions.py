import openai
import os
from PIL import Image

openai.api_key = "YOUR_API_KEY"

def generate_description(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        
    response = openai.Image.create(
        file=image_data,
        model="dall-e"
    )
    
    description = response['choices'][0]['text']
    return description

def main():
    output_folder = 'extracted_frames'
    frames = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.png')]

    for frame in frames:
        description = generate_description(frame)
        print(f"Description for {frame}:\n{description}\n")

if __name__ == "__main__":
    main()
