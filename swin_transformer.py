import torch
import timm
import torchvision.transforms as transforms
from PIL import Image

def feature_extractor(img_path):
    # Load image
    image = Image.open(img_path).convert('RGB')

    # Preprocess image for Swin Transformer
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Load pretrained Swin Transformer
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)  # num_classes=0 disables classification head
    model.eval()

    # Extract features
    with torch.no_grad():
        features = model(input_tensor)  # Output shape: [1, feature_dim]

    print("Extracted Feature Vector Shape:", features.shape)
    print("Feature Vector:\n", features.squeeze().cpu().numpy())  # Print the vector nicely
    return features
