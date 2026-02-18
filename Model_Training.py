import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import torch
from torchvision import transforms
import timm
from classsifier import Proposed_Grad_CAM_BNN



def tr_feature_extraction(main_dir,file_path):
    # Path to the main dataset folder
    main_dir = '../Dataset/chest_xray/train'

    # Preprocessing pipeline (resize + normalize)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load Swin Transformer (no classification head)
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)
    model.eval()

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Prepare lists to store features and labels
    all_features = []
    all_labels = []

    # Process each label directory (subdirectory under main_dir)
    for label_name in sorted(os.listdir(main_dir)):
        sub_dir = os.path.join(main_dir, label_name)
        if not os.path.isdir(sub_dir):
            continue

        print(f"Processing label: {label_name}")

        # Limit the number of images processed per label to 1300
        img_count = 0

        for img_name in os.listdir(sub_dir):
            if img_count >= 1300:  # Stop once we've processed 1300 images
                break

            img_path = os.path.join(sub_dir, img_name)

            try:
                # Load and apply median filter to the image
                image = Image.open(img_path).convert('RGB')
                image = image.filter(ImageFilter.MedianFilter(size=3))

                # Preprocess the image
                image_tensor = transform(image).unsqueeze(0).to(device)

                # Extract features using the Swin Transformer
                with torch.no_grad():
                    features = model(image_tensor).cpu().numpy().flatten()

                # Save the features and the label
                all_features.append(features)
                all_labels.append(label_name)

                img_count += 1  # Increment the image count

            except Exception as e:
                print(f"❌ Failed to process {img_path}: {e}")

    # Convert the list of features to a DataFrame and save to CSV
    features_df = pd.DataFrame(all_features)
    features_df['label'] = all_labels
    features_df.to_csv(file_path, index=False)

    print(f"✅ Feature extraction complete. Saved to '{file_path}'")
class OptimizedTensorFusionNetwork:
    def __init__(self, input_dim1, input_dim2, output_dim=512):
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim

        self.full_dim = (self.input_dim1 + 1) * (self.input_dim2 + 1)

        if self.output_dim < self.full_dim:
            self.projection_matrix = np.random.randn(self.full_dim, self.output_dim)
        else:
            self.projection_matrix = None

    def fuse(self, x, y):
        x = np.squeeze(x)
        y = np.squeeze(y)

        assert x.shape[0] == self.input_dim1, f"x should be shape ({self.input_dim1},) but got {x.shape}"
        assert y.shape[0] == self.input_dim2, f"y should be shape ({self.input_dim2},) but got {y.shape}"

        x_aug = np.concatenate(([1.0], x))
        y_aug = np.concatenate(([1.0], y))

        fusion_tensor = np.outer(x_aug, y_aug).flatten()

        if self.projection_matrix is not None:
            fused_vector = fusion_tensor @ self.projection_matrix
        else:
            fused_vector = fusion_tensor

        return fused_vector

def fuse_csv_features(csv1_path, csv2_path, output_csv_path, output_dim=512):
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    if df1.shape[0] != df2.shape[0]:
        raise ValueError("The number of rows in both CSVs must be the same for pairwise fusion.")

    labels1 = df1['label'].values
    labels2 = df2['label'].values

    if not np.array_equal(labels1, labels2):
        raise ValueError("Labels are not aligned. Ensure both CSVs are in the same order.")

    features1 = df1.drop(columns=['label']).values
    features2 = df2.drop(columns=['label']).values

    input_dim1 = features1.shape[1]
    input_dim2 = features2.shape[1]

    otfn = OptimizedTensorFusionNetwork(input_dim1, input_dim2, output_dim)

    fused_features = []
    for x, y in zip(features1, features2):
        fused = otfn.fuse(x, y)
        fused_features.append(fused)

    fused_df = pd.DataFrame(fused_features)
    fused_df['label'] = labels1
    fused_df.to_csv(output_csv_path, index=False)
    print(f"✅ Fused features saved to {output_csv_path}")



if not os.path.exists("xray_features_with_labels.csv"):
    print("Processing X-ray dataset")
    tr_feature_extraction("../Dataset/chest_xray/train","xray_features_with_labels.csv")
if not os.path.exists("ct_features_with_labels.csv"):
    print("Processing CT dataset")
    tr_feature_extraction("../Dataset/CT_image","ct_features_with_labels.csv")
if not os.path.exists("fused_tr_features.csv"):
    fuse_csv_features("xray_features_with_labels.csv", "ct_features_with_labels.csv", "fused_tr_features.csv")
if not os.path.exists("bnn_model.pth"):
    Proposed_Grad_CAM_BNN.pr_classifier("fused_tr_features.csv", "../img_res/tr_feature_index.jpg")