import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import os
from model import (
    AlexNet, LeNet, VGG16, VGG16BatchNorm,
    resnet18, resnet34, resnet50, resnet101,
    InceptionV3, MobileNetV3, VisionTransformer,
    EfficientNetV2
)

# ==========================================
# 1. DATASET CONFIGURATION
# ==========================================

# Class names for Intel Image Classification
CLASSES_INTEL = {0: "buildings", 1: "forest", 2: "glacier", 3: "mountain", 4: "sea", 5: "street"}

# Class names for MIT Indoor Scenes
CLASSES_MIT = {
    0: "airport_inside", 1: "artstudio", 2: "auditorium", 3: "bakery", 4: "bar",
    5: "bathroom", 6: "bedroom", 7: "bookstore", 8: "bowling", 9: "buffet",
    10: "casino", 11: "children_room", 12: "church_inside", 13: "classroom", 14: "cloister",
    15: "closet", 16: "clothing_store", 17: "computer_room", 18: "concert_hall", 19: "corridor",
    20: "deli", 21: "dental_office", 22: "dining_room", 23: "elevator", 24: "fastfood_restaurant",
    25: "florist", 26: "game_room", 27: "garage", 28: "greenhouse", 29: "grocery_store",
    30: "gym", 31: "hair_salon", 32: "hospital", 33: "inside_bus", 34: "inside_subway",
    35: "jewelry_store", 36: "kindergarten", 37: "kitchen", 38: "laboratorywet", 39: "laundromat",
    40: "library", 41: "living_room", 42: "lobby", 43: "lockeroom", 44: "mall",
    45: "meeting_room", 46: "movie_theater", 47: "museum", 48: "nursery", 49: "office",
    50: "operating_room", 51: "pantry", 52: "poolinside", 53: "prison_cell", 54: "restaurant",
    55: "restaurant_kitchen", 56: "shoeshop", 57: "stairscase", 58: "studiomusic", 59: "subway",
    60: "toy_store", 61: "train_station", 62: "tv_studio", 63: "video_store", 64: "waiting_room",
    65: "warehouse", 66: "winecellar"
}

# Class names for Fashion MNIST
CLASSES_FASHION = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
}

# Class names for MNIST
CLASSES_MNIST = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9"
}

# Main Configuration Dictionary
DATASET_CONFIG = {
    "Intel Image": {
        "num_classes": 6,
        "folder": "intel",
        "classes": CLASSES_INTEL,
        "input_size": 224,
        "in_channels": 3
    },
    "MIT Indoor": {
        "num_classes": 67,
        "folder": "mit",
        "classes": CLASSES_MIT,
        "input_size": 224,
        "in_channels": 3
    },
    "CIFAR-100": {
        "num_classes": 100,
        "folder": "cifar100",
        "classes": None,
        "input_size": 224,
        "in_channels": 3
    },
    "CIFAR-100 (224x224)": {
        "num_classes": 100,
        "folder": "cifar100_224",
        "classes": None,
        "input_size": 224,
        "in_channels": 3
    },
    "Caltech101": {
        "num_classes": 101,
        "folder": "caltech101",
        "classes": None,
        "input_size": 224,
        "in_channels": 3
    },
    "Fashion MNIST": {
        "num_classes": 10,
        "folder": "fashionmnist",
        "classes": CLASSES_FASHION,
        "input_size": 224,
        "in_channels": 1
    },
    "MNIST": {
        "num_classes": 10,
        "folder": "mnist",
        "classes": CLASSES_MNIST,
        "input_size": 224,
        "in_channels": 1
    },
    "ImageNet": {
        "num_classes": 1000,
        "folder": "imagenet",
        "classes": None,
        "input_size": 224,
        "in_channels": 3
    }
}

# List of supported models (matching train.py)
AVAILABLE_MODELS = [
    "alexnet", "lenet", "vgg16", "vgg16_bn",
    "resnet18", "resnet34", "resnet50", "resnet101",
    "inceptionv3", "mobilenetv3_s", "mobilenetv3_l",
    "vit", "efficientnetv2_s", "efficientnetv2_m", "efficientnetv2_l"
]

# Detect device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. MODEL BUILDING FUNCTION
# ==========================================
def get_model_structure(model_name, num_classes, input_size=224, in_channels=3, dropout_rate=0.4):
    """
    Builds the model architecture matching train.py logic.
    """
    try:
        if model_name == 'alexnet':
            model = AlexNet(num_classes=num_classes)
        elif model_name == 'lenet':
            model = LeNet(num_classes=num_classes, in_channels=in_channels)
        elif model_name == 'vgg16':
            model = VGG16(num_classes=num_classes, in_channels=in_channels, dropout_rate=dropout_rate, input_size=input_size)
        elif model_name == 'vgg16_bn':
            model = VGG16BatchNorm(num_classes=num_classes, in_channels=in_channels, dropout_rate=dropout_rate, input_size=input_size)
        elif model_name == 'resnet18':
            model = resnet18(num_classes=num_classes, in_channels=in_channels)
        elif model_name == 'resnet34':
            model = resnet34(num_classes=num_classes, in_channels=in_channels)
        elif model_name == 'resnet50':
            model = resnet50(num_classes=num_classes, in_channels=in_channels)
        elif model_name == 'resnet101':
            model = resnet101(num_classes=num_classes, in_channels=in_channels)
        elif model_name == 'inceptionv3':
            model = InceptionV3(num_classes=num_classes, in_channels=in_channels)
        elif model_name == 'mobilenetv3_s':
            model = MobileNetV3(mode='small', num_classes=num_classes, dropout=dropout_rate)
        elif model_name == 'mobilenetv3_l':
            model = MobileNetV3(mode='large', num_classes=num_classes, dropout=dropout_rate)
        elif model_name == 'vit':
            model = VisionTransformer(num_classes=num_classes, dropout_rate=dropout_rate)
        elif model_name == 'efficientnetv2_s':
            model = EfficientNetV2(version='s', num_classes=num_classes, dropout_rate=dropout_rate)
        elif model_name == 'efficientnetv2_m':
            model = EfficientNetV2(version='m', num_classes=num_classes, dropout_rate=dropout_rate)
        elif model_name == 'efficientnetv2_l':
            model = EfficientNetV2(version='l', num_classes=num_classes, dropout_rate=dropout_rate)
        else:
            raise ValueError(f"Model {model_name} not recognized.")
        
        return model.to(DEVICE)
    except Exception as e:
        raise ValueError(f"Failed to initialize model {model_name}: {str(e)}")

# ==========================================
# 3. IMAGE PROCESSING & WEIGHT LOADING
# ==========================================
def get_transform(in_channels=3):
    """Transform based on number of input channels"""
    if in_channels == 1:
        # Grayscale images (MNIST, Fashion-MNIST)
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        # RGB images (standard ImageNet normalization)
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

def predict(image, dataset_name, model_name):
    """
    Main prediction function called by Gradio interface.
    """
    if image is None:
        return None, "⚠️ Please upload an image."
    
    # 1. Retrieve config info
    config = DATASET_CONFIG.get(dataset_name)
    num_classes = config["num_classes"]
    folder = config["folder"]
    class_map = config["classes"]
    input_size = config["input_size"]
    in_channels = config["in_channels"]
    
    # 2. Construct weight file path
    # Assumption: weights are stored as: weights/intel/resnet50.pth
    weight_path = os.path.join("weights", folder, f"{model_name}.pth")
    
    if not os.path.exists(weight_path):
        return None, f"❌ Weights file not found at:\n{weight_path}\n\nPlease ensure you have trained and saved the model correctly."

    try:
        # 3. Initialize and Load Model
        model = get_model_structure(
            model_name=model_name,
            num_classes=num_classes,
            input_size=input_size,
            in_channels=in_channels,
            dropout_rate=0.4
        )
        
        # Load weights to CPU/GPU
        state_dict = torch.load(weight_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()

        # 4. Preprocess Image
        if in_channels == 1:
            # Convert to grayscale
            image = image.convert("L")
        else:
            # Convert to RGB to handle Grayscale or RGBA
            image = image.convert("RGB")
        
        transform = get_transform(in_channels)
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # 5. Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1).squeeze()
        
        # 6. Get Top 5 Results
        topk = min(5, num_classes)
        topk_probs, topk_indices = torch.topk(probs, topk)
        
        results = {}
        for i in range(topk_probs.size(0)):
            idx = topk_indices[i].item()
            score = topk_probs[i].item()
            
            # Get class name
            if class_map:
                label = class_map.get(idx, f"Class {idx}")
            else:
                label = f"Class {idx}"
            
            results[label] = score

        return results, f"✅ Prediction successful on {DEVICE}"

    except Exception as e:
        import traceback
        error_msg = f"❌ Runtime Error: {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg

# ==========================================
# 4. GRADIO INTERFACE
# ==========================================
def create_demo():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🧠 Universal Computer Vision Demo
            A versatile image classification application supporting multiple Datasets and Model Architectures.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input Controls
                dataset_dd = gr.Dropdown(
                    choices=list(DATASET_CONFIG.keys()),
                    value="Intel Image",
                    label="1. Select Dataset",
                    info="Choose the dataset you want to test against"
                )
                
                model_dd = gr.Dropdown(
                    choices=AVAILABLE_MODELS,
                    value="resnet18",
                    label="2. Select Model Architecture",
                    info="Ensure the model matches the saved .pth file"
                )
                
                input_img = gr.Image(type="pil", label="3. Upload Image")
                
                predict_btn = gr.Button("🚀 Classify Now", variant="primary")

            with gr.Column(scale=1):
                # Output Display
                output_label = gr.Label(num_top_classes=5, label="Prediction Results (Top 5)")
                status_txt = gr.Textbox(label="System Status", interactive=False)

        # Event Linking
        predict_btn.click(
            fn=predict,
            inputs=[input_img, dataset_dd, model_dd],
            outputs=[output_label, status_txt]
        )
        
        gr.Markdown("---")
        gr.Markdown("### 📝 Setup Instructions:")
        gr.Markdown(
            """
            1. Create a `weights/` directory in the root folder.
            2. Inside `weights/`, create subfolders for each dataset:
               - `weights/intel/` for Intel Image dataset
               - `weights/mit/` for MIT Indoor dataset
               - `weights/cifar100/` for CIFAR-100 dataset
               - `weights/caltech101/` for Caltech101 dataset
               - `weights/fashionmnist/` for Fashion-MNIST dataset
               - `weights/mnist/` for MNIST dataset
               - `weights/imagenet/` for ImageNet dataset
            3. Place your trained model `.pth` files in the corresponding folder with names matching the model architecture:
               - Example: `weights/intel/resnet18.pth`
               - Example: `weights/fashionmnist/vgg16_bn.pth`
            
            **Supported Models:**
            - AlexNet, LeNet, VGG16, VGG16-BN
            - ResNet18, ResNet34, ResNet50, ResNet101
            - InceptionV3
            - MobileNetV3 (Small & Large)
            - Vision Transformer (ViT)
            - EfficientNetV2 (Small, Medium, Large)
            """
        )

    return demo

if __name__ == "__main__":
    app = create_demo()
    app.launch(share=True)