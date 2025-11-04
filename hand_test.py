BASE_DIRECTORY = " your directory"
MODEL_FILENAME = "hand_classifier_colab.pth"
DATASET_FOLDER_NAME = "hand_dataset"
AI_SUBFOLDER = "ai_hands"
REAL_SUBFOLDER = "real_hands"

MODEL_PATH = f"{BASE_DIRECTORY}/{MODEL_FILENAME}"
AI_FOLDER = f"{BASE_DIRECTORY}/{DATASET_FOLDER_NAME}/{AI_SUBFOLDER}"
REAL_FOLDER = f"{BASE_DIRECTORY}/{DATASET_FOLDER_NAME}/{REAL_SUBFOLDER}"


import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
import sys

print("✅ All libraries imported successfully!")
print(f"   Base Directory: {BASE_DIRECTORY}")
print(f"   Model Path: {MODEL_PATH}")
print(f"   AI Dataset Path: {AI_FOLDER}")
print(f"   Real Dataset Path: {REAL_FOLDER}\n")


def load_model(model_path, device):
    """Load the trained model - MUST MATCH TRAINING ARCHITECTURE"""

    print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model file not found at {model_path}")
        print("Please check the BASE_DIRECTORY path at the top of the script.")
        return None

    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1)
    )

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ ERROR loading model state_dict: {e}")
        print("This might mean the model architecture doesn't match the saved file.")
        return None


def predict_image(model, image_path, device):
    """Predict if image is AI-generated or Real (phone-captured)"""

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        raw_output = model(image_tensor).item()
        probability = torch.sigmoid(torch.tensor(raw_output)).item()

    is_real = probability >= 0.5
    confidence = probability if is_real else (1 - probability)

    return {
        'is_real': is_real,
        'probability': probability,
        'confidence': confidence * 100,
        'raw_output': raw_output
    }


def display_prediction(image_path, result):
    """Display image with beautiful prediction overlay"""

    img = Image.open(image_path)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(img)
    ax.axis('off')

    if result['is_real']:
        color = 'green'
        prediction = "REAL HAND\n(Phone Captured)"
        emoji = "✅"
        bg_color = '#d4edda'
    else:
        color = 'red'
        prediction = "AI GENERATED HAND"
        emoji = "🤖"
        bg_color = '#f8d7da'

    title = f"{emoji} {prediction}\n\nConfidence: {result['confidence']:.2f}%"
    ax.set_title(title, fontsize=18, fontweight='bold', color=color, pad=20,
                 bbox=dict(boxstyle='round,pad=1', facecolor=bg_color, alpha=0.8))

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 70)
    print("📊 DETAILED PREDICTION RESULTS")
    print("=" * 70)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Prediction: {emoji} {prediction}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print(f"-" * 70)
    print(f"Technical Details:")
    print(f"  Probability (Real): {result['probability']:.6f}")
    print(f"  Raw Model Output: {result['raw_output']:.6f}")
    print(f"  Threshold: 0.5 (above = Real, below = AI)")
    print("=" * 70 + "\n")


def test_folder(model, folder_path, device, max_images=20):
    """Test multiple images from a folder"""

    path = Path(folder_path)
    if not path.exists():
        print(f"❌ Folder not found: {folder_path}")
        return []

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(list(path.glob(ext)))

    if len(image_files) == 0:
        print(f"❌ No images found in {folder_path}")
        return []

    print(f"\n{'=' * 70}")
    print(f"📁 BATCH TEST")
    print(f"{'=' * 70}")
    print(f"Folder: {folder_path}")
    print(f"Found {len(image_files)} images")
    print(f"Testing first {min(max_images, len(image_files))} images...")
    print(f"{'=' * 70}\n")

    results = []

    for i, img_file in enumerate(image_files[:max_images], 1):
        print(f"[{i}/{min(max_images, len(image_files))}] {img_file.name}")

        try:
            result = predict_image(model, str(img_file), device)
            results.append((img_file.name, result))

            emoji = "✅" if result['is_real'] else "🤖"
            pred = "REAL" if result['is_real'] else "AI"
            print(f"   {emoji} {pred} (Confidence: {result['confidence']:.1f}%)")
            print(f"   Probability: {result['probability']:.4f}\n")
        except Exception as e:
            print(f"   ❌ Error: {e}\n")

    print("\n" + "=" * 70)
    print("📊 BATCH TEST SUMMARY")
    print("=" * 70)

    if results:
        ai_count = sum(1 for _, r in results if not r['is_real'])
        real_count = sum(1 for _, r in results if r['is_real'])
        avg_conf = np.mean([r['confidence'] for _, r in results])

        print(f"Total tested: {len(results)}")
        print(f"Predicted as AI: {ai_count} ({100 * ai_count / len(results):.1f}%)")
        print(f"Predicted as Real: {real_count} ({100 * real_count / len(results):.1f}%)")
        print(f"Average confidence: {avg_conf:.2f}%")

    print("=" * 70 + "\n")

    return results


def test_single_image(model, device):
    """Ask user for a path to a single image and test it."""
    print("\n" + "=" * 70)
    print("🖼️  TEST A SINGLE IMAGE")
    print("=" * 70)
    print("Please enter the full path to the image you want to test.")
    print("Example: C:/Users/sahas/Pictures/my_test_image.jpg")
    print("=" * 70 + "\n")

    image_path = input("Enter image path: ").strip().strip('"')

    if not os.path.exists(image_path):
        print(f"\n❌ ERROR: File not found at '{image_path}'")
        return

    try:
        result = predict_image(model, image_path, device)
        display_prediction(image_path, result)
    except Exception as e:
        print(f"\n❌ Error processing image: {e}")


def main():
    """Main testing interface"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print()

    model = load_model(MODEL_PATH, device)
    if model is None:
        input("Press Enter to exit...")
        sys.exit()

    while True:
        print("\n" + "🎯" * 35)
        print(" " * 20 + "FTT HAND DETECTOR - TEST MENU")
        print("🎯" * 35)
        print("\nChoose an option:\n")
        print("1. 🖼️  Test a single image (provide path)")
        print("2. 📁 Test images from AI folder")
        print("3. 📁 Test images from Real folder")
        print("4. 📂 Test images from custom folder path")
        print("5. ℹ️  Show model information")
        print("6. ❌ Exit")
        print("\n" + "─" * 70)

        choice = input("\nEnter your choice (1-6): ").strip()

        if choice == '1':
            test_single_image(model, device)

        elif choice == '2':
            print("\nTesting AI-generated images from training dataset...")
            results = test_folder(model, AI_FOLDER, device, max_images=10)
            if results:
                correct = sum(1 for _, r in results if not r['is_real'])
                accuracy = 100 * correct / len(results)
                print(f"\n📊 Accuracy on AI images: {accuracy:.2f}% ({correct}/{len(results)})")

        elif choice == '3':
            print("\nTesting real hand images from training dataset...")
            results = test_folder(model, REAL_FOLDER, device, max_images=10)
            if results:
                correct = sum(1 for _, r in results if r['is_real'])
                accuracy = 100 * correct / len(results)
                print(f"\n📊 Accuracy on Real images: {accuracy:.2f}% ({correct}/{len(results)})")

        elif choice == '4':
            folder_path = input("\nEnter full folder path: ").strip().strip('"')
            max_imgs = input("How many images to test? (default 20): ").strip()
            max_imgs = int(max_imgs) if max_imgs.isdigit() else 20
            test_folder(model, folder_path, device, max_images=max_imgs)

        elif choice == '5':
            print("\n" + "=" * 70)
            print("ℹ️  MODEL INFORMATION")
            print("=" * 70)
            print(f"Model Path: {MODEL_PATH}")
            print(f"Architecture: EfficientNet-B0")
            print(f"Classifier: 512 → 256 → 128 → 1")
            print(f"Device: {device}")
            print(f"Input Size: 224x224")
            print(f"Output: Raw logits (sigmoid applied during inference)")
            print(f"Classes: 0 = AI Generated, 1 = Real (Phone)")
            print("=" * 70)

        elif choice == '6':
            print("\n👋 Thank you for using FTT Hand Detector!")
            print("   Model trained and tested successfully! 🎉\n")
            break

        else:
            print("\n❌ Invalid choice! Please enter 1-6.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    print("\n" + "🤖" * 35)
    print(" " * 15 + "WELCOME TO FTT HAND DETECTOR TEST UI")
    print(" " * 20 + "AI vs Real Hand Classification")
    print("🤖" * 35)

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input("Press Enter to close the program.")