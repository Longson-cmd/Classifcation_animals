import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import Simple_model

# Define class names
class_names = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# Load the model
model = Simple_model()
checkpoint_path = "best_accuracy_model.pth"  # Adjust if needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_accuracy_model = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(best_accuracy_model) 
model.eval()  # Set model to evaluation mode

# Define the transformation (adjust based on your model's training process)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess an image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Convert to RGB
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Run inference and display result
def infer_and_display(image_path):
    image_tensor = preprocess_image(image_path)
    
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(image_tensor)
        print(output)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        print(probabilities)  # Convert logits to probabilities
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        print(predicted_class_idx)
        confidence = probabilities[0, predicted_class_idx].item() * 100  # Convert to percentage
    
    predicted_class = class_names[predicted_class_idx]

    # Display image with title
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis("off")  # Hide axis
    
    # Set title
    plt.title(f"Predicted: {predicted_class} ({confidence:.2f}%)", fontsize=12, color="black")
    
    plt.show()

# Example usage
image_path = "C:/Users/PC/Desktop/unfinish_project/animal/5.png"  # Replace with your test image
infer_and_display(image_path)




