# Import Streamlit for building the web UI
import streamlit as st

# Import PIL Image for image loading and manipulation
from PIL import Image

# Import tempfile to store uploaded video files temporarily
import tempfile

# Import PyTorch core library
import torch

# Import neural network modules from PyTorch
import torch.nn as nn

# Import functional utilities like softmax
import torch.nn.functional as F

# Import torchvision transforms for image preprocessing
from torchvision import transforms

# Import Vision Transformer model and its image processor
from transformers import ViTForImageClassification, ViTImageProcessor

# Import OpenCV for video processing
import cv2


# =========================
# DEVICE
# =========================

# Select GPU (CUDA) if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# CNN ARCHITECTURE
# =========================

# Define a simple Convolutional Neural Network
class SimpleCNN(nn.Module):

    # Constructor for the CNN
    def __init__(self):
        # Call parent class constructor
        super().__init__()

        # Define convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 3-channel RGB → 32 feature maps
            nn.ReLU(),                                  # Apply ReLU activation
            nn.MaxPool2d(2),                            # Downsample by factor of 2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),# 32 → 64 channels
            nn.ReLU(),                                  # Activation
            nn.MaxPool2d(2),                            # Downsampling

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# 64 → 128 channels
            nn.ReLU(),                                   # Activation
            nn.MaxPool2d(2)                              # Downsampling
        )

        # Define fully connected (feature vector) layers
        self.fv_layer = nn.Sequential(
            nn.Flatten(),                               # Flatten feature maps
            nn.Linear(128 * 28 * 28, 512),               # Dense layer
            nn.ReLU(),                                   # Activation
            nn.Dropout(0.5),                             # Dropout for regularization
            nn.Linear(512, 1)                            # Output single logit
        )

    # Forward pass of the CNN
    def forward(self, x):
        x = self.conv_layers(x)  # Pass input through convolution layers
        x = self.fv_layer(x)     # Pass features through fully connected layers
        return x                 # Return output logits


# =========================
# LOAD CNN
# =========================

# Initialize CNN model
cnn_model = SimpleCNN()

# Load pretrained CNN weights
cnn_model.load_state_dict(
    torch.load(r"E:\pytorch\simple_cnn.pth", map_location=device)
)

# Move CNN model to selected device
cnn_model.to(device)

# Set model to evaluation mode
cnn_model.eval()

# Define image preprocessing for CNN
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224
    transforms.ToTensor()           # Convert image to PyTorch tensor
])


# =========================
# LOAD ViT
# =========================

# Path to fine-tuned Vision Transformer model
vit_path = r"E:\pytorch\content\ai_vs_real_vit_model"

# Load image processor for ViT
vit_processor = ViTImageProcessor.from_pretrained(vit_path)

# Load ViT model for image classification
vit_model = ViTForImageClassification.from_pretrained(vit_path)

# Move ViT model to device
vit_model.to(device)

# Set ViT model to evaluation mode
vit_model.eval()


# =========================
# PREDICTION FUNCTIONS
# =========================

# Predict probabilities using CNN
def predict_cnn(image):
    # Apply preprocessing and add batch dimension
    x = cnn_transform(image).unsqueeze(0).to(device)

    # Disable gradient calculation for inference
    with torch.no_grad():
        logits = cnn_model(x)          # Get raw output
        probs = torch.sigmoid(logits)  # Convert to probability
        # Convert single output to two-class format [REAL, FAKE]
        probs = torch.cat([probs, 1 - probs], dim=1)

    return probs


# Predict probabilities using ViT
def predict_vit(image):
    # Preprocess image for ViT
    inputs = vit_processor(image, return_tensors="pt")

    # Move tensors to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Disable gradient computation
    with torch.no_grad():
        logits = vit_model(**inputs).logits  # Get logits
        probs = F.softmax(logits, dim=1)     # Convert to probabilities

    return probs


# Combine CNN and ViT predictions
def ensemble_predict(image, is_image=True):

    # Load image if a file path is provided
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    # Get predictions from CNN and ViT
    cnn_probs = predict_cnn(image)
    vit_probs = predict_vit(image)

    if is_image:
        # If ViT predicts REAL with ≥ 20% confidence → REAL
        if vit_probs[0, 0] >= 0.2:
            final_pred = 0
        else:
            # Weighted ensemble (CNN 30%, ViT 70%)
            final_probs = 0.3 * cnn_probs + 0.7 * vit_probs
            final_pred = final_probs.argmax(dim=1).item()
    else:
        # For videos, always use weighted ensemble
        final_probs = 0.3 * cnn_probs + 0.7 * vit_probs
        final_pred = final_probs.argmax(dim=1).item()

    # Return human-readable label
    return "REAL" if final_pred == 0 else "FAKE"


# Predict video by sampling frames
def predict_video(video_file, frame_interval=10, return_frame_preds=False):

    # Save uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name

    # Open video file
    cap = cv2.VideoCapture(tmp_path)

    frame_count = 0                 # Track frame index
    frame_preds = []                # Store predictions per frame

    # Create Streamlit progress bar
    progress_bar = st.progress(0)

    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()     # Read next frame
        if not ret:
            break

        # Process every nth frame
        if frame_count % frame_interval == 0:
            # Convert OpenCV frame to PIL Image
            frame_pil = Image.fromarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            # Predict frame label
            pred = ensemble_predict(frame_pil, is_image=False)
            frame_preds.append(pred)

        frame_count += 1

        # Update progress bar
        progress_bar.progress(min(frame_count / total_frames, 1.0))

    # Release video capture
    cap.release()

    # Final decision: REAL if any frame is REAL
    final_video_pred = "REAL" if "REAL" in frame_preds else "FAKE"

    if return_frame_preds:
        return final_video_pred, frame_preds

    return final_video_pred


# =========================
# STREAMLIT UI
# =========================

# Display app title
st.title("AI vs Real Detector")

# Choose input type
type_choice = st.radio("Choose type:", ["Image", "Video"])

# Upload file widget
uploaded_file = st.file_uploader(f"Upload a {type_choice.lower()} file")

if uploaded_file is not None:

    if type_choice == "Image":
        # Load and display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict image when button clicked
        if st.button("Predict Image"):
            result = ensemble_predict(image, is_image=True)
            st.write(f"Prediction: {result}")

    else:
        # Display uploaded video
        st.video(uploaded_file)

        # Predict video when button clicked
        if st.button("Predict Video"):
            st.write("Processing video... This may take a while.")
            final_pred = predict_video(uploaded_file)
            st.write(f"Final Prediction: {final_pred}")

else:
    # Warning if no file uploaded
    st.warning("Please upload a file first!")
