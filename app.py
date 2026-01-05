import streamlit as st
from PIL import Image
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
import cv2

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# CNN ARCHITECTURE
# =========================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fv_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*28*28,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,1)
        )

    def forward(self,x):
        x = self.conv_layers(x)
        x = self.fv_layer(x)
        return x

# =========================
# LOAD CNN
# =========================
cnn_model = SimpleCNN()
cnn_model.load_state_dict(torch.load(r"E:\pytorch\simple_cnn.pth", map_location=device))
cnn_model.to(device)
cnn_model.eval()

cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# LOAD ViT
# =========================
vit_path = r"E:\pytorch\content\ai_vs_real_vit_model"
vit_processor = ViTImageProcessor.from_pretrained(vit_path)
vit_model = ViTForImageClassification.from_pretrained(vit_path)
vit_model.to(device)
vit_model.eval()

# =========================
# PREDICTION FUNCTIONS
# =========================
def predict_cnn(image):
    x = cnn_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = cnn_model(x)
        # Since CNN output is 1 neuron, use sigmoid for probability
        probs = torch.sigmoid(logits)
        probs = torch.cat([probs, 1 - probs], dim=1)  # [REAL, FAKE]
    return probs

def predict_vit(image):
    inputs = vit_processor(image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = vit_model(**inputs).logits
        probs = F.softmax(logits, dim=1)
    return probs

def ensemble_predict(image, is_image=True):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    cnn_probs = predict_cnn(image)
    vit_probs = predict_vit(image)

    if is_image:
        # For images, ViT ≥ 20% REAL → REAL
        if vit_probs[0, 0] >= 0.2:
            final_pred = 0
        else:
            final_probs = 0.3 * cnn_probs + 0.7 * vit_probs
            final_pred = final_probs.argmax(dim=1).item()
    else:
        # For videos, always use weighted ensemble
        final_probs = 0.3 * cnn_probs + 0.7 * vit_probs
        final_pred = final_probs.argmax(dim=1).item()

    return "REAL" if final_pred == 0 else "FAKE"

def predict_video(video_file, frame_interval=10, return_frame_preds=False):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    frame_count = 0
    frame_preds = []

    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pred = ensemble_predict(frame_pil, is_image=False)
            frame_preds.append(pred)

        frame_count += 1
        progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    final_video_pred = "REAL" if "REAL" in frame_preds else "FAKE"

    if return_frame_preds:
        return final_video_pred, frame_preds
    return final_video_pred

# =========================
# STREAMLIT UI
# =========================
st.title("AI vs Real Detector")

type_choice = st.radio("Choose type:", ["Image", "Video"])
uploaded_file = st.file_uploader(f"Upload a {type_choice.lower()} file")

if uploaded_file is not None:
    if type_choice == "Image":
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Predict Image"):
            result = ensemble_predict(image, is_image=True)
            st.write(f"Prediction: {result}")
    else:
        st.video(uploaded_file)
        if st.button("Predict Video"):
            st.write("Processing video... This may take a while.")
            final_pred = predict_video(uploaded_file)
            st.write(f"Final Prediction: {final_pred}")
else:
    st.warning("Please upload a file first!")
