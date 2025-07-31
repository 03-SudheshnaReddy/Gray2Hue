import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as T
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet
from skimage.color import rgb2lab, lab2rgb
import numpy as np
import requests
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(
    page_title="Gray2Hue AI Colorization",
    page_icon="🎨",
    layout="wide",
)

# --- Custom CSS for Footer ---
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: #808080;
    text-align: center;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar Content ---
with st.sidebar:
    st.title("🎨 Gray2Hue AI")
    st.info("This web app uses a Generative Adversarial Network (GAN) to bring grayscale images to life with color.")
    st.write("---")
    st.write("The model is a U-Net with a ResNet-18 backbone, trained on a subset of the COCO dataset.")
    st.write("---")
    st.subheader("🔗 Project & Author Links")
    st.markdown("""
    - **GitHub Profile:** [03-SudheshnaReddy](https://github.com/03-SudheshnaReddy)
    - **Project Repository:** [Gray2Hue](https://github.com/03-SudheshnaReddy/Gray2Hue)
    """)

# --- Model & Image Processing Functions ---

@st.cache_resource
def load_generator_model():
    """Loads the pre-trained generator model and sets it to evaluation mode."""
    model_path = 'model/final_model_weights.pt'
    net_G = build_res_unet(n_input=1, n_output=2, size=256)

    # Load the full state dictionary from the file
    full_state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Create a new dictionary to hold only the generator's weights
    generator_state_dict = {}
    for key, value in full_state_dict.items():
        if key.startswith('net_G.'):
            # Remove the 'net_G.' prefix from the key
            new_key = key.replace('net_G.', '', 1)
            generator_state_dict[new_key] = value

    # Load the cleaned dictionary into the generator model
    net_G.load_state_dict(generator_state_dict)
    
    net_G.eval()
    return net_G

def build_res_unet(n_input=1, n_output=2, size=256):
    """Builds the U-Net Generator model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet18(), pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G

def lab_to_rgb(L, ab):
    """Converts a batch of LAB image tensors to RGB numpy arrays."""
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

def process_and_display(image, model):
    """Takes a PIL image, processes it, and displays the result."""
    # Prepare image for the model
    img_resized = image.resize((256, 256))
    img_array = np.array(img_resized)
    img_lab = rgb2lab(np.stack([img_array] * 3, axis=2)).astype("float32")
    img_lab = T.ToTensor()(img_lab)
    L = img_lab[[0], ...] / 50. - 1.
    L = L.unsqueeze(0)

    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Original Grayscale Image', use_column_width=True)

    with col2:
        with st.spinner('The AI is painting your image... 🎨'):
            # Generate prediction
            with torch.no_grad():
                preds_ab = model(L)

            # Convert to RGB color image
            colorized_image_np = lab_to_rgb(L, preds_ab.cpu())[0]

            # Display the result
            st.image(colorized_image_np, caption='Colorized by Gray2Hue AI', use_column_width=True)
            st.success("Colorization complete!")
            st.balloons()

# --- Main Application Interface ---

st.title("Automatic Image Colorization")
st.markdown("Upload a black and white photo or try one of the examples below and let our AI model predict the colors.")
st.write("---")

# Load the model
model = load_generator_model()

# --- Example Images ---
st.subheader("Try an Example")
col1_ex, col2_ex, col3_ex = st.columns(3)

example_images = {
    "Portrait": "https://images.pexels.com/photos/3774567/pexels-photo-3774567.jpeg?auto=compress&cs=tinysrgb&w=800&lazy=load",
    "Landscape": "https://images.pexels.com/photos/164502/pexels-photo-164502.jpeg?auto=compress&cs=tinysrgb&w=800&lazy=load",
    "Architecture": "https://images.pexels.com/photos/1034664/pexels-photo-1034664.jpeg?auto=compress&cs=tinysrgb&w=800&lazy=load"
}

with col1_ex:
    st.image(example_images["Portrait"], use_column_width=True, caption="Portrait Example")
    if st.button("Colorize Portrait", use_container_width=True):
        response = requests.get(example_images["Portrait"])
        image = Image.open(BytesIO(response.content)).convert("L")
        process_and_display(image, model)

with col2_ex:
    st.image(example_images["Landscape"], use_column_width=True, caption="Landscape Example")
    if st.button("Colorize Landscape", use_container_width=True):
        response = requests.get(example_images["Landscape"])
        image = Image.open(BytesIO(response.content)).convert("L")
        process_and_display(image, model)

with col3_ex:
    st.image(example_images["Architecture"], use_column_width=True, caption="Architecture Example")
    if st.button("Colorize Architecture", use_container_width=True):
        response = requests.get(example_images["Architecture"])
        image = Image.open(BytesIO(response.content)).convert("L")
        process_and_display(image, model)

st.write("---")

# --- User Image Uploader ---
st.subheader("Or Upload Your Own")
uploaded_file = st.file_uploader("Choose a grayscale image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    user_image = Image.open(uploaded_file).convert("L")
    process_and_display(user_image, model)

# --- Footer ---
st.markdown('<div class="footer"><p>Made with ❤️ by Sudheshna Reddy</p></div>', unsafe_allow_html=True)