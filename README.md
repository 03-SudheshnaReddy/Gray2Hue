# Gray2Hue
# Image Colorization using a Conditional GAN (Gray2Hue)

A deep learning project that automatically adds color to grayscale images using a Conditional GAN (cGAN) based on the Pix2Pix architecture. The model learns to map the Lightness (L) channel of an image to its color (ab) channels in the CIELAB color space.


## About The Project

This project implements a Generative Adversarial Network to colorize black and white images. The core of the project is a cGAN that is trained to generate plausible color information for a given grayscale input.

* **Generator:** A U-Net with a ResNet-18 backbone is used as the generator. The U-Net architecture is excellent for image-to-image tasks as its skip connections help preserve low-level details between the input and output.
* **Discriminator:** A "PatchGAN" discriminator is used, which evaluates the realism of the image in overlapping patches rather than on the image as a whole. This encourages the generator to produce sharper, higher-quality textures.
* **Dataset:** The model was trained on a subset of the [COCO (Common Objects in Context)](https://cocodataset.org/) dataset.

### Built With

* [PyTorch](https://pytorch.org/)
* [Fastai](https://www.fast.ai/) (for the U-Net builder)
* [Streamlit](https://streamlit.io/) (for the web interface)
* [Pillow](https://python-pillow.org/) & [Scikit-Image](https://scikit-image.org/)

---

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

* Python 3.8 or higher
* Git

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/Gray2Hue.git](https://github.com/your-username/Gray2Hue.git)
    cd Gray2Hue
    ```
2.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```
3.  **Download the pre-trained model:**
    This project uses a pre-trained model. Make sure you have downloaded the `res18-unet.pt` file and placed it in the `model/` directory.

---

## Usage

To launch the web application, run the following command in your terminal from the project's root directory:

```sh
streamlit run app.py
```

This will open a new tab in your web browser. Simply upload a grayscale image and click the "Colorize Image" button to see the result!

---

## Model Architecture

The model is based on the Pix2Pix paper, which uses a Conditional GAN for image-to-image translation tasks.

* **Generator Loss:** The generator's loss is a combination of an adversarial loss (how well it fools the discriminator) and a content loss (L1 distance), which ensures the output is structurally similar to the ground truth.
    $L_{G} = L_{GAN}(G, D) + \lambda L_{L1}(G)$

* **Discriminator Loss:** The discriminator is trained to distinguish between real color images and fake (generated) color images.
    $L_{D} = \frac{1}{2} ( L_{real} + L_{fake} )$

