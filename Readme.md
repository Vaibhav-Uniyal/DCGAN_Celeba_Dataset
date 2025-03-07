# DCGAN on CelebA Dataset 🎭

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate human faces using the **CelebA dataset**. The model is trained following the principles from the paper **"Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"** by Radford et al.

---

## 📂 Dataset Preprocessing

### **1️⃣ Download the Dataset**
The CelebA dataset can be downloaded from Kaggle:

```bash
!mkdir data
!wget -P data https://s3.amazonaws.com/fast-ai-imageclas/celeba.tgz
!tar -xvzf data/celeba.tgz -C data
```

Alternatively, if using Kaggle:
```bash
!pip install kaggle
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d jessicali9530/celeba-dataset --unzip
```

### **2️⃣ Apply Preprocessing Steps**
- **Resize images** to `64x64 pixels` (DCGAN standard size)
- **Normalize pixel values** to `[-1,1]` for better GAN training
- **Use PyTorch's `ImageFolder` to load dataset**

```python
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Define transformations
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load dataset
dataset = datasets.ImageFolder(root="data/celeba", transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
```

---

## 🚀 Training the DCGAN Model

### **1️⃣ Train the Model**
Run the following script to train the DCGAN model:
```bash
python train.py
```

Alternatively, if running in a notebook:
```python
import torch
num_epochs = 50
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        # Training loop here
```

The model is trained using **Binary Cross-Entropy Loss (BCELoss)** and the **Adam optimizer** with:
- Learning rate: `0.0002`
- Momentum: `β1 = 0.5`

### **2️⃣ Save the Model**
```python
torch.save(netG.state_dict(), "dcgan_celeba_generator.pth")
torch.save(netD.state_dict(), "dcgan_celeba_discriminator.pth")
```

---

## 🧪 Testing the Model (Generate New Images)
After training, you can generate new images using the trained **Generator**:

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

# Load the trained Generator model
netG.load_state_dict(torch.load("dcgan_celeba_generator.pth"))
netG.eval()

# Generate new images
noise = torch.randn(16, 100, 1, 1, device="cuda")
with torch.no_grad():
    fake_images = netG(noise).cpu()

# Convert and display images
fake_images = fake_images * 0.5 + 0.5  # Normalize to [0,1]

def show_images(images, num_images=16):
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i >= num_images:
            break
        img = np.transpose(images[i].numpy(), (1, 2, 0))
        ax.imshow(img)
        ax.axis("off")
    plt.show()

show_images(fake_images)
```

---

## 🎨 Expected Outputs
After training for several epochs, the **generated faces** should look realistic. Initially, they may appear noisy, but over time, they improve in quality.

| **Epoch 0** | **Epoch 10** | **Epoch 50** |
|-----------|------------|------------|
| ![Epoch 0]("C:\Users\vaibh\Desktop\image_epoch5.png") | ![Epoch 10](examples/epoch_10.png) | ![Epoch 50](examples/epoch_50.png) |

*Note: The more epochs, the better the image quality.*

---

## 🏆 Next Steps
- Tune **hyperparameters** for better results
- Train for **more epochs** to improve image quality
- Use **FID Score** to measure image realism

Made with ❤️ by [Your Name]

