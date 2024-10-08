# Exercise 7: Failure Modes & Limits of Deep Learning

## Getting this repo

If you are working from the super repository https://github.com/dlmbl/DL-MBL-2024, don't forget to update this submodule:
```
git submodule update --init --recursive 07_failure_modes
```

## Goal
In Exercise 7: Failure Modes and Limits of Deep Learning, we delve into understanding the limits and failure modes of neural networks in the context of image classification. By tampering with image datasets and introducing extra visual information, the exercise mimics real-world scenarios where data collection inconsistencies can corrupt datasets.

The exercise examines how neural networks handle local and global data corruptions. We will reason about a classification network's performance through confusion matrices, and use tools like Integrated Gradients to identify areas of an image that influence classification decisions. Additionally, the exercise explores how denoising networks cope with domain changes by training a UNet model on noisy MNIST data and testing it on both similar and different datasets like FashionMNIST. 

Through these activities, participants are encouraged to think deeply about neural network behavior, discuss their findings in groups, and reflect on the impact of dataset inconsistencies on model performance and robustness. By exploring failure modes, participants gain insights into the internal workings of neural networks and learn how to diagnose and mitigate issues that are common in real-world scendarios.


## Methodology
1. **Data Preparation**:
   - **Load Data**: Load the MNIST dataset for training and testing.
   - **Create Tainted Dataset**: Make copies of the original datasets to create tainted versions.
   - **Local Corruption**: Add a white pixel to images of the digit '7' in the tainted dataset.
   - **Global Corruption**: Add a grid texture to images of the digit '4' in the tainted dataset.

2. **Visualization**:
   - Visualize examples of corrupted images to understand the modifications made.

3. **Train Neural Networks**:
   - **Define Models**: Create a dense neural network model for classification.
   - **Initialize Models**: Set up clean and tainted models with identical initial weights for comparison.
   - **Load Data**: Initialize data loaders for clean and tainted datasets.
   - **Train Models**: Train both models on their respective datasets (clean and tainted).

4. **Evaluate Performance**:
   - **Loss Visualization**: Plot training loss for both clean and tainted models to compare performance.
   - **Confusion Matrix**: Generate confusion matrices to analyze model performance on clean and tainted test sets.

5. **Interpret Results**:
   - **Integrated Gradients**: Use the Integrated Gradients method to visualize the important regions of the images that influence the model's decisions.
   - **Visualize Attention**: Compare the attention maps for clean and tainted models on specific images.

6. **Denoising Task**:
   - **Add Noise**: Introduce noise to MNIST images to create a dataset for training a denoising model.
   - **Define UNet Model**: Use a UNet model architecture for denoising.
   - **Train Denoising Model**: Train the UNet model on the noisy MNIST dataset.
   - **Evaluate on FashionMNIST**: Apply the trained denoising model to FashionMNIST data to see how it performs on unseen data.

### Technology Used

1. **Programming Language**:
   - Python

2. **Libraries and Tools**:
   - **PyTorch**: For building and training neural networks.
     - `torchvision`: For loading and transforming datasets.
     - `torch.nn`: For defining neural network models.
     - `torch.optim`: For optimization algorithms.
   - **Matplotlib**: For visualizing images and plotting graphs.
   - **Scipy**: For image manipulation (e.g., adding textures).
   - **Numpy**: For numerical operations.
   - **TQDM**: For displaying progress bars during training.
   - **Captum**: For implementing Integrated Gradients and other interpretability methods.
   - **Seaborn**: For creating confusion matrices.

3. **Datasets**:
   - **MNIST**: Handwritten digit dataset for training and testing classification models.
   - **FashionMNIST**: Fashion item dataset for evaluating the denoising model on different data.

## Setup
Please run the setup script to create the environment for this exercise and download data.

```bash
source setup.sh
```

When you are ready to start the exercise, open the `exercise.ipynb` file in VSCode
and select the `07-failure-modes` kernel
