# Physics-informed KAN (Kolmogorov-Arnold) PointNet
![pic](./pi_kan_pointnet.png)

**Physics-informed KAN PointNet: Deep learning for simultaneous solutions to inverse problems in incompressible flow on numerous irregular geometries**

**Software by:** Ali Kashefi (kashefi@stanford.edu) 

**Citation** <br>
If you use the code, please cite the following article: <br>

**[Physics-informed KAN PointNet: Deep learning for simultaneous solutions to inverse problems in incompressible flow on numerous irregular geometries](https://arxiv.org/abs/2504.06327)**

    @article{kashefi2025PhysicsInformedKANpointnet,
      title={Physics-informed KAN PointNet: Deep learning for simultaneous solutions to inverse problems in incompressible flow on numerous irregular 
      geometries},
      author={Kashefi, Ali and Mukerji, Tapan},
      journal={arXiv preprint arXiv:2504.06327},
      year={2025}}

**Abstract** <be>

Kolmogorov-Arnold Networks (KANs) have gained attention as a promising alternative to traditional Multilayer Perceptrons (MLPs) for deep learning applications in computational physics, especially within the framework of physics-informed neural networks (PINNs). Physics-informed Kolmogorov-Arnold Networks (PIKANs) and their variants have been introduced and evaluated to solve inverse problems. However, similar to PINNs, current versions of PIKANs are limited to obtaining solutions for a single computational domain per training run; consequently, a new geometry requires retraining the model from scratch. Physics-informed PointNet (PIPN) was introduced to address this limitation for PINNs. In this work, we introduce physics-informed Kolmogorov-Arnold PointNet (PI-KAN-PointNet) to extend this capability to PIKANs. PI-KAN-PointNet enables the simultaneous solution of an inverse problem over multiple irregular geometries within a single training run, reducing computational costs. We construct KANs using Jacobi polynomials and investigate their performance by considering Jacobi polynomials of different degrees, as well as special cases such as Legendre polynomials, Chebyshev polynomials of the first and second kinds, and Gegenbauer polynomials, in terms of both computational cost and prediction accuracy. As a benchmark test case, we consider natural convection in a square enclosure with a cylinder, where the cylinder's shape varies across a dataset of 135 geometries. We compare the performance of PI-KAN-PointNet with that of PIPN (i.e., physics-informed PointNet with MLPs) and observe that, with approximately an equal number of trainable parameters and similar computational cost, PI-KAN-PointNet provides more accurate predictions. Finally, we explore the combination of KAN and MLP in constructing a physics-informed PointNet. Our findings indicate that a physics-informed PointNet model employing MLP layers as the encoder and KAN layers as the decoder represents the optimal configuration among all models investigated.

**Installation** <be>
This guide will help you set up the environment required to run the code. Follow the steps below to install the necessary dependencies.

**Step 1: Download and Install Miniconda**

1. Visit the [Miniconda installation page](https://docs.conda.io/en/latest/miniconda.html) and download the installer that matches your operating system.
2. Follow the instructions to install Miniconda.

**Step 2: Create a Conda Environment**

After installing Miniconda, create a new environment:

```bash
conda create --name myenv python=3.8
```

Activate the environment:

```bash
conda activate myenv
```

**Step 3: Install PyTorch**

Install PyTorch with CUDA 11.8 support:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Step 4: Install Additional Dependencies**

Install the required Python libraries:

```bash
pip3 install numpy matplotlib torchsummary
```

**Step 5: Download the Data** <be>

Use the following batch command to download the dataset (a NumPy array, approximately 42MB in size).

```bash
wget https://web.stanford.edu/~kashefi/data/HeatTransferData.npy
```

**Questions?** <br>
If you have any questions or need assistance, please do not hesitate to contact Ali Kashefi (kashefi@stanford.edu) via email. 

**About the Author** <br>
Please see the author's website: [Ali Kashefi](https://web.stanford.edu/~kashefi/) 

