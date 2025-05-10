# Physics-informed KAN (Kolmogorov-Arnold) PointNet

**Physics-informed KAN PointNet: Deep learning for simultaneous solutions to inverse problems in incompressible flow on numerous irregular geometries**

**Software by:** Ali Kashefi (kashefi@stanford.edu) 

**Citation** <br>
If you use the code, please cite the following article: <br>

**[Kolmogorov-Arnold PointNet: Deep learning for prediction of fluid fields on irregular geometries](https://doi.org/10.1016/j.cma.2025.117888)**

    @article{kashefi2024KANpointnetCFD,
      title = {Kolmogorov–Arnold PointNet: Deep learning for prediction of fluid fields on irregular geometries},
      journal = {Computer Methods in Applied Mechanics and Engineering},
      volume = {439},
      pages = {117888},
      year = {2025},
      doi = {https://doi.org/10.1016/j.cma.2025.117888},
      author = {Ali Kashefi}}

**Abstract** <be>

Kolmogorov-Arnold Networks ...

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

