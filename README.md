# EUMP
Embedding-based Urban Modeling &amp; Planning

# Main Figs Reproduce 

This repository contains scripts to reproduce the figures for our project. 

## Prerequisites
- **Conda** (recommended) or **pip**

---

## 🚀 Step-by-Step Guide

Follow these instructions to set up the environment and generate the figures.

### 1. Clone the Repository
Clone this project to your local machine and enter the project directory:
```bash
git clone https://github.com/jiwk-Fighoh/EUMP
cd EUMP
```

### 2. Create and Activate Conda Environment
Create a dedicated virtual environment to avoid dependency conflicts:

```bash
# Create the environment with Python 3.9
conda create -n fig_env python=3.9 -y

# Activate the environment
conda activate fig_env
```

### 3.Install Dependencies
Install all required libraries using the provided requirements file:

```Bash
pip install -r requirements.txt
```

### 4. Running the Scripts
Each figure script relies on local data files within its directory. To avoid Path Errors, you must navigate into the specific sub-folder before running the script.

```bash
# Example: Generate Figure 2

# Change directory to the specific figure folder

cd main_figs_v1/fig2

# Execute the plotting script

python fig2.py
```

Note: Follow the same logic for fig3, fig4, and fig5. The output images will be saved in their respective folders.
