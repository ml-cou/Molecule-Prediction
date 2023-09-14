## Installation

### Step 1: Clone the Repository

```bash
$ git clone https://github.com/ml-cou/Molecule-Prediction.git
$ cd ml-cou/Molecule-Prediction
```

### Step 2: Install Dependencies

```bash
$ pip install -r requirements.txt
```

### Step 3: Creating Output Directories

Before running the project, it's essential to create two specific output directories: `Output/Node_Features` and `Output/Adjacency_Matrix`. These directories will be used to store the project's output files.

You can create these directories using the following command line instructions:

```bash
$ mkdir -p Output/Node_Features
$ mkdir -p Output/Adjacency_Matrix

The -p flag ensures that parent directories (Output in this case) will be created if they do not exist. After executing these commands, your directory structure should look like this:

Your_Project_Directory/
├── Output/
│   ├── Node_Features/
│   └── Adjacency_Matrix/
│
├── other_project_files.py
├── ...
```

## Step 4: Running the Project


### Using a Python Script
Run the project using a Python script (if applicable):

```bash
$ python main.py
```
