# 0. Introduction

# 1. Project Setup
## AW SageMaker
1. Navigate to project root directory:
    - ``` cd SageMaker/predictive_maintenance```
2. Activate virtual environment:
    - `source venv/bin/activate`
3. Add virtual environment as a kernel 
    - `ipython kernel install --user --name=masterarbei`

## Local 
1. Navigate to project root directory
2. Create virtual environment: 
    - `python3 -m venv venv_cm`
3. Activate virtual environment:
    - `source venv_cm/bin/activat`
4. Install ipykernel to add virtual environment to jupyter lab: 
    - `pip install ipykerne`
    - `ipython kernel install --user --name=condition_monitoring`
5. Install libraries from requirements.txt
6. Add path of own packages to virtual environment
    - Create a file with the name _sitecustomize.py_ in this directory: `venv_cm/lib/python3.6/site-packages/`
    - Insert this code to the file: 
    
```pyhton
import os
import sys

sys.path.append("<Pfad>/src/ai/models/")
sys.path.append("<Pfad>/src/ai/utils/")
```

# 2. Use of the Code


