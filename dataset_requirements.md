# run split_dataset.py to split the dataset into train, valid, test in ration 3:1:2
Labels -> masks

data/
├── train/
│   ├── Images/
│   └── Labels/
├── val/
│   ├── Images/
│   └── Labels/
└── test/
    ├── Images/
    └── Labels/

# to see the logs written in tensorboard -> give the path to write the run summary such as runs/segmentation (main.py)
# run this command in terminal to see the stored run
tensorboard --logdir=runs/segmentation/


'''
# To run on linux cluster
'''
Run train.sh in remote cluster (Linux system)
Run train.ps1 in locally (Windows) 

# To freeze the requirements.txt

#Keep only the used libraries
pip install pipreqs

#run  
pipreqs . --force

$ pip freeze > requirements.txt




