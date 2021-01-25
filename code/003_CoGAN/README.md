# DATASET
Datasets are brain MRI images consisting of 4 scans.

4 scans were co-registered(paired data).

We used **300 "T1 & T2" images**.

Dataset can be downloaded in [[link]](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2). (Task01_BrainTumour.tar)

    data
    ├── train
    │     ├──── BRATS_001.nii.gz
    │     ├──── BRATS_002.nii.gz
    │     ├──── BRATS_003.nii.gz
    │     ├──── ...

# Training
Input Size : 256x256

Batch Size : 4

Number of epoch : 400

Latent Vector Size : 1000

**Generator Weight Sharing : All layer except last conv layer**

**Discriminator Weight Sharing : None**


# Result

| T1                            | T2                            |
| ----------------------------- | ----------------------------- |
| ![](result/T1.gif)   | ![](result/T2.gif)   |


# Run Example
```
$ cd code/003_CoGAN
$ python3 cogan.py
```


