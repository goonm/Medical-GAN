# DATASET
We used 1,340 normal chest X-ray images.

Dataset can be downloaded in [[link]](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

    data/
    ├── img-0001.jpeg
    ├── img-0002.jpeg
    ├── img-0003.jpeg
    ├── ...

# Training
Input Size : 256x256

Latent Vector Size : 1000

Batch Size : 4

Number of epoch : 300


# Result
<p align="left">
    <img src="result/result_image.gif" width="256"\>
</p>


# Run Example
```
$ cd code/002_DCGAN
$ python3 dcgan.py
```
