# DATASET
We used 1,340 normal chest X-ray images.

Dataset can be downloaded in [[link]](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

    data/
    ├── img-0001.jpeg
    ├── img-0002.jpeg
    ├── img-0003.jpeg
    ├── ...

# Training
Input Size : 512x512

Latent Vector Size : 512**2

Batch Size : 64

Number of epoch : 200


# Result
<p align="left">
    <img src="result/result_image.gif" width="256"\>
</p>


# Run Example
```
$ cd code/1_GAN
$ python3 gan.py
```
