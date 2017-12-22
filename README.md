### Light-Field-Depth-Estimation
#### *Light Filed Depth Estimation using cGAN*

#### Data
* 600 Light Field Images from [DDFF 12-scene](http://hazirbas.com/datasets/ddff12scene/)

#### Method
* Conditional GAN using pix2pix (Tensorflow)
* Fed 5 focal image stacks + LSTM embeddings

#### Run
* Download any checkpoint from [here](https://drive.google.com/open?id=1zV6wRKh1gkEIZg687LAFQbOlwnzK-YIH)
* Download manual cropped data from [here](https://drive.google.com/open?id=1js-jLasmGDigc0pNgbc4INcmUn6Mp7Fu)
* `python main.py --phase train --dataset_name scene12_v3_400`

#### Acknowledgments
Code borrows heavily from [pix2pix-tensorflow](https://github.com/yenchenlin/pix2pix-tensorflow). Thanks for Yen-Chen!
