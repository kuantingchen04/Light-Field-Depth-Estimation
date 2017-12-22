### Light-Field-Depth-Estimation
#### *Light Filed Depth Estimation using cGAN*

#### Data
* 600 Light Field Images from [DDFF 12-scene](http://hazirbas.com/datasets/ddff12scene/)

#### Method
* pix2pix in tensorflow: Revised from [here](https://github.com/yenchenlin/pix2pix-tensorflow)   
* focal stacks + LSTM embeddings


#### Run
* Download a checkpoint from [here](https://drive.google.com/open?id=1zV6wRKh1gkEIZg687LAFQbOlwnzK-YIH)
* Also, download data from [here](https://drive.google.com/open?id=1js-jLasmGDigc0pNgbc4INcmUn6Mp7Fu)
* `python main.py --phase train --dataset_name scene12_v3_400`
