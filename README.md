# $VP^2Net$：Visual Perception-Inspired Network for Exploring the Causes of Drivers' Attention Shift

Authors: Chunyu Zhao, Tao Deng*, Pengcheng Du, Wenbo Liu, Yi Huang, Fei Yan

​	[note] * Corresponding Author.

​	[note] We will release our complete code after the paper is **accepted** ✔️! Please look forward to it.🕓

## 📰 News

**[2024.01.15]** 🎈Everyone can download the ADED dataset, the data is stored on BaiduNetdisk, from this [link][1].👈

[1]: http://www.google.com	"ADED-Dataset"

**[2024.03.17]** 🎈We propose $VP^2Net$, which is a visual dual perception-inspired network for exploring the causes of driver’s attention shifts.

**[2024.06.06]** 🎈We will submit the article to ***TITS*** (IEEE **T**ransactions on **I**ntelligent **T**tansportation **S**ystems).😃

## ✨Model

<img src="pic\model.jpg" style="zoom:40%;" />

## 💻Dataset

Through the process of re-labeling, we obtain an attention-based driving event dataset (ADED) consisting of 1101 videos. The dataset provides semantic annotation for each driving video, and the semantic information contains annotations of driving event categories and driving event time windows. The driving event categories contain six categories, which are Driving Normally (DN), Avoiding Pedestrian Crossing (ACP) Waiting for Vehicle Ahead (WVA), Waiting for Red Light (SRL), Stop Sign Stopping (SSS) and Avoiding Lane Changing Vehicle (ALC).

<img src="pic\dataset.jpg" alt="dataset" style="zoom:20%;" />

## 🚀 Quantitative Analysis

| Model         | Year | Pub.  | DN         | ACP        | WVA        | SRL        | SSS        | ALC        | Acc        | F1         | mAP        |
| ------------- | ---- | ----- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| I3D           | 2017 | CVPR  | 0.4912     | 0.0674     | 0.7583     | 0.7033     | 0.6967     | 0.1842     | 0.6558     | 0.4551     | 0.4712     |
| ResNet MC 18  | 2018 | CVPR  | 0.1053     | 0.0787     | 0.7854     | 0.7833     | 0.9300     | 0.1930     | 0.6974     | 0.4718     | 0.5465     |
| ResNet (2+1)D | 2018 | CVPR  | 0.4123     | 0.3258     | 0.9125     | 0.9400     | 0.8833     | 0.1754     | 0.8093     | 0.6105     | 0.6554     |
| SlowOnly      | 2019 | ICCV  | 0.0614     | 0.6222     | 0.9385     | 0.8067     | 0.9367     | 0.2105     | 0.8045     | 0.6065     | 0.6737     |
| SlowFast      | 2019 | ICCV  | 0.2018     | 0.2556     | 0.8311     | 0.4667     | 0.8700     | 0.3421     | 0.6835     | 0.4885     | 0.4808     |
| CSN           | 2019 | ICCV  | 0.2544     | 0.2333     | 0.9051     | 0.9200     | 0.9100     | 0.3070     | 0.8002     | 0.6001     | 0.5661     |
| TPN           | 2020 | CVPR  | 0.1842     | 0.5393     | 0.8583     | 0.8133     | **0.9500** | **0.4123** | 0.7826     | 0.6417     | 0.6537     |
| Timesformer   | 2021 | ICML  | 0.0526     | 0.1556     | 0.8561     | 0.8433     | 0.9433     | 0.0614     | 0.7373     | 0.4741     | 0.5246     |
| ViViT         | 2021 | ICCV  | 0.1316     | 0.1333     | 0.8229     | 0.6800     | 0.8633     | 0.1491     | 0.6906     | 0.4584     | 0.4964     |
| VideoMAE      | 2022 | NIPS  | 0.3421     | 0.3111     | 0.8646     | 0.8200     | 0.9300     | 0.3596     | 0.7790     | 0.6071     | 0.5920     |
| Uniformer     | 2023 | TPAMI | 0.1579     | **0.6556** | 0.9167     | 0.8433     | 0.9133     | 0.3070     | 0.8088     | 0.6417     | 0.6137     |
| videoFocalNet | 2023 | ICCV  | 0.2456     | 0.6444     | 0.9469     | 0.7633     | 0.8933     | 0.3333     | 0.8147     | 0.6508     | 0.6363     |
| DERNet        | 2023 | IJCNN | 0.4649     | 0.2022     | 0.9510     | 0.9367     | 0.9233     | 0.3070     | 0.8402     | 0.6493     | 0.6887     |
| $VP^2Net$     | 2024 | -     | **0.5263** | 0.3889     | **0.9531** | **0.9700** | 0.9267     | 0.2632     | **0.8568** | **0.6920** | **0.7252** |

## 🚀Visualisation of intermediate results

The visualization of the intermediate features. (a) represents the original image, (b) depicts the driving scene feature $F_{SIE}$, (c) shows the attention information $\hat{S}$, (d) displays the perception-enhanced information $\hat{S}^*$, and (e) illustrates the attention-encoded information $F_{APE}$.

<img src="pic\feature_show.jpg" alt="feature_show_1" style="zoom:20%;" />

## 📝Model Zoo

We give the weights obtained by training in the paper. Includes weights from ablation experiments. These weights may be able to be used as your pre-training weights, reducing the time required for learning.

| Model                | pth           | Model                     | pth           |
| -------------------- | ------------- | ------------------------- | ------------- |
| $VP^2Net$            | [BaiduYun][2] | $VP^2Net$ w/ Add          | [BaiduYun][6] |
| $VP^2Net$ w/ offline | [BaiduYun][3] | $VP^2Net$ w/ Add-Multiply | [BaiduYun][7] |
| $VP^2Net$ w/ online  | [BaiduYun][2] | $VP^2Net$ w/ DER-Net      | [BaiduYun][8] |
| $VP^2Net$ w/o APE    | [BaiduYun][5] | $VP^2Net$ w/ (2D)         | [BaiduYun][9] |
| $VP^2Net$ w/o PEM    | [BaiduYun][5] | $VP^2Net$ w/ (3D)         | [BaiduYun][2] |

[2]: https://pan.baidu.com/s/1YgmhD9Nq8AAkEKrXYsMTDA?pwd=V2PN "V2PNet"
[3]: https://pan.baidu.com/s/1WdVunAkihHX9DZPGDga38Q?pwd=V2PN "offline"
[5]: https://pan.baidu.com/s/1lG9Cn7l8TjcA9C28Ukq7xQ?pwd=V2PN "w/o pem"
[6]: https://pan.baidu.com/s/18TFOhjXw-FqdNFkaLjjLDQ?pwd=V2PN "w/ add"
[7]: https://pan.baidu.com/s/1OD_xuD2X0OOgGNk13RnEog?pwd=V2PN "w/ add-multiply"
[8]:https://pan.baidu.com/s/1Wb_mTrpTx0A5LqPLXvRenA?pwd=V2PN "DER-Net"
[9]: https://pan.baidu.com/s/1QaHok0aCX94tcCAKUldm3Q?pwd=V2PN "w/ 2d"



## 💖Support the Project

Thanks to the open-source video action detection models (ViViT, VideoMAE) at [huggingface🤗][10]  for supporting this paper.

[10]: https://huggingface.curated.co/	"huggingface"

## 📄Cite

Welcome to our work ! 

