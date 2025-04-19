# DAT-Dual-Ascent-Inspired-Transformer-for-Compressed-Sensing

This repository is for DAT introduced in the following paper：Lin, R., Shen, Y., & Chen, Y. (2025). Dual-Ascent-Inspired Transformer for Compressed Sensing. Sensors, 25(7), 2157. [pdf](https://doi.org/10.3390/s25072157)

## :memo: Abstract

Deep learning has revolutionized image compressed sensing (CS) by enabling lightweight models that achieve high-quality reconstruction with low latency. However, most deep neural network-based CS models are pre-trained for specific compression ratios (CS ratios), limiting their flexibility compared to traditional iterative algorithms. To address this limitation, we propose the Dual-Ascent-Inspired Transformer (DAT), a novel architecture that maintains stable performance across different compression ratios with minimal training costs. DAT’s design incorporates the mathematical properties of the dual ascent method (DAM), leading to accelerated training convergence. The architecture features an innovative asymmetric primal–dual space at each iteration layer, enabling dimension-specific operations that balance reconstruction quality with computational efficiency. We also optimize the Cross Attention module through parameter sharing, effectively reducing its training complexity. Experimental results demonstrate DAT’s superior performance in two key aspects: First, during early-stage training (within 10 epochs), DAT consistently outperforms existing methods across multiple CS ratios (10%, 30%, and 50%). Notably, DAT achieves comparable PSNR to the ISTA-Net+ baseline within just one epoch, while competing methods require significantly more training time. Second, DAT exhibits enhanced robustness to variations in initial learning rates, as evidenced by loss function analysis during training.

## :hammer_and_wrench: Network Architecture
![Network](/Figs/Sampling.png)

![Network](/Figs/Architecture.png)

![Network](/Figs/Network.png)

## :package: Requirements
- Python == 3.9.16
- Pytorch == 2.0.1+cu118

## :test_tube: Results
![Network](/Figs/Comparison.png)

## :eyes: Datasets
- Train data: [400 images](https://drive.google.com/file/d/1hELlT70R56KIM0VFMAylmRZ5n2IuOxiz/view?usp=sharing) from BSD500 dataset
- Test data: Set11, [Urban100](https://drive.google.com/file/d/1cmYjEJlR2S6cqrPq8oQm3tF9lO2sU0gV/view?usp=sharing)

## :computer: Command
### Train
To reproduce the early training results from the paper, you should start training with your desired sensing rate (0.1/0.3/0.5) using the following command:
`python train_DAT.py --sensing_rate 0.1/0.3/0.5`
Note that the model will not automatically stop after 10 epochs — you will need to stop the training manually.
### Test
`python test_DAT.py --sensing_rate 0.1/0.3/0.5 --epoch_num 1/10`

## :e-mail: Contact
If you have any question, please email `lin-rui398@g.ecc.u-tokyo.ac.jp`.

## :hugs: Acknowledgements
This code is built on [OCTUF](https://github.com/songjiechong/OCTUF?tab=readme-ov-file). We thank the authors for sharing their codes. 

