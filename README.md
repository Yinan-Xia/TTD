# Test-Time Dynamic Image Fusion
This is the official implementation of Test-Time Dynamic Image Fusion(NeurIPS 2024) by Bing Cao, Yinan Xia, Yi Ding, Changqing Zhang, and Qinghua Hu.

<p align="center">
<img src="https://github.com/Yinan-Xia/TTD/blob/main/frame.png">
</p>

## Abstract
The inherent challenge of image fusion lies in capturing the correlation of multi-source images and comprehensively integrating effective information from different sources. Most existing techniques fail to perform dynamic image fusion while notably lacking theoretical guarantees, leading to potential deployment risks in this field. Is it possible to conduct dynamic image fusion with a clear theoretical justification? In this paper, we give our solution from a generalization perspective. We proceed to reveal the generalized form of image fusion and derive a new test-time dynamic image fusion paradigm. It provably reduces the upper bound of generalization error. Specifically, we decompose the fused image into multiple components corresponding to its source data. The decomposed components represent the effective information from the source data, thus the gap between them reflects the Relative Dominability(RD) of the uni-source data in constructing the fusion image. Theoretically, we prove that the key to reducing generalization error hinges on the negative correlation between the RD-based fusion weight and the uni-source reconstruction loss. Intuitively, RD dynamically highlights the dominant regions of each source and can be naturally converted to the corresponding fusion weight, achieving robust results. Extensive experiments and discussions with in-depth analysis on multiple benchmarks confirm our findings and superiority.

## Citation
```
@inproceedings{cao2024test-time,
  title={Test-Time Dynamic Image Fusion},
  author={Cao, Bing and Xia, Yinan and Ding, Yi and Zhang, Changqing and Hu, Qinghua},
  booktitle={Advances in neural information processing systems},
  year={2024}
}
