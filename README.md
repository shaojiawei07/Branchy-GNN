# Branchy-GNN

This code is for the [paper](https://arxiv.org/abs/2006.02166): "Branchy-GNN: a Device-Edge Co-Inference Framework for Efficient Point Cloud Processing", which is submitted to ICASSP2021.



## Framework

We propose a branchy structure for GNN-based point cloud classification to speedup edge inference. We adopt branch structures for early exiting the main branch to reduce the on-device computational cost and introduce joint source-channel coding (JSCC) to reduce the communication overhead.

In the experiment, we have four exit points.

Note that the main branch in the framework is based on [DGCNN](https://github.com/WangYueFt/dgcnn).

![avatar](./Branchy_GNN_Framework.png)



### Dependency

```
Pytorch 1.6.0
```



### Dataset

```
ModelNet40
```





### How to run

1. Pretrain a DGCNN model based on the [code](https://github.com/WangYueFt/dgcnn/tree/master/pytorch) or download from [here](https://github.com/WangYueFt/dgcnn/tree/master/pytorch/pretrained). (``./pretrained/model.1024.t7``)
2. Train the branch network by ``python edge_main.py --num_p=1024 --use_sgd=True --model EXIT1``. (Note that ``--model`` could be ``EXIT1``, ``EXIT2``, ``EXIT3``, and ``EXIT4``.)

