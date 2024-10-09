Point cloud is an important geometric representation that can be used to represent 3D data. It is very close to the raw output of laser sensors, often used to scan volumetric data into digital form. Unlike images which are represented in regular grids of pixels, point clouds are irregular and do not present explicit order making it difficult to use convolutional layers, given that such layers excel in learning from local patterns. Inspired by recent results of applying Self-attention mechanism on deep neural networks, this paper proposes the use of such technique to improve representations learned from 3D geometrics data. We provide experiments by adding Self-attention layers in state-of-the-art models, such as PointNet and DGCNN trained for the task of classification in the widely used ModelNet40 dataset. Our experiments show that self-attention improves the accuracy, while making the algorithm more stable in terms of loss and accuracy values during training.


# Acknowledgement

The Implementation of DGCNN and PointNet are based on https://github.com/WangYueFt/dgcnn/tree/master/pytorch
and re-implementation from: https://github.com/andersonnascimento/PointCloudAttentionNet
