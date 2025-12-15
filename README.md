# Exploring the Role of Local Centrality in Graph Neural Networks for Point Cloud Classification

_This is my final thesis project in the advanced course "Pattern Recognition and Machine Learning"_

## Abstract
Graph Neural Networks (GNNs) have demonstrated strong capabilities in pointcloud processing, yet face challenges in capturing local structural properties efficiently. 
In this work, we propose a lightweight approach that enhances GNNbased point cloud classification through local centrality measures. 
Our method introduces an efficient local centrality computation mechanism and a novel feature integration strategy that effectively combines structural and geometric information.
Through theoretical analysis and experimental validation, we demonstrate that our approach achieves competitive classification accuracy (89.8%) while significantly reducing model complexity (0.27M parameters). 
The results show that carefully designed structural feature integration can enhance model performance without introducing substantial computational overhead. 
Our work contributes to the understanding of structural feature learning in geometric deep learning and provides insights for developing efficient point cloud processing architectures.

## 1 Introduction
### 1.1 Background
Point cloud processing has emerged as a fundamental challenge in computer vision, with applications spanning autonomous driving, robotics, and augmented reality. The inherent characteristics of point
clouds—their irregular structure, unordered nature, and variable density—pose unique challenges for deep learning approaches. While Graph Neural Networks (GNNs) have shown promising results in handling such unstructured data, they face critical limitations in feature degradation and
computational efficiency. Recent theoretical advances have revealed that the feature degradation in deep GNNs, commonly known as over-smoothing, occurs primarily due to the feature transformation mechanism rather than
message propagation. This insight suggests that addressing feature transformation strategies may be
more crucial than modifying message passing schemes. Furthermore, the computational complexity
of current GNN-based methods often scales poorly with input size, making them impractical for
dense point clouds.

### 1.2 Related Work
#### 1.2.1 Point Cloud Processing

The evolution of deep learning approaches for point cloud processing has witnessed several paradigm
shifts. Early attempts focused on transforming irregular point clouds into regular 3D voxel grids
(Maturana & Scherer, 2015) or multi-view projections (Su et al., 2015), but these methods either
suffered from resolution limitations or lost critical geometric information. A breakthrough came
with PointNet (Qi et al., 2017a), which pioneered direct point cloud processing through permutationinvariant operations. However, its point-wise processing nature limited its ability to capture local
geometric relationships.
Subsequent research addressed these limitations through hierarchical architectures. PointNet++ (Qi
et al., 2017b) introduced multi-scale grouping and sampling strategies, enabling better local feature
learning. The field further evolved with graph-based approaches, notably DGCNN (Wang et al.,
2019), which explicitly modeled point relationships through dynamic graph construction.


