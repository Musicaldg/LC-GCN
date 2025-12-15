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

#### 1.2.2 GNN Over-smoothing
The over-smoothing phenomenon in GNNs has emerged as a critical challenge, particularly in deep
architectures. Theoretical analyses (Li et al., 2018; Chen et al., 2020) have revealed that repeated
message passing causes node features to converge to similar values, effectively losing discriminative
information. This issue becomes more pronounced in point cloud processing due to the geometric
nature of the data and the importance of preserving local structural information.
Recent theoretical developments (Zhou et al., 2020) have shifted focus from message propagation to
feature transformation mechanisms as the primary source of over-smoothing. This insight suggests
that addressing feature transformation strategies may be more crucial than modifying message passing
schemes.

#### 1.2.3 Network Centrality in Graph Learning
Network centrality measures have long been recognized as powerful tools for understanding graph
structures (Borgatti, 2005). While traditional centrality metrics focus on global network properties,
recent research has explored efficient local variants (You et al., 2020), making them more applicable
to deep learning contexts. The adaptation of centrality measures to GNNs has shown promise in
tasks requiring structural understanding, though their application to point cloud processing remains
under-explored.

### 1.3 Contributions
Our work advances the field of point cloud processing through several key contributions:
First, we propose an efficient local centrality computation mechanism that captures structural information in point clouds without requiring expensive global computations. This approach enables
the integration of structural features while maintaining computational efficiency, addressing a key
challenge in geometric deep learning.
Second, we develop a novel feature integration strategy that effectively combines centrality information with geometric features through an adaptive weighting mechanism. This design helps prevent
feature degradation while enhancing the model’s ability to capture local structural patterns.
Finally, we demonstrate that our lightweight architecture achieves competitive performance (89.8%
accuracy) with significantly fewer parameters (0.27M) compared to standard architectures like PointNet (3.5M parameters). These results show that carefully designed structural feature learning can
maintain high performance while reducing model complexity.

## 2 Method
### 2.1 Overview
We propose a centrality-enhanced architecture that integrates local structural properties into the
dynamic graph convolution framework for point cloud classification. Our approach is motivated
by the observation that structural importance information can effectively complement geometric
features while preventing over-smoothing. The architecture, illustrated in Figure 1, consists of three
key components that work in concert to achieve this goal: (1) an efficient local centrality module
that computes structural importance scores based on both geometric relationships and feature space
similarities, (2) a novel structural edge convolution layer that effectively combines the computed
centrality information with geometric features, and (3) a multi-scale feature extraction pathway with
residual connections.
The architecture progressively processes point clouds through feature extraction (N×64) and multiple
edge convolution layers with increasing receptive fields (r=0.1, 0.2, 0.4). This hierarchical design
enables the model to capture multi-scale structural patterns while maintaining computational efficiency. The residual connections help prevent feature degradation during deep propagation, while
the increasing receptive fields allow the model to gradually expand its structural awareness.
This design enables our model to:

* Capture multi-scale structural patterns through hierarchical feature learning
* Maintain geometric fidelity while incorporating structural importance
* Prevent over-smoothing through adaptive feature aggregation
* Scale efficiently to large point clouds through localized computations
