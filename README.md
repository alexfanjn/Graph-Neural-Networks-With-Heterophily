# Awesome Resources on Graph Neural Networks With Heterophily

This repository contains the relevant resources on graph neural network (GNN) considering heterophily. 

It's worth noting that the **heterophily** we consider here is not the same as **heterogeneity**. Heterogeneity is more related to the node type difference such as the user and item nodes in recommender systems, but heterophily is more like the feature or label difference between the neighbors under the nodes with the same type. Traditional GNNs usually assume that similar nodes (features/classes) are connected together, but the "opposites attract" phenomenon also widely exists in general graphs.

If you find anything incorrect, please let me know. Thanks!

<!--[[Paper](https://arxiv.org/abs/2101.00797)], [[Code](https://github.com/bdy9527/FAGCN)]-->

## Papers

### 2024

- Consistency Training with Learnable Data Augmentation for Graph Anomaly Detection with Limited Supervision, ICLR, [[Paper](https://openreview.net/forum?id=elMKXvhhQ9)], [Code]
- PolyGCL: Graph Contrastive Learning via Learnable Spectral Polynomial Filters, ICLR, [[Paper](https://openreview.net/forum?id=y21ZO6M86t)], [Code]
- Partitioning Message Passing for Graph Fraud Detection, ICLR, [[Paper](https://openreview.net/forum?id=tEgrUrUuwA)], [Code]
- NetInfoF Framework: Measuring and Exploiting Network Usable Information, ICLR, [[Paper](https://openreview.net/forum?id=KY8ZNcljVU)], [Code]
- Polynormer: Polynomial-Expressive Graph Transformer in Linear Time, ICLR, [[Paper](https://openreview.net/forum?id=hmv1LpNfXa)], [[Code]](https://github.com/cornell-zhang/Polynormer)
- From Latent Graph to Latent Topology Inference: Differentiable Cell Complex Module, ICLR, [[Paper](https://openreview.net/forum?id=0JsRZEGZ7L)], [Code]
- Mixture of Weak & Strong Experts on Graphs, ICLR, [[Paper](https://openreview.net/forum?id=wYvuY60SdD)], [Code]
- HoloNets: Spectral Convolutions do extend to Directed Graphs, ICLR, [[Paper](https://openreview.net/forum?id=EhmEwfavOW)], [Code]
- VCR-Graphormer: A Mini-batch Graph Transformer via Virtual Connections, ICLR, [[Paper](https://openreview.net/forum?id=SUUrkC3STJ)], [[Code]](https://github.com/DongqiFu/VCR-Graphormer)
- PC-Conv: Unifying Homophily and Heterophily with Two-fold Filtering, AAAI, [[Paper](https://arxiv.org/abs/2312.14438)], [[Code](https://github.com/uestclbh/PC-Conv)]
- Refining Latent Homophilic Structures over Heterophilic Graphs for Robust Graph Convolution Networks, AAAI, [[Paper](https://arxiv.org/abs/2312.16418)], [Code]
- Homophily-Related: Adaptive Hybrid Graph Filter for Multi-View Graph Clustering, AAAI, [[Paper](https://arxiv.org/abs/2401.02682)], [Code]
- PhoGAD: Graph-based Anomaly Behavior Detection with Persistent Homology Optimization, WSDM, [[Paper](https://arxiv.org/abs/2401.10547)], [Code]
- Challenging Low Homophily in Social Recommendation, WWW, [[Paper](https://arxiv.org/abs/2401.14606)], [Code]
- Cross-Space Adaptive Filter: Integrating Graph Topology and Node Attributes for Alleviating the Over-smoothing Problem, WWW, [[Paper](https://arxiv.org/abs/2401.14876)], [[Code]](https://github.com/huangzichun/Cross-Space-Adaptive-Filter)
- Disambiguated Node Classification with Graph Neural Networks, WWW, [[Paper](https://arxiv.org/abs/2402.08824)], [Code]
- Permutation Equivariant Graph Framelets for Heterophilous Graph Learning, TNNLS, [[Paper](https://arxiv.org/abs/2306.04265)], [[Code]](https://github.com/zrgcityu/PEGFAN)
- Breaking the Entanglement of Homophily and Heterophily in Semi-supervised Node Classification, ICDE, [[Paper](https://arxiv.org/abs/2312.04111)], [Code]
- Lying Graph Convolution: Learning to Lie for Node Classification Tasks, IJCNN, [[Paper](https://arxiv.org/abs/2405.01247)], [[Code]](https://github.com/danielecastellana22/lying-graph-convolution)
- SeBot: Structural Entropy Guided Multi-View Contrastive Learning for Social Bot Detection, KDD, [[Paper](https://arxiv.org/abs/2405.11225)], [[Code]](https://github.com/846468230/SEBot)
- How Universal Polynomial Bases Enhance Spectral Graph Neural Networks: Heterophily, Over-smoothing, and Over-squashing, ICML, [[Paper](https://arxiv.org/abs/2405.12474)], [[Code]](https://github.com/kkhuang81/UniFilter)
- Rethinking Spectral Graph Neural Networks with Spatially Adaptive Filtering, arXiv, [[Paper](https://arxiv.org/abs/2401.09071)], [Code]
- Understanding Heterophily for Graph Neural Networks, arXiv, [[Paper](https://arxiv.org/abs/2401.09125)], [Code]
- Universally Robust Graph Neural Networks by Preserving Neighbor Similarity, arXiv, [[Paper](https://arxiv.org/abs/2401.09754)], [Code]
- ***[Survey Paper]*** Learning from Graphs with Heterophily: Progress and Future, arXiv, [[Paper](https://arxiv.org/abs/2401.09769)], [[Code](https://github.com/gongchenghua/Awesome-Survey-Graphs-with-Heterophily)]
- Design Your Own Universe: A Physics-Informed Agnostic Method for Enhancing Graph Neural Networks, arXiv, [[Paper](https://arxiv.org/abs/2401.14580)], [Code]
- Hypergraph Neural Networks through the Lens of Message Passing: A Common Perspective to Homophily and Architecture Design, arXiv, [[Paper](https://arxiv.org/abs/2310.07684v2)], [Code]
- Heterophily-Aware Fair Recommendation using Graph Convolutional Networks, arXiv, [[Paper](https://arxiv.org/abs/2402.03365)], [[Code]](https://github.com/NematGH/HetroFair)
- Feature Distribution on Graph Topology Mediates the Effect of Graph Convolution: Homophily Perspective, arXiv, [[Paper](https://arxiv.org/abs/2402.04621)], [Code]
- Flexible infinite-width graph convolutional networks and the importance of representation learning, arXiv, [[Paper](https://arxiv.org/abs/2402.06525)], [Code]
- Link Prediction under Heterophily: A Physics-Inspired Graph Neural Network Approach, arXiv, [[Paper](https://arxiv.org/abs/2402.14802)], [Code]
- Representation Learning on Heterophilic Graph with Directional Neighborhood Attention, arXiv, [[Paper](https://arxiv.org/abs/2403.01475)], [Code]
- Self-Attention Empowered Graph Convolutional Network for Structure Learning and Node Embedding, arXiv, [[Paper](https://arxiv.org/abs/2403.03465)], [[Code]](https://github.com/mengyingjiang/GCN-SA)
- Robust Graph Structure Learning under Heterophily, arXiv, [[Paper](https://arxiv.org/abs/2403.03659)], [Code]
- Provable Filter for Real-world Graph Clustering, arXiv, [[Paper](https://arxiv.org/abs/2403.03666)], [Code]
- Simplified PCNet with Robustness, arXiv, [[Paper](https://arxiv.org/abs/2403.03676)], [Code]
- Optimizing Polynomial Graph Filters: A Novel Adaptive Krylov Subspace Approach, arXiv, [[Paper](https://arxiv.org/abs/2403.07954)], [[Code]](https://github.com/kkhuang81/AdaptKry)
- Generation is better than Modification: Combating High Class Homophily Variance in Graph Anomaly Detection, arXiv, [[Paper](https://arxiv.org/abs/2403.10339)], [Code]
- Discovering Invariant Neighborhood Patterns for Heterophilic Graphs, arXiv, [[Paper](https://arxiv.org/abs/2403.10572)], [Code]
- Distinguishing Neighborhood Representations Through Reverse Process of GNNs for Heterophilic Graphs, arXiv, [[Paper](https://arxiv.org/abs/2403.10543)], [Code]
- Learn from Heterophily: Heterophilous Information-enhanced Graph Neural Network, arXiv, [[Paper](https://arxiv.org/abs/2403.17351)], [Code]
- Spectral Graph Pruning Against Over-Squashing and Over-Smoothing, arXiv, [[Paper](https://arxiv.org/abs/2404.04612)], [Code]
- All Nodes are created Not Equal: Node-Specific Layer Aggregation and Filtration for GNN, arXiv, [[Paper](https://arxiv.org/abs/2405.07892)], [[Code]](https://github.com/wslong20/NoSAF-NoSAF-D)
- LOGIN: A Large Language Model Consulted Graph Neural Network Training Framework, arXiv, [[Paper](https://arxiv.org/abs/2405.13902)], [[Code]](https://github.com/QiaoYRan/LOGIN)
- Learning Graph Filters for Spectral GNNs via Newton Interpolation, arXiv, [[Paper](https://arxiv.org/abs/2310.10064)], [Code]

### 2023
- Beyond Smoothing: Unsupervised Graph Representation Learning with Edge Heterophily Discriminating, AAAI, [[Paper](https://arxiv.org/abs/2211.14065)], [[Code](https://github.com/yixinliu233/GREET)]
- Restructuring Graph for Higher Homophily via Adaptive Spectral Clustering, AAAI, [[Paper](https://arxiv.org/abs/2206.02386)], [[Code](https://github.com/seanli3/graph_restructure)]
- 2-hop Neighbor Class Similarity (2NCS): A graph structural metric indicative of graph neural network performance, AAAI-W, [[Paper](https://arxiv.org/abs/2212.13202)], [Code]
- Ordered GNN: Ordering Message Passing to Deal with Heterophily and Over-smoothing, ICLR, [[Paper](https://openreview.net/forum?id=wKPmPBHSnT6)], [[Code](https://github.com/LUMIA-Group/OrderedGNN)]
- Gradient Gating for Deep Multi-Rate Learning on Graphs, ICLR, [[Paper](https://openreview.net/forum?id=JpRExTbl1-)], [Code]
- ACMP: Allen-Cahn Message Passing with Attractive and Repulsive Forces for Graph Neural Networks, ICLR, [[Paper](https://openreview.net/forum?id=4fZc_79Lrqs)], [[Code](https://github.com/ykiiiiii/ACMP)]
- A Critical Look at Evaluation of GNNs Under Heterophily: Are We Really Making Progress?, ICLR, [[Paper](https://openreview.net/forum?id=tJbbQfw-5wv)], [[Code](https://github.com/yandex-research/heterophilous-graphs)]
- GReTo: Remedying dynamic graph topology-task discordance via target homophily, ICLR, [[Paper](https://openreview.net/forum?id=8duT3mi_5n)], [[Code](https://github.com/zzyy0929/ICLR23-GReTo)]
- Projections of Model Spaces for Latent Graph Inference, ICLR-W, [[Paper](https://arxiv.org/abs/2303.11754v3)], [Code]
- Addressing Heterophily in Graph Anomaly Detection: A Perspective of Graph Spectrum, WWW,  [[Paper](https://hexiangnan.github.io/papers/www23-graphAD.pdf)], [[Code](https://github.com/blacksingular/GHRN)]
- Homophily-oriented Heterogeneous Graph Rewiring, WWW, [[Paper](https://arxiv.org/abs/2302.06299)], [Code]
- Auto-HeG: Automated Graph Neural Network on Heterophilic Graphs, WWW, [[Paper](https://arxiv.org/abs/2302.12357)], [Code]
- Label Information Enhanced Fraud Detection against Low Homophily in Graphs, WWW,  [[Paper](https://arxiv.org/abs/2302.10407)], [[Code](https://github.com/Orion-wyc/GAGA)]
- SE-GSL: A General and Effective Graph Structure Learning Framework through Structural Entropy Optimization, WWW,  [[Paper](https://arxiv.org/abs/2303.09778)], [[Code](https://github.com/RingBDStack/SE-GSL)]
- Graph Neural Networks with Diverse Spectral Filtering, WWW,  [[Paper](https://arxiv.org/abs/2312.09041)], [[Code](https://github.com/jingweio/DSF)]
- GCNH: A Simple Method For Representation Learning on Heterophilous Graphs, IJCNN, [[Paper](https://arxiv.org/abs/2304.10896)], [[Code](https://github.com/SmartData-Polito/GCNH)]
- Exploiting Neighbor Effect: Conv-Agnostic GNNs Framework for Graphs with Heterophily, TNNLS, [[Paper](https://arxiv.org/abs/2203.11200)], [[Code]](https://github.com/JC-202/CAGNN)
- Multi-View Graph Representation Learning Beyond Homophily, TKDD, [[Paper](https://arxiv.org/abs/2304.07509)], [[Code]](https://github.com/G-AILab/MVGE)
- Spatial Heterophily Aware Graph Neural Networks, KDD, [[Paper](https://arxiv.org/abs/2306.12139)], [[Code]](https://github.com/PaddlePaddle/PaddleSpatial/tree/main/research/SHGNN)
- Node Classification Beyond Homophily: Towards a General Solution, KDD, [[Paper](https://dl.acm.org/doi/abs/10.1145/3580305.3599446)], [[Code]](https://github.com/pricexu/ALT)
- Examining the Effects of Degree Distribution and Homophily in Graph Learning Models, KDD-W, [[Paper](https://arxiv.org/abs/2307.08881)], [[Code]](https://github.com/google-research/graphworld)
- Finding the Missing-half: Graph Complementary Learning for Homophily-prone and Heterophily-prone Graphs, ICML, [[Paper](https://arxiv.org/abs/2306.07608)], [[Code]](https://github.com/zyzisastudyreallyhardguy/GOAL-Graph-Complementary-Learning)
- Contrastive Learning Meets Homophily: Two Birds with One Stone, ICML, [[Paper](https://openreview.net/forum?id=YIcb3pR8ld)], [Code]
- Beyond Homophily: Reconstructing Structure for Graph-agnostic Clustering, ICML, [[Paper](https://arxiv.org/abs/2305.02931)], [[Code]](https://github.com/Panern/DGCN)
- GOAT: A Global Transformer on Large-scale Graphs, ICML, [[Paper](https://openreview.net/forum?id=z29R0uMiF3v)], [[Code]](https://github.com/devnkong/GOAT)
- Towards Deep Attention in Graph Neural Networks: Problems and Remedies, ICML, [[Paper](https://arxiv.org/abs/2306.02376)], [[Code]](https://github.com/syleeheal/AERO-GNN)
- Half-Hop: A graph upsampling approach for slowing down message passing, ICML, [[Paper](https://openreview.net/forum?id=lXczFIwQkv)], [[Code]](https://github.com/nerdslab/halfhop)
- Evolving Computation Graphs, ICML-W, [[Paper](https://arxiv.org/abs/2306.12943)], [Code]
- ***[Survey Paper]*** Heterophily and Graph Neural Networks: Past, Present and Future, Data Engineering, [[Paper](http://sites.computer.org/debull/A23june/p10.pdf)], [Code]
- Homophily-Enhanced Self-Supervision for Graph Structure Learning: Insights and Directions, TNNLS, [[Paper](https://ieeexplore.ieee.org/abstract/document/10106110)], [[Code]](https://github.com/LirongWu/Homophily-Enhanced-Self-supervision)
- LSGNN: Towards General Graph Neural Network in Node Classification by Local Similarity, IJCAI, [[Paper](https://arxiv.org/abs/2305.04225)], [[Code]](https://github.com/draym28/LSGNN)
- Graph Neural Convection-Diffusion with Heterophily, IJCAI, [[Paper](https://arxiv.org/abs/2305.16780)], [[Code]](https://github.com/zknus/Graph-Diffusion-CDE)
- Imbalanced Node Classification Beyond Homophilic Assumption, IJCAI, [[Paper](https://arxiv.org/abs/2304.14635)], [Code]
- Beyond Homophily: Robust Graph Anomaly Detection via Neural Sparsification, IJCAI, [[Paper](https://www.ijcai.org/proceedings/2023/234)], [Code]
- Taming over-smoothing representation on heterophilic graphs, Inf. Sci., [[Paper](https://www.sciencedirect.com/science/article/pii/S0020025523010484)], [[Code]](https://github.com/KaiGuo20/LE-GNN)
- Homophily-enhanced Structure Learning for Graph Clustering, CIKM, [[Paper](https://arxiv.org/abs/2308.05309)], [[Code]](https://github.com/galogm/HoLe)
- Neighborhood Homophily-based Graph Convolutional Network, CIKM, [[Paper](https://arxiv.org/abs/2301.09851v2)], [[Code]](https://github.com/rockcor/NHGCN)
- HOVER: Homophilic Oversampling via Edge Removal for Class-Imbalanced Bot Detection on Graphs, CIKM, [[Paper](https://dl.acm.org/doi/10.1145/3583780.3615264)], [Code]
- Self-supervised Learning and Graph Classification under Heterophily, CIKM, [[Paper](https://arxiv.org/abs/2306.08469)], [Code]
- MUSE: Multi-View Contrastive Learning for Heterophilic Graphs, CIKM, [[Paper](https://arxiv.org/abs/2307.16026)], [Code]
- SplitGNN: Spectral Graph Neural Network for Fraud Detection against Heterophily, CIKM, [[Paper](https://dl.acm.org/doi/abs/10.1145/3583780.3615067)], [[Code]](https://github.com/Split-GNN/SplitGNN)
- Signed attention based graph neural network for graphs with heterophily, Neurocomputing, [[Paper](https://www.sciencedirect.com/science/article/pii/S0925231223008548)], [Code]
-  Improving the Homophily of Heterophilic Graphs for Semi-Supervised Node Classification, ICME, [[Paper](https://www.computer.org/csdl/proceedings-article/icme/2023/689100b865/1PTMICbZ1wQ)], [Code]
- Imbalanced node classification with Graph Neural Networks: A unified approach leveraging homophily and label information, Appl. Soft Comput., [[Paper](https://www.sciencedirect.com/science/article/pii/S1568494623010037)], [Code]
- Leveraging Free Labels to Power up Heterophilic Graph Learning in Weakly-Supervised Settings: An Empirical Study, ECML-PKDD, [[Paper](https://www.springerprofessional.de/leveraging-free-labels-to-power-up-heterophilic-graph-learning-i/26051948)], [Code]
- Learning to Augment Graph Structure for both Homophily and Heterophily, ECML-PKDD, [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-43418-1_1)], [[Code]](https://github.com/LirongWu/L2A)
- Single-Pass Contrastive Learning Can Work for Both Homophilic and Heterophilic Graph?, TMLR, [[Paper](https://openreview.net/forum?id=XE0cIoi-sZ1)], [[Code]](https://github.com/haonan3/SPGCL)
- On Performance Discrepancies Across Local Homophily Levels in Graph Neural Networks, LoG, [[Paper](https://arxiv.org/abs/2306.05557)], [Code]
- UniG-Encoder: A Universal Feature Encoder for Graph and Hypergraph Node Classification, PR, [[Paper](https://www.sciencedirect.com/science/article/pii/S0031320323008129?via%3Dihub)], [[Code](https://github.com/MinhZou/UniG-Encoder)]
- Simple and Asymmetric Graph Contrastive Learning without Augmentations, NeurIPS, [[Paper](https://arxiv.org/abs/2310.18884)], [[Code]](https://github.com/tengxiao1/GraphACL)
- When Do Graph Neural Networks Help with Node Classification: Investigating the Homophily Principle on Node Distinguishability, NeurIPS, [[Paper](https://arxiv.org/abs/2304.14274)], [[Code]](https://github.com/SitaoLuan/When-Do-GNNs-Help)
- Characterizing Graph Datasets for Node Classification: Homophily-Heterophily Dichotomy and Beyond, NeurIPS, [[Paper](https://arxiv.org/abs/2209.06177)], [Code]
- Truncated Affinity Maximization: One-class Homophily Modeling for Graph Anomaly Detection, NeurIPS, [[Paper](https://arxiv.org/abs/2306.00006)], [[Code](https://github.com/mala-lab/TAM-master/)]
- LD2: Scalable Heterophilous Graph Neural Network with Decoupled Embeddings, NeurIPS, [[Paper](https://openreview.net/forum?id=7zkFc9TGKz)], [[Code](https://github.com/gdmnl/LD2)]
- Predicting Global Label Relationship Matrix for Graph Neural Networks under Heterophily, NeurIPS, [[Paper](https://openreview.net/forum?id=nBFMCyEi0j)], [[Code](https://github.com/Jinx-byebye/LRGNN/)]
- From Trainable Negative Depth to Edge Heterophily in Graphs, NeurIPS, [[Paper](https://openreview.net/forum?id=p8lowHbuv8)], [Code]
- Alleviating Structural Distribution Shift in Graph Anomaly Detection, WSDM, [[Paper](https://arxiv.org/abs/2401.14155)], [[Code]](https://github.com/blacksingular/wsdm_GDN)
- SlenderGNN: Accurate, Robust, and Interpretable GNN, and the Reasons for its Success, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=lMgFRIILVB)], [Code]
- Simple Spectral Graph Convolution from an Optimization Perspective, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=cZM4iZmxzR7)], [Code]
- The Impact of Neighborhood Distribution in Graph Convolutional Networks, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=XUqTyU9VlWp)], [Code]
- Wide Graph Neural Network, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=Ih0fKoIUyEh)], [Code]
- Are Graph Attention Networks Attentive Enough? Rethinking Graph Attention by Capturing Homophily and Heterophily, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=Xk10fyKR8G)], [Code]
- From ChebNet to ChebGibbsNet, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=2a5Ru3JtNe0)], [Code]
- ProtoGNN: Prototype-Assisted Message Passing Framework for Non-Homophilous Graphs, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=LeZ39Gkwbi0)], [Code]
- Low-Rank Graph Neural Networks Inspired by the Weak-balance Theory in Social Networks, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=ufCQZeAMZzf)], [Code]
- Graph Contrastive Learning Under Heterophily: Utilizing Graph Filters to Generate Graph Views, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=NzcUQuhEGef)], [Code]
- ReD-GCN: Revisit the Depth of Graph Convolutional Network, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=tMg5hKRiW-2)], [Code]
- Graph Neural Networks as Gradient Flows: Understanding Graph Convolutions via Energy, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=M3GzgrA7U4)], [Code]
- Causally-guided Regularization of Graph Attention improves Generalizability, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=U086TJFWy4p)], [Code]
- GReTo: Remedying dynamic graph topology-task discordance via target homophily, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=8duT3mi_5n)], [Code]
- Homophily modulates double descent generalization in graph convolution networks, arXiv, [[Paper](https://arxiv.org/abs/2212.13069v2)], [Code]
- Semi-Supervised Classification with Graph Convolutional Kernel Machines, arXiv, [[Paper](https://arxiv.org/abs/2301.13764)], [Code]
- A Graph Neural Network with Negative Message Passing for Graph Coloring, arXiv, [[Paper](https://arxiv.org/abs/2301.11164)], [Code]
- Is Signed Message Essential for Graph Neural Networks?, arXiv, [[Paper](https://arxiv.org/abs/2301.08918)], [Code]
- Attending to Graph Transformers, arXiv, [[Paper](https://arxiv.org/abs/2302.04181)], [[Code](https://github.com/luis-mueller/probing-graph-transformers)]
- Heterophily-Aware Graph Attention Network, arXiv, [[Paper](https://arxiv.org/abs/2302.03228)], [Code]
- Steering Graph Neural Networks with Pinning Control, arXiv, [[Paper](https://arxiv.org/abs/2303.01265)], [Code]
- Contrastive Learning under Heterophily, arXiv, [[Paper](https://arxiv.org/abs/2303.06344)], [Code]
- Graph Positional Encoding via Random Feature Propagation, arXiv, [[Paper](https://arxiv.org/abs/2303.02918)], [Code]
- GPatcher: A Simple and Adaptive MLP Model for Alleviating Graph Heterophily, arXiv, [[Paper](https://arxiv.org/abs/2306.14340)], [Code]
- PathMLP: Smooth Path Towards High-order Homophily, arXiv, [[Paper](https://arxiv.org/abs/2306.13532)], [Code]
- Demystifying Structural Disparity in Graph Neural Networks: Can One Size Fit All?, arXiv, [[Paper](https://arxiv.org/abs/2306.01323)], [Code]
- Edge Directionality Improves Learning on Heterophilic Graphs, arXiv, [[Paper](https://arxiv.org/abs/2305.10498)], [Code]
- Addressing Heterophily in Node Classification with Graph Echo State Networks, arXiv, [[Paper](https://arxiv.org/abs/2305.08233)], [[Code](https://github.com/dtortorella/addressing-heterophily-gesn)]
- SIMGA: A Simple and Effective Heterophilous Graph Neural Network with Efficient Global Aggregation, arXiv, [[Paper](https://arxiv.org/abs/2305.09958)], [Code]
- A Fractional Graph Laplacian Approach to Oversmoothing, arXiv, [[Paper](https://arxiv.org/abs/2305.13084)], [[Code](https://github.com/RPaolino/fLode)]
- From Latent Graph to Latent Topology Inference: Differentiable Cell Complex Module, arXiv, [[Paper](https://arxiv.org/abs/2305.16174)], [Code]
- Self-attention Dual Embedding for Graphs with Heterophily, arXiv, [[Paper](https://arxiv.org/abs/2305.18385)], [Code]
- Explaining and Adapting Graph Conditional Shift, arXiv, [[Paper](https://arxiv.org/abs/2306.03256)], [Code]
- Heterophily-aware Social Bot Detection with Supervised Contrastive Learning, arXiv, [[Paper](https://arxiv.org/abs/2306.07478)], [Code]
- Diffusion-Jump GNNs: Homophiliation via Learnable Metric Filters, arXiv, [[Paper](https://arxiv.org/abs/2306.16976)], [Code]
- HOFA: Twitter Bot Detection with Homophily-Oriented Augmentation and Frequency Adaptive Attention, arXiv, [[Paper](https://arxiv.org/abs/2306.12870)], [Code]
- Heterophily-aware Social Bot Detection with Supervised Contrastive Learning, arXiv, [[Paper](https://arxiv.org/abs/2306.07478)], [Code]
- Extended Graph Assessment Metrics for Graph Neural Networks, arXiv, [[Paper](https://arxiv.org/abs/2307.10112)], [Code]
- Automated Polynomial Filter Learning for Graph Neural Networks, arXiv, [[Paper](https://arxiv.org/abs/2307.07956)], [Code]
- Frameless Graph Knowledge Distillation, arXiv, [[Paper](https://arxiv.org/abs/2307.06631)], [[Code](https://github.com/dshi3553usyd/Frameless_Graph_Distillation)]
- QDC: Quantum Diffusion Convolution Kernels on Graphs, arXiv, [[Paper](https://arxiv.org/abs/2307.11234)], [Code]
- Contrastive Learning for Non-Local Graphs with Multi-Resolution Structural Views, arXiv, [[Paper](https://arxiv.org/abs/2308.10077)], [Code]
- Domain-adaptive Message Passing Graph Neural Network, arXiv, [[Paper](https://arxiv.org/abs/2308.16470)], [[Code](https://github.com/shenxiaocam/DM_GNN)]
- Unifying over-smoothing and over-squashing in graph neural networks: A physics informed approach and beyond, arXiv, [[Paper](https://arxiv.org/abs/2309.02769)], [Code]
- Bregman Graph Neural Network, arXiv, [[Paper](https://arxiv.org/abs/2309.06645)], [[Code](https://github.com/jiayuzhai1207/BregmanGNN)]
- HoloNets: Spectral Convolutions do extend to Directed Graphs, arXiv, [[Paper](https://arxiv.org/abs/2310.02232)], [Code]
- SignGT: Signed Attention-based Graph Transformer for Graph Representation Learning, arXiv, [[Paper](https://arxiv.org/abs/2310.11025)], [Code]
- Hetero2Net: Heterophily-aware Representation Learning on Heterogenerous Graphs, arXiv, [[Paper](https://arxiv.org/abs/2310.11664)], [Code]
- Augment with Care: Enhancing Graph Contrastive Learning with Selective Spectrum Perturbation, arXiv, [[Paper](https://arxiv.org/abs/2310.13845)], [Code]
- Enhancing Graph Neural Networks with Structure-Based Prompt, arXiv, [[Paper](https://arxiv.org/abs/2310.17394)], [Code]
- Resist Label Noise with PGM for Graph Neural Networks, arXiv, [[Paper](https://arxiv.org/abs/2311.02116)], [Code]
- Prioritized Propagation in Graph Neural Networks, arXiv, [[Paper](https://arxiv.org/abs/2311.02832)], [Code]
- Dirichlet Energy Enhancement of Graph Neural Networks by Framelet Augmentation, arXiv, [[Paper](https://arxiv.org/abs/2311.05767)], [Code]
- Revisiting Graph-based Fraud Detection in Sight of Heterophily and Spectrum, arXiv, [[Paper](https://arxiv.org/abs/2312.06441)], [Code]
- CAT: A Causally Graph Attention Network for Trimming Heterophilic Graph, arXiv, [[Paper](https://arxiv.org/abs/2312.08672)], [[Code](https://github.com/GeoX-Lab/CAT)]
- GraphRARE: Reinforcement Learning Enhanced Graph Neural Network with Relative Entropy, arXiv, [[Paper](https://arxiv.org/abs/2312.09708)], [Code]

### 2022

- Block Modeling-Guided Graph Convolutional Neural Networks, AAAI, [[Paper](https://arxiv.org/abs/2112.13507)], [[Code](https://github.com/hedongxiao-tju/BM-GCN)]
- Powerful Graph Convolutioal Networks with Adaptive Propagation Mechanism for Homophily and Heterophily, AAAI, [[Paper](https://arxiv.org/abs/2112.13562)], [[Code](https://github.com/hedongxiao-tju/HOG-GCN)]
- Deformable Graph Convolutional Networks, AAAI, [[Paper](https://arxiv.org/abs/2112.14438)], [Code]
- Graph Pointer Neural Networks, AAAI, [[Paper](https://arxiv.org/abs/2110.00973)], [Code]
- Is Homophily A Necessity for Graph Neural Networks?, ICLR, [[Paper](https://arxiv.org/abs/2106.06134)], [[Code]](https://openreview.net/attachment?id=ucASPPD9GKN&name=supplementary_material)
- Designing the Topology of Graph Neural Networks: A Novel Feature Fusion Perspective, WWW, [[Paper](https://arxiv.org/abs/2112.14531)], [[Code](https://github.com/AutoML-Research/F2GNN)]
- GBK-GNN: Gated Bi-Kernel Graph Neural Networks for Modeling Both Homophily and Heterophily, WWW, [[Paper](https://arxiv.org/abs/2110.15777)], [[Code](https://github.com/xzh0u/gbk-gnn)]
- Meta-Weight Graph Neural Network: Push the Limits Beyond Global Homophily, WWW, [[Paper](https://arxiv.org/abs/2203.10280)], [Code]
- Understanding and Improving Graph Injection Attack by Promoting Unnoticeability, ICLR, [[Paper](https://arxiv.org/abs/2202.08057)], [[Code](https://github.com/lfhase/gia-hao)]
- Neural Link Prediction with Walk Pooling, ICLR, [[Paper](https://arxiv.org/abs/2110.04375)], [[Code](https://github.com/DaDaCheng/WalkPooling)]
- Finding Global Homophily in Graph Neural Networks When Meeting Heterophily, ICML, [[Paper](https://arxiv.org/abs/2205.07308)], [[Code](https://github.com/recklessronan/glognn)]
- How Powerful are Spectral Graph Neural Networks, ICML, [[Paper](https://arxiv.org/abs/2205.11172)], [Code]
- Optimization-Induced Graph Implicit Nonlinear Diffusion, ICML, [[Paper](https://arxiv.org/abs/2206.14418)], [[Code](https://github.com/7qchen/GIND)]
- Sheaf Neural Networks with Connection Laplacians, ICML-W, [[Paper](https://arxiv.org/abs/2206.08702)], [Code]
- How does Heterophily Impact Robustness of Graph Neural Networks? Theoretical Connections and Practical Implications, KDD, [[Paper](https://arxiv.org/abs/2106.07767)], [[Code](https://github.com/GemsLab/HeteRobust)]
- On Graph Neural Network Fairness in the Presence of Heterophilous Neighborhoods, KDD-W, [[Paper](https://arxiv.org/abs/2207.04376)], [Code]
- NCGNN: Node-Level Capsule Graph Neural Network for Semisupervised Classification, TNNLS, [[Paper](https://arxiv.org/abs/2012.03476)], [Code]
- Beyond Low-pass Filtering: Graph Convolutional Networks with Automatic Filtering, TKDE, [[Paper](https://arxiv.org/abs/2107.04755)], [[Code](https://github.com/nnzhan/AutoGCN)]
- Beyond Homophily: Structure-aware Path Aggregation Graph Neural Network, IJCAI, [[Paper](https://www.ijcai.org/proceedings/2022/310)], [[Code](https://github.com/zjunet/PathNet)]
- RAW-GNN: RAndom Walk Aggregation based Graph Neural Network, IJCAI, [[Paper](https://arxiv.org/abs/2206.13953)], [[Code]](https://github.com/jindi-tju/RAWGNN)
- EvenNet: Ignoring Odd-Hop Neighbors Improves Robustness of Graph Neural Networks, NeurIPS, [[Paper](https://arxiv.org/abs/2205.13892)], [Code]
- Simplified Graph Convolution with Heterophily, NeurIPS, [[Paper](https://arxiv.org/abs/2202.04139)], [Code]
- Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs, NeurIPS, [[Paper](https://arxiv.org/abs/2202.04579)], [[Code]](https://github.com/twitter-research/neural-sheaf-diffusion)
- Revisiting Heterophily For Graph Neural Networks, NeurIPS, [[Paper]](https://arxiv.org/abs/2210.07606), [Code]
- From Local to Global: Spectral-Inspired Graph Neural Networks, NeurIPS-W, [[Paper](https://arxiv.org/abs/2209.12054)], [[Code]](https://github.com/nhuang37/spectral-inspired-gnn)
- Complete the Missing Half: Augmenting Aggregation Filtering with Diversification for Graph Convolutional Networks, NeurIPS-W, [[Paper](https://arxiv.org/abs/2008.08844v4)], [Code]
- Beyond Homophily with Graph Echo State Networks, ESANN, [[Paper](https://arxiv.org/abs/2210.15731)], [Code]
- Memory-based Message Passing: Decoupling the Message for Propogation from Discrimination, ICASSP, [[Paper](https://arxiv.org/abs/2202.00423)], [[Code]](https://github.com/JC-202/MMP)
- Label-Wise Message Passing Graph Neural Network on Heterophilic Graphs, LoG, [[Paper](https://arxiv.org/abs/2110.08128)], [[Code]](https://github.com/EnyanDai/LWGCN)
- Leave Graphs Alone: Addressing Over-Squashing without Rewiring, LoG, [[Paper](https://openreview.net/forum?id=vEbUaN9Z2V8)], [Code]
- Global-Local Graph Neural Networks for Node-Classification, LoG, [[Paper](https://openreview.net/forum?id=YCgwkDo56q)], [Code]
- GARNET: Reduced-Rank Topology Learning for Robust and Scalable Graph Neural Networks, LoG, [[Paper](https://arxiv.org/abs/2201.12741)], [Code]
- DiffWire: Inductive Graph Rewiring via the Lovász Bound, LoG, [[Paper](https://proceedings.mlr.press/v198/arnaiz-rodri-guez22a.html)], [[Code]](https://github.com/AdrianArnaiz/DiffWire)
- ***[Tutorial]*** Graph Rewiring: From Theory to Applications in Fairness, LoG, [[Link]](https://ellisalicante.org/tutorials/GraphRewiring), [[Code]](https://github.com/ellisalicante/GraphRewiring-Tutorial)
- Unsupervised Network Embedding Beyond Homophily, TMLR, [[Paper](https://arxiv.org/abs/2203.10866v3)], [[Code](https://github.com/zhiqiangzhongddu/Selene)]
- Improving Your Graph Neural Networks: A High-Frequency Booster, ICDM-W, [[Paper](https://arxiv.org/abs/2210.08251)], [[Code]](https://github.com/sajqavril/Complement-Laplacian-Regularization)
- GSN: A Universal Graph Neural Network Inspired by Spring Network, arXiv, [[Paper](https://arxiv.org/abs/2201.12994)], [Code]
- Graph Decoupling Attention Markov Networks for Semi-supervised Graph Node Classification, arXiv, [[Paper](https://arxiv.org/abs/2104.13718)], [Code]
- Relational Graph Neural Network Design via Progressive Neural Architecture Search, arXiv, [[Paper](https://arxiv.org/abs/2105.14490)], [Code]
- Graph Neural Network with Curriculum Learning for Imbalanced Node Classification, arXiv, [[Paper](https://arxiv.org/abs/2202.02529)], [Code]
- ***[Survey Paper]*** Graph Neural Networks for Graphs with Heterophily: A Survey, arXIv, [[Paper](https://arxiv.org/abs/2202.07082)], [Code]
- When Does A Spectral Graph Neural Network Fail in Node Classification?, arXIv, [[Paper](https://arxiv.org/abs/2202.07902)], [Code]
- Graph Representation Learning Beyond Node and Homophily, arXiv, [[Paper](https://arxiv.org/abs/2203.01564)], [[Code](https://github.com/syvail/PairE-Graph-Representation-Learning-Beyond-Node-and-Homophily)]
- Incorporating Heterophily into Graph Neural Networks for Graph Classification, arXiv, [[Paper](https://arxiv.org/abs/2203.07678)], [[Code](https://github.com/yeweiysh/IHGNN)]
- Augmentation-Free Graph Contrastive Learning, arXiv, [[Paper](https://arxiv.org/abs/2204.04874)], [Code]
- Simplifying Node Classification on Heterophilous Graphs with Compatible Label Propagation, arXiv, [[Paper](https://arxiv.org/abs/2205.09389)], [Code]
- Revisiting the Role of Heterophily in Graph Representation Learning: An Edge Classification Perspective, arXiv, [[Paper](https://arxiv.org/abs/2205.11322)], [Code]
- ES-GNN: Generalizing Graph Neural Networks Beyond Homophily with Edge Splitting, arXiv, [[Paper](https://arxiv.org/abs/2205.13700)], [Code]
- Restructuring Graph for Higher Homophily via Learnable Spectral Clustering, arXiv, [[Paper](https://arxiv.org/abs/2206.02386)], [Code]
- Decoupled Self-supervised Learning for Non-Homophilous Graphs, arXiv, [[Paper](https://arxiv.org/abs/2206.03601)], [Code]
- Graph Neural Networks as Gradient Flows, arXiv, [[Paper](https://arxiv.org/abs/2206.10991)], [Code]
- What Do Graph Convolutional Neural Networks Learn?, arXiv, [[Paper](https://arxiv.org/abs/2207.01839)], [Code]
- Deformable Graph Transformer, arXiv, [[Paper](https://arxiv.org/abs/2206.14337)], [Code]
- Demystifying Graph Convolution with a Simple Concatenation, arXiv, [[Paper](https://arxiv.org/abs/2207.12931)], [Code]
- Link Prediction on Heterophilic Graphs via Disentangled Representation Learning, arXiv, [[Paper](https://arxiv.org/abs/2208.01820)], [[Code](https://github.com/sjz5202/DisenLink)]
- Graph Polynomial Convolution Models for Node Classification of Non-Homophilous Graphs, arXiv, [[Paper](https://arxiv.org/abs/2209.05020)], [[Code](https://github.com/kishanwn/GPCN)]
- Make Heterophily Graphs Better Fit GNN: A Graph Rewiring Approach, arXiv, [[Paper](https://arxiv.org/abs/2209.08264)], [Code]
- Break the Wall Between Homophily and Heterophily for Graph Representation Learning, arXiv, [[Paper](https://arxiv.org/abs/2210.05382)], [Code]
- GPNet: Simplifying Graph Neural Networks via Multi-channel Geometric Polynomials, arXiv, [[Paper](https://arxiv.org/abs/2209.15454)], [Code]
- HP-GMN: Graph Memory Networks for Heterophilous Graphs, arXiv, [[Paper](https://arxiv.org/abs/2210.08195)], [[Code]](https://github.com/junjie-xu/hp-gmn)
- When Do We Need GNN for Node Classification?, arXiv, [[Paper](https://arxiv.org/abs/2210.16979)], [Code]
- Revisiting Heterophily in Graph Convolution Networks by Learning Representations Across Topological and Feature Spaces, arXiv, [[Paper](https://arxiv.org/abs/2211.00565v2)], [[Code]](https://github.com/SresthTosniwal17/HETGCN)
- GLINKX: A Scalable Unified Framework For Homophilous and Heterophilous Graphs, arXiv, [[Paper](https://arxiv.org/abs/2211.00550)], [Code]
- Clenshaw Graph Neural Networks, arXiv, [[Paper](https://arxiv.org/abs/2210.16508)], [Code]
- Unifying Label-inputted Graph Neural Networks with Deep Equilibrium Models, arXiv, [[Paper](https://arxiv.org/abs/2211.10629)], [[Code]](https://github.com/cf020031308/GQN)
- Neighborhood Convolutional Network: A New Paradigm of Graph Neural Networks for Node Classification, arXiv, [[Paper](https://arxiv.org/abs/2211.07845)], [Code]
- Enhancing Intra-class Information Extraction for Heterophilous Graphs: One Neural Architecture Search Approach, arXiv, [[Paper](https://arxiv.org/abs/2211.10990)], [Code]
- Transductive Kernels for Gaussian Processes on Graphs, arXiv, [[Paper](https://arxiv.org/abs/2211.15322)], [Code]
- Flip Initial Features: Generalization of Neural Networks for Semi-supervised Node Classification, arXiv, [[Paper](https://arxiv.org/abs/2211.15081v2)], [Code]
- VR-GNN: Variational Relation Vector Graph Neural Network for Modeling both Homophily and Heterophily, arXiv, [[Paper](https://arxiv.org/abs/2211.14523)], [Code]
- GREAD: Graph Neural Reaction-Diffusion Equations, arXiv, [[Paper](https://arxiv.org/abs/2211.14208)], [Code]
- Node-oriented Spectral Filtering for Graph Neural Networks, arXiv, [[Paper](https://arxiv.org/abs/2212.03654)], [Code]


### 2021

- AdaGNN: Graph Neural Networks with Adaptive Frequency Response Filter, CIKM, [[Paper](https://arxiv.org/abs/2104.12840)], [[Code](https://github.com/yushundong/AdaGNN)]
- Tree Decomposed Graph Neural Network, CIKM, [[Paper](https://arxiv.org/abs/2108.11022)], [[Code](https://github.com/YuWVandy/TDGNN)]
- Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods, NeurIPS, [[Paper](https://arxiv.org/abs/2110.14446)], [[Code](https://github.com/cuai/non-homophily-large-scale)]
- Diverse Message Passing for Attribute with Heterophily, NeurIPS, [[Paper](https://openreview.net/forum?id=4jPVcKEYpSZ)], [Code]
- Universal Graph Convolutional Networks, NeurIPS, [[Paper](https://papers.nips.cc/paper/2021/hash/5857d68cd9280bc98d079fa912fd6740-Abstract.html)], [[Code](https://github.com/jindi-tju/U-GCN)]
- EIGNN: Efficient Infinite-Depth Graph Neural Networks, NeurIPS, [[Paper](https://arxiv.org/abs/2202.10720)], [[Code](https://github.com/liu-jc/EIGNN)]
- BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation, [[Paper](https://arxiv.org/abs/2106.10994)], [[Code](https://github.com/ivam-he/BernNet)]
- Beyond Low-frequency Information in Graph Convolutional Networks, AAAI, [[Paper](https://arxiv.org/abs/2101.00797)], [[Code](https://github.com/bdy9527/FAGCN)]
- Graph Neural Networks with Heterophily, AAAI, [[Paper](https://arxiv.org/abs/2009.13566)], [[Code](https://github.com/GemsLab/CPGNN)]
- Node Similarity Preserving Graph Convolutional Networks, WSDM, [[Paper](https://arxiv.org/abs/2011.09643)], [[Code](https://github.com/ChandlerBang/SimP-GCN)]
- Adaptive Universal Generalized PageRank Graph Neural Network, ICLR, [[Paper](https://arxiv.org/abs/2006.07988)], [[Code](https://github.com/jianhao2016/GPRGNN)]
- How to Find Your Friendly Neighborhood: Graph Attention Design with Self-Supervision, ICLR, [[Paper](https://openreview.net/forum?id=Wi5KUNlqWty)], [[Code](https://github.com/dongkwan-kim/SuperGAT)]
- Energy Levels Based Graph Neural Networks for Heterophily, Journal of Physics: Conference Series, [[Paper](https://iopscience.iop.org/article/10.1088/1742-6596/1948/1/012042/meta)], [Code]
- Geometric Scattering Attention Networks, ICASSP, [[Paper](https://arxiv.org/abs/2010.15010)], [Code]
- Breaking the Limit of Graph Neural Networks by Improving the Assortativity of Graphs with Local Mixing Patterns, KDD, [[Paper](https://arxiv.org/abs/2106.06586)], [[Code](https://github.com/susheels/gnns-and-local-assortativity)]
- Non-Local Graph Neural Networks, TPAMI, [[Paper](https://arxiv.org/abs/2005.14612)], [[Code](https://github.com/divelab/Non-Local-GNN)]
- Is Heterophily A Real Nightmare For Graph Neural Networks To Do Node Classification?, arXiv, [[Paper](https://arxiv.org/abs/2109.05641)], [Code]
- GCN-SL: Graph Convolutional Networks with Structure Learning for Graphs under Heterophily, arXiv, [[Paper](https://arxiv.org/abs/2105.13795)], [Code]
- Unifying Homophily and Heterophily Network Transformation via Motifs, arXiv, [[Paper](https://arxiv.org/abs/2012.11400)], [Code]
- Two Sides of The Same Coin: Heterophily and Oversmoothing in Graph Convolutional Neural Networks, arXiv, [[Paper](https://arxiv.org/abs/2102.06462)], [Code]
- Beyond Low-Pass Filters: Adaptive Feature Propagation on Graphs, arXiv, [[Paper](https://arxiv.org/abs/2103.14187)], [Code]
- SkipNode: On Alleviating Over-smoothing for Deep Graph Convolutional Networks, arXiv, [[Paper](https://arxiv.org/abs/2112.11628)], [Code]
- Node2Seq: Towards Trainable Convolutions in Graph Neural Networks, arXiv, [[Paper](https://arxiv.org/abs/2101.01849)], [Code]
- Simplifying Approach to Node Classification in Graph Neural Networks, arXiv, [[Paper](https://arxiv.org/abs/2111.06748)], [[Code](https://github.com/sunilkmaurya/FSGNN)]


### 2020

- Simple and Deep Graph Convolutional Networks, ICML, [[Paper](https://arxiv.org/abs/2007.02133)], [[Code](https://github.com/chennnM/GCNII)]
- Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs, NeurIPS, [[Paper](https://arxiv.org/abs/2006.11468)], [[Code](https://github.com/GemsLab/H2GCN)]
- Geom-GCN: Geometric Graph Convolutional Networks, ICLR, [[Paper](https://arxiv.org/abs/2002.05287)], [[Code](https://github.com/graphdml-uiuc-jlu/geom-gcn)]

### 2019 and before

- To be added

## Datasets

- To be added

  <!--cora、citeseer、pubmed------------cornell、texas、wisconsin、chameleon、squirrel、actor、FB100、SNAP-->
