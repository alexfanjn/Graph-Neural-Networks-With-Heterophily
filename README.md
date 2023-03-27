# Awesome Resources on Graph Neural Networks With Heterophily

This repository contains the relevant resources on graph neural network (GNN) considering heterophily. 

It's worth noting that the **heterophily** we consider here is not the same as **heterogeneity**. Heterogeneity is more related to the node type difference such as the user and item nodes in recommender systems, but heterophily is more like the feature or label difference between the neighbors under the nodes with the same type. Traditional GNNs usually assume that similar nodes (features/classes) are connected together, but the "opposites attract" phenomenon also widely exists in general graphs.

If you find anything incorrect, please let me know. Thanks!

<!--[[Paper](https://arxiv.org/abs/2101.00797)], [[Code](https://github.com/bdy9527/FAGCN)]-->

## Papers

### 2023

- Beyond Smoothing: Unsupervised Graph Representation Learning with Edge Heterophily Discriminating, AAAI, [[Paper](https://arxiv.org/abs/2211.14065)], [[Code](https://github.com/yixinliu233/GREET)]
- 2-hop Neighbor Class Similarity (2NCS): A graph structural metric indicative of graph neural network performance, AAAI-W, [[Paper](https://arxiv.org/abs/2212.13202)], [Code]
- Ordered GNN: Ordering Message Passing to Deal with Heterophily and Over-smoothing, ICLR, [[Paper](https://openreview.net/forum?id=wKPmPBHSnT6)], [[Code](https://github.com/LUMIA-Group/OrderedGNN)]
- Gradient Gating for Deep Multi-Rate Learning on Graphs, ICLR, [[Paper](https://openreview.net/forum?id=JpRExTbl1-)], [Code]
- ACMP: Allen-Cahn Message Passing with Attractive and Repulsive Forces for Graph Neural Networks, ICLR, [[Paper](https://openreview.net/forum?id=4fZc_79Lrqs)], [[Code](https://github.com/ykiiiiii/ACMP)]
- A Critical Look at Evaluation of GNNs Under Heterophily: Are We Really Making Progress?, ICLR, [[Paper](https://openreview.net/forum?id=tJbbQfw-5wv)], [[Code](https://github.com/yandex-research/heterophilous-graphs)]
- GReTo: Remedying dynamic graph topology-task discordance via target homophily, ICLR, [[Paper](https://openreview.net/forum?id=8duT3mi_5n)], [[Code](https://github.com/zzyy0929/ICLR23-GReTo)]
- SlenderGNN: Accurate, Robust, and Interpretable GNN, and the Reasons for its Success, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=lMgFRIILVB)], [Code]
- Simple Spectral Graph Convolution from an Optimization Perspective, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=cZM4iZmxzR7)], [Code]
- The Impact of Neighborhood Distribution in Graph Convolutional Networks, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=XUqTyU9VlWp)], [Code]
- Wide Graph Neural Network, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=Ih0fKoIUyEh)], [Code]
- Are Graph Attention Networks Attentive Enough? Rethinking Graph Attention by Capturing Homophily and Heterophily, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=Xk10fyKR8G)], [Code]
- Node Classification Beyond Homophily: Towards a General Solution, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=kh3JurmKlux)], [Code]
- From ChebNet to ChebGibbsNet, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=2a5Ru3JtNe0)], [Code]
- ProtoGNN: Prototype-Assisted Message Passing Framework for Non-Homophilous Graphs, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=LeZ39Gkwbi0)], [Code]
- Low-Rank Graph Neural Networks Inspired by the Weak-balance Theory in Social Networks, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=ufCQZeAMZzf)], [Code]
- Can Single-Pass Contrastive Learning Work for Both Homophilic and Heterophilic Graph?, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=XE0cIoi-sZ1)], [Code]
- Graph Contrastive Learning Under Heterophily: Utilizing Graph Filters to Generate Graph Views, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=NzcUQuhEGef)], [Code]
- GOAT: A Global Transformer on Large-scale Graphs, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=z29R0uMiF3v)], [Code]
- ReD-GCN: Revisit the Depth of Graph Convolutional Network, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=tMg5hKRiW-2)], [Code]
- Graph Neural Networks as Gradient Flows: Understanding Graph Convolutions via Energy, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=M3GzgrA7U4)], [Code]
- Causally-guided Regularization of Graph Attention improves Generalizability, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=U086TJFWy4p)], [Code]
- GReTo: Remedying dynamic graph topology-task discordance via target homophily, ICLR OpenReview, [[Paper](https://openreview.net/forum?id=8duT3mi_5n)], [Code]
- Leave Graphs Alone: Addressing Over-Squashing without Rewiring, arXiv, [[Paper](https://arxiv.org/abs/2212.06538)], [Code]
- Statistical Mechanics of Generalization In Graph Convolution Networks, arXiv, [[Paper](https://arxiv.org/abs/2212.13069)], [Code]
- Semi-Supervised Classification with Graph Convolutional Kernel Machines, arXiv, [[Paper](https://arxiv.org/abs/2301.13764)], [Code]
- A Graph Neural Network with Negative Message Passing for Graph Coloring, arXiv, [[Paper](https://arxiv.org/abs/2301.11164)], [Code]
- Neighborhood Homophily-Guided Graph Convolutional Network, arXiv, [[Paper](https://arxiv.org/abs/2301.09851)], [Code]
- Is Signed Message Essential for Graph Neural Networks?, arXiv, [[Paper](https://arxiv.org/abs/2301.08918)], [Code]
- Attending to Graph Transformers, arXiv, [[Paper](https://arxiv.org/abs/2302.04181)], [[Code](https://github.com/luis-mueller/probing-graph-transformers)]
- Heterophily-Aware Graph Attention Network, arXiv, [[Paper](https://arxiv.org/abs/2302.03228)], [Code]

### 2022

- Block Modeling-Guided Graph Convolutional Neural Networks, AAAI, [[Paper](https://arxiv.org/abs/2112.13507)], [[Code](https://github.com/hedongxiao-tju/BM-GCN)]
- Powerful Graph Convolutioal Networks with Adaptive Propagation Mechanism for Homophily and Heterophily, AAAI, [[Paper](https://arxiv.org/abs/2112.13562)], [[Code](https://github.com/hedongxiao-tju/HOG-GCN)]
- Deformable Graph Convolutional Networks, AAAI, [[Paper](https://arxiv.org/abs/2112.14438)], [Code]
- Graph Pointer Neural Networks, AAAI, [[Paper](https://arxiv.org/abs/2110.00973)], [Code]
- Is Homophily A Necessity for Graph Neural Networks?, ICLR, [[Paper](https://arxiv.org/abs/2106.06134)], [Code]
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
- Unsupervised Network Embedding Beyond Homophily, TMLR, [[Paper](https://arxiv.org/abs/2203.10866v3)], [[Code](https://github.com/zhiqiangzhongddu/Selene)]
- GSN: A Universal Graph Neural Network Inspired by Spring Network, arXiv, [[Paper](https://arxiv.org/abs/2201.12994)], [Code]
- Graph Decoupling Attention Markov Networks for Semi-supervised Graph Node Classification, arXiv, [[Paper](https://arxiv.org/abs/2104.13718)], [Code]
- Relational Graph Neural Network Design via Progressive Neural Architecture Search, arXiv, [[Paper](https://arxiv.org/abs/2105.14490)], [Code]
- Graph Neural Network with Curriculum Learning for Imbalanced Node Classification, arXiv, [[Paper](https://arxiv.org/abs/2202.02529)], [Code]
- ***[Survey Paper]*** Graph Neural Networks for Graphs with Heterophily: A Survey, arXIv, [[Paper](https://arxiv.org/abs/2202.07082)], [Code]
- When Does A Spectral Graph Neural Network Fail in Node Classification?, arXIv, [[Paper](https://arxiv.org/abs/2202.07902)], [Code]
- Graph Representation Learning Beyond Node and Homophily, arXiv, [[Paper](https://arxiv.org/abs/2203.01564)], [[Code](https://github.com/syvail/PairE-Graph-Representation-Learning-Beyond-Node-and-Homophily)]
- Incorporating Heterophily into Graph Neural Networks for Graph Classification, arXiv, [[Paper](https://arxiv.org/abs/2203.07678)], [[Code](https://github.com/yeweiysh/IHGNN)]
- Exploiting Neighbor Effect: Conv-Agnostic GNNs Framework for Graphs with Heterophily, arXiv, [[Paper](https://arxiv.org/abs/2203.11200)], [[Code]](https://github.com/JC-202/CAGNN)
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
- Characterizing Graph Datasets for Node Classification: Beyond Homophily-Heterophily Dichotomy, arXiv, [[Paper](https://arxiv.org/abs/2209.06177)], [Code]
- Make Heterophily Graphs Better Fit GNN: A Graph Rewiring Approach, arXiv, [[Paper](https://arxiv.org/abs/2209.08264)], [Code]
- Break the Wall Between Homophily and Heterophily for Graph Representation Learning, arXiv, [[Paper](https://arxiv.org/abs/2210.05382)], [Code]
- GPNet: Simplifying Graph Neural Networks via Multi-channel Geometric Polynomials, arXiv, [[Paper](https://arxiv.org/abs/2209.15454)], [Code]
- HP-GMN: Graph Memory Networks for Heterophilous Graphs, arXiv, [[Paper](https://arxiv.org/abs/2210.08195)], [[Code]](https://github.com/junjie-xu/hp-gmn)
- Improving Your Graph Neural Networks: A High-Frequency Booster, arXiv, [[Paper](https://arxiv.org/abs/2210.08251)], [Code]
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
