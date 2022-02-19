# Awesome Resources on Graph Neural Networks With Heterophily

This repository contains the relevant resources on graph neural network (GNN) considering heterophily. 

It's worth noting that the **heterophily** we consider here is not the same as **heterogeneity**. Heterogeneity is more related to the node type difference such as the user and item nodes in recommender systems, but heterophily is more like the feature or label difference between the neighbors under the nodes with the same type. Traditional GNNs usually assume that similar nodes (features/classes) are connected together, but the "opposites attract" phenomenon also widely exists in general graphs.

If you find anything incorrect, please let me know.

<!--[[Paper](https://arxiv.org/abs/2101.00797)], [[Code](https://github.com/bdy9527/FAGCN)]-->

## Papers

### 2022

- Block Modeling-Guided Graph Convolutional Neural Networks, AAAI, [[Paper](https://arxiv.org/abs/2112.13507)], [[Code](https://github.com/hedongxiao-tju/BM-GCN)]
- Powerful Graph Convolutioal Networks with Adaptive Propagation Mechanism for Homophily and Heterophily, AAAI, [[Paper](https://arxiv.org/abs/2112.13562)], [[Code](https://github.com/hedongxiao-tju/HOG-GCN)]
- Deformable Graph Convolutional Networks, AAAI, [[Paper](https://arxiv.org/abs/2112.14438)], [Code]
- Is Homophily A Necessity for Graph Neural Networks?, ICLR, [[Paper](https://arxiv.org/abs/2106.06134)], [Code]
- Designing the Topology of Graph Neural Networks: A Novel Feature Fusion Perspective, WWW, [[Paper](https://arxiv.org/abs/2112.14531)], [[Code](https://github.com/AutoML-Research/F2GNN)]
- GBK-GNN: Gated Bi-Kernel Graph Neural Networks for Modeling Both Homophily and Heterophily, WWW, [[Paper](https://arxiv.org/abs/2110.15777)], [[Code](https://github.com/xzh0u/gbk-gnn)]
- GSN: A Universal Graph Neural Network Inspired by Spring Network, arXiv, [[Paper](https://arxiv.org/abs/2201.12994)], [Code]
- GARNET: Reduced-Rank Topology Learning for Robust and Scalable Graph Neural Networks, arXiv, [[Paper](https://arxiv.org/abs/2201.12741)], [Code]
- Graph Decoupling Attention Markov Networks for Semi-supervised Graph Node Classification, arXiv, [[Paper](https://arxiv.org/abs/2104.13718)], [Code]
- Relational Graph Neural Network Design via Progressive Neural Architecture Search, arXiv, [[Paper](https://arxiv.org/abs/2105.14490)], [Code]
- Graph Neural Network with Curriculum Learning for Imbalanced Node Classification, arXiv, [[Paper](https://arxiv.org/abs/2202.02529)], [Code]
- Simplified Graph Convolution with Heterophily, arXiv, [[Paper](https://arxiv.org/abs/2202.04139)], [Code]
- Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs, arXiv, [[Paper](https://arxiv.org/abs/2202.04579)], [Code]
- ***[Survey Paper]*** Graph Neural Networks for Graphs with Heterophily: A Survey, arXIv, [[Paper](https://arxiv.org/abs/2202.07082)], [Code]


### 2021

- Graph Neural Networks with Adaptive Frequency Response Filter, CIKM, [[Paper](https://arxiv.org/abs/2104.12840)], [[Code](https://github.com/yushundong/AdaGNN)]
- Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods, NeurIPS, [[Paper](https://arxiv.org/abs/2110.14446)], [[Code](https://github.com/cuai/non-homophily-large-scale)]
- Diverse Message Passing for Attribute with Heterophily, NeurIPS, [[Paper](https://openreview.net/forum?id=4jPVcKEYpSZ)], [Code]
- Universal Graph Convolutional Networks, NeurIPS, [[Paper](https://papers.nips.cc/paper/2021/hash/5857d68cd9280bc98d079fa912fd6740-Abstract.html)], [[Code](https://github.com/jindi-tju/U-GCN)]
- BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation, [[Paper](https://arxiv.org/abs/2106.10994)], [[Code](https://github.com/ivam-he/BernNet)]
- Beyond Low-frequency Information in Graph Convolutional Networks, AAAI, [[Paper](https://arxiv.org/abs/2101.00797)], [[Code](https://github.com/bdy9527/FAGCN)]
- Graph Neural Networks with Heterophily, AAAI, [[Paper](https://arxiv.org/abs/2009.13566)], [[Code](https://github.com/GemsLab/CPGNN)]
- Node Similarity Preserving Graph Convolutional Networks, WSDM, [[Paper](https://arxiv.org/abs/2011.09643)], [[Code](https://github.com/ChandlerBang/SimP-GCN)]
- Adaptive Universal Generalized PageRank Graph Neural Network, ICLR, [[Paper](https://arxiv.org/abs/2006.07988)], [[Code](https://github.com/jianhao2016/GPRGNN)]
- How to Find Your Friendly Neighborhood: Graph Attention Design with Self-Supervision, ICLR, [[Paper](https://openreview.net/forum?id=Wi5KUNlqWty)], [[Code](https://github.com/dongkwan-kim/SuperGAT)]
- Energy Levels Based Graph Neural Networks for Heterophily, Journal of Physics: Conference Series, [[Paper](https://iopscience.iop.org/article/10.1088/1742-6596/1948/1/012042/meta)], [Code]
- Geometric Scattering Attention Networks, ICASSP, [[Paper](https://arxiv.org/abs/2010.15010)], [Code]
- Breaking the Limit of Graph Neural Networks by Improving the Assortativity of Graphs with Local Mixing Patterns, KDD, [[Paper](https://arxiv.org/abs/2106.06586)], [[Code](https://github.com/susheels/gnns-and-local-assortativity)]
- Global Node Attentions via Adaptive Spectral Filters, OpenReview, [[Paper](https://openreview.net/forum?id=w6Vm1Vob0-X)], [Code]
- Is Heterophily A Real Nightmare For Graph Neural Networks To Do Node Classification?, arXiv, [[Paper](https://arxiv.org/abs/2109.05641)], [Code]
- GCN-SL: Graph Convolutional Networks with Structure Learning for Graphs under Heterophily, arXiv, [[Paper](https://arxiv.org/abs/2105.13795)], [Code]
- Unifying Homophily and Heterophily Network Transformation via Motifs, arXiv, [[Paper](https://arxiv.org/abs/2012.11400)], [Code]
- Non-Local Graph Neural Networks, arXiv, [[Paper](https://arxiv.org/abs/2005.14612)], [Code]
- Two Sides of The Same Coin: Heterophily and Oversmoothing in Graph Convolutional Neural Networks, arXiv, [[Paper](https://arxiv.org/abs/2102.06462)], [Code]
- On The Relationship between Heterophily and Robustness of Graph Neural Networks, arXiv, [[Paper](https://arxiv.org/abs/2106.07767)], [Code]
- Beyond Low-Pass Filters: Adaptive Feature Propagation on Graphs, arXiv, [[Paper](https://arxiv.org/abs/2103.14187)], [Code]
- Label-Wise Message Passing Graph Neural Network on Heterophilic Graphs, arXiv, [[Paper](https://arxiv.org/abs/2110.08128)], [Code]
- SkipNode: On Alleviating Over-smoothing for Deep Graph Convolutional Networks, arXiv, [[Paper](https://arxiv.org/abs/2112.11628)], [Code]


### 2020

- Simple and Deep Graph Convolutional Networks, ICML, [[Paper](https://arxiv.org/abs/2007.02133)], [[Code](https://github.com/chennnM/GCNII)]
- Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs, NeurIPS, [[Paper](https://arxiv.org/abs/2006.11468)], [[Code](https://github.com/GemsLab/H2GCN)]
- Geom-GCN: Geometric Graph Convolutional Networks, ICLR, [[Paper](https://arxiv.org/abs/2002.05287)], [[Code](https://github.com/graphdml-uiuc-jlu/geom-gcn)]

### 2019 and before

- To be added

## Datasets

- To be added

  <!--cora、citeseer、pubmed------------cornell、texas、wisconsin、chameleon、squirrel、actor、FB100、SNAP-->
