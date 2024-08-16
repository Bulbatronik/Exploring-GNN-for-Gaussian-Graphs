# Graph Neural Networks for Graph Classification.

## Objective
The goal of the project is to implement a GNN for graph classification in unbalanced settings that consider edge weights, utilizing attention mechanisms to improve the interpretability of the predictions.


## Dataset
The data used for simulation are two synthetic datasets simulated using [`Gaussian graph model`](https://arxiv.org/pdf/1707.04345) for *balanced* and *unbalanced* cases (credits to [Alessia Mapelli](https://humantechnopole.it/it/people/alessia-mapelli/)). Each case consists of *500* undirected graphs with *20* nodes. Each of the `.RData` files is structured the following way:
- **Adj_matrices**: Contains the weighted adjacency matrix for each sample.
- **Sim_data1**: Contains the node observations and graph class for each sample.
 

## Literature 

In parallel to the dataset, papers/articles/books on the following topics were provided:

- A review on the use of Graph Neural Networks for Graph Classification.
- Basic implementations of GNN for Graph Classification in PyTorch: [LINK](https://github.com/qbxlvnf11/graph-neural-networks-for-graph-classification).
- A paper addressing class imbalance.
- A paper discussing the use of attention to enhance interpretability.
 



## References

- Sui, Y., Wang, X., Wu, J., He, X., & Chua, T.-S. (2021). ***Deconfounded training for graph neural networks.*** CoRR, abs/2112.15089. Retrieved from https://arxiv.org/abs/2112.15089
- Gong, L., & Cheng, Q. (2018). ***Adaptive edge features guided graph attention networks.*** CoRR, abs/1809.02709. Retrieved from http://arxiv.org/abs/1809.02709
- Morris, C. (2022). ***Graph Neural Networks: Graph classification.*** In L. Wu, P. Cui, J. Pei, & L. Zhao (Eds.), Graph Neural Networks: Foundations, Frontiers, and Applications (pp. 179-193). Singapore: Springer Nature. https://doi.org/10.1007/978-981-16-6054-2_9
- Wang, Y., Zhao, Y., Shah, N., & Derr, T. (2022). ***Imbalanced graph classification via Graph-of-Graph Neural Networks.*** In Proceedings of the 31st ACM International Conference on Information & Knowledge Management (CIKM '22) (pp. 2067-2076). New York, NY, USA: Association for Computing Machinery. https://doi.org/10.1145/3511808.3557356