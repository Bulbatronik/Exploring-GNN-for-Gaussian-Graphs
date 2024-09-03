# Graph Neural Networks for Graph Classification

## :dart: Objective ##

The project's goal is to implement Graph Neural Networks (GNN) for graph classification in unbalanced settings that consider edge weights, utilizing attention mechanisms and other techniques to improve the interpretability of the predictions.

## :checkered_flag: Usage ##

Install the necessary dependencies
```bash
pip install -r requirements.txt
```

## :checkered_flag: Dataset
The data used for simulation are two synthetic datasets simulated using [`Gaussian graph model`](https://arxiv.org/pdf/1707.04345) for *balanced* and *unbalanced* cases (credits to [Alessia Mapelli](https://humantechnopole.it/it/people/alessia-mapelli/)). Each case consists of *500* undirected graphs with *20* nodes. Each of the `.RData` files is structured the following way:
- **Adj_matrices**: Contains the weighted adjacency matrix for each sample.
- **Sim_data1**: Contains the node observations and graph class for each sample.
 

## :rocket: Pipeline

- **Exploratory Data Analysis (EDA)**: the step involves investigation of the dataset focusing on the node statistics and features, edge statistics and weights distribution and general topology analysis. 
- **Data Preprocessing**: this step we implement scaling of the node features and weights of the adjacency matrices. In addition, we implement additional features extracted from the encoding of the node identifier and embeddings provided by `Node2Vec`.
- **Model Training**: we implement training of various GNN models, including `GCN`, `GAT`, `GraphConv` and `GIN`, including the implementation of some of them from scratch. We also tran `MLP` to compare the graph models with a topology-agnostic model.
- **Explainability (XAI)**: using `Graph Neural Network Explainer (GNNE)` and `Integrated Gradients (IG)`, we visualize the most influential nodes and edges to the final performance of the model.

## :memo: Results
The main reults are provided in the [Presentation.pdf](Presentation.pdf) file. 


## Extra References

- Morris, C. (2022). ***Graph Neural Networks: Graph classification.*** In L. Wu, P. Cui, J. Pei, & L. Zhao (Eds.), Graph Neural Networks: Foundations, Frontiers, and Applications (pp. 179-193). Singapore: Springer Nature. https://doi.org/10.1007/978-981-16-6054-2_9
- Sui, Y., Wang, X., Wu, J., He, X., & Chua, T.-S. (2021). ***Deconfounded training for graph neural networks.*** CoRR, abs/2112.15089. Retrieved from https://arxiv.org/abs/2112.15089
- Gong, L., & Cheng, Q. (2018). ***Adaptive edge features guided graph attention networks.*** CoRR, abs/1809.02709. Retrieved from http://arxiv.org/abs/1809.02709
- Wang, Y., Zhao, Y., Shah, N., & Derr, T. (2022). ***Imbalanced graph classification via Graph-of-Graph Neural Networks.*** In Proceedings of the 31st ACM International Conference on Information & Knowledge Management (CIKM '22) (pp. 2067-2076). New York, NY, USA: Association for Computing Machinery. https://doi.org/10.1145/3511808.3557356
