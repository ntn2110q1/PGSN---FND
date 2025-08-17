# PGSN
Deploy PGSN - DFN: "Detecting Fake News Based on Propagation Graph Structure and Sequential Network"

# Overview
Combine user profiles/preferences, news propagation context, and sequential chain of time-based interactions to detect fake news. User features are encoded (BERT, Spacy) as nodes in a news propagation graph and sequential chain. Then, use GNN layers (GCN, GAT, GraphSage, GTN) to process the news propagation graph and Transformer/LSTM to process the sequential information layer . These vectors are aggregated and trained to create a fake news detector.

## Datasets
My model was trained and evaluated on the 'UPFD_Politifact' and 'UPFD_Gossipcop' datasets. For detailed information about these two datasets, please refer to the following two papers:

[Fakenewsnet: A data repository with news content, social context, and spatiotemporal information for studying fake news on social media](https://arxiv.org/pdf/1809.01286)

[User Preference-aware Fake News Detection](https://dl.acm.org/doi/pdf/10.1145/3404835.3462990)

Overview:

| Data  | #Graphs  | #Fake News| #Total Nodes  | #Total Edges  | #Avg. Nodes per Graph  |
|-------|--------|--------|--------|--------|--------|
| Politifact | 314   |   157    |  41,054  | 40,740 |  131 |
| Gossipcop |  5464  |   2732   |  314,262  | 308,798  |  58  |

You can collect the raw data and refer to the data processing procedure at: [GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews/tree/main)

<p align="center">
    <br>
    <a href="[https://github.com/safe](https://github.com/ntn2110q1/PGSN---FND)">
        <img src="https://github.com/ntn2110q1/PGSN---FND/blob/main/picture/DataProcessing.png" width="1000"/>
    </a>
    <br>
<p>
To run the model, you should get data and construct the following formation.
<pre> <code> 
GPSN-DFN/
├── mappingdata/
│   ├── gos_id_time_mapping.pkl
│   ├── gos_id_twitter_mapping.pkl
│   ├── gos_news_list.txt
│   ├── pol_id_time_mapping.pkl
│   ├── pol_id_twitter_mapping.pkl
│   └── pol_news_list.txt
├── data/   
│   └── gossipcop/raw/
│   │   ├── A.txt
│   │   ├── graph_labels.npy
│   │   ├── new_bert_feature.npz
│   │   ├── new_content_feature.npz
│   │   ├── new_profile_feature.npz
│   │   ├── new_spacy_feature.npz
│   │   └── node_graph_id.npy
│   └── politifact/raw/
│   │   ├── A.txt
│   │   ├── graph_labels.npy
│   │   ├── new_bert_feature.npz
│   │   ├── new_content_feature.npz
│   │   ├── new_profile_feature.npz
│   │   ├── new_spacy_feature.npz
└   └   └── node_graph_id.npy 
...
</code> </pre>

## Models
To run the code in this repo, you need to have `Python>=3.6`, `PyTorch>=1.6`, and `PyTorch-Geometric>=1.6.1`.

Here is the operational framework of my model:
<p align="center">
    <br>
    <a href="[https://github.com/safe](https://github.com/ntn2110q1/PGSN---FND)">
        <img src="https://github.com/ntn2110q1/PGSN---FND/blob/main/picture/FrameWork.png" width="1000"/>
    </a>
    <br>
<p>

All GNN-based fake news detection models are under the `\gnn_model` directory, sourced from [GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews/tree/main/gnn_model) :

* **[GNN-CL](https://arxiv.org/pdf/2007.03316.pdf)**: Han, Yi, Shanika Karunasekera, and Christopher Leckie. "Graph neural networks with continual learning for fake news detection from social media." arXiv preprint arXiv:2007.03316 (2020).
* **[GCNFN](https://arxiv.org/pdf/1902.06673.pdf)**: Monti, Federico, Fabrizio Frasca, Davide Eynard, Damon Mannion, and Michael M. Bronstein. "Fake news detection on social media using geometric deep learning." arXiv preprint arXiv:1902.06673 (2019).
* **[BiGCN](https://arxiv.org/pdf/2001.06362.pdf)**: Bian, Tian, Xi Xiao, Tingyang Xu, Peilin Zhao, Wenbing Huang, Yu Rong, and Junzhou Huang. "Rumor detection on social media with bi-directional graph convolutional networks." In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 01, pp. 549-556. 2020.
* **[UPFD](https://dl.acm.org/doi/pdf/10.1145/3404835.3462990)**: Dou, Yingtong and Shu, Kai and Xia, Congying and Yu, Philip S. and Sun, Lichao. "User Preference-aware Fake News Detection." Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval.

### Train and Evaluation
To train and evaluate PGSN or UPFD, run the 'pgsn.py' or 'upfd.py' scripts; with lr = 0.01 corresponding to the GraphSage model on the Politifact dataset, and lr = 0.001 for all other cases.
```python
python [pgsn; upfd].py --model[gcn; gat; sage; gtn] --dataset [politifact; gossipcop] --feature [bert; spacy; profile; content] --lr [0.01; 0.001] --epochs
```
To train and evaluate BiGCN; GCNFN or GNN-CL; run the 'bigcn.py'; 'gcnfn.py' or 'gnncl.py'
```python
python [bigcn; gcnfn; gnncl].py --dataset [politifact; gossipcop] --feature [bert; spacy; profile; content] --lr 0.001 --epochs 
```
### Baselines Evaluation
Performance Table of My Model and Baseline Models. This shows that my model has a slight improvement in Accuracy (Acc) and F1-Score (F1):
| Model         | POL (ACC) | POL (F1) | GOS (ACC) | GOS (F1) |
|---------------|-----------|----------|-----------|----------|
| GNN-CL    | 62.90     | 62.25    | 95.11     | 95.09    |
| GCNFN    | 83.16     | 83.56    | 96.38     | 96.36    |
| UPFD   | 84.62   | 84.65   | 97.23   | 97.22 |
| PGSN   | **85.52**    | **85.45**   | **97.62**   | **97.63** |
