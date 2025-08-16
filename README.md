# GPSN
Deploy GPSN - DFN: "Detecting Fake News Based on Graph Propagation Structure and Sequential Network"

# Overview
Combine user profiles/preferences, news propagation context, and sequential chain of time-based interactions to detect fake news. User features are encoded (BERT, Spacy) as nodes in a news propagation graph and sequential chain. Then, use GNN layers (GCN, GAT, GraphSage, GTN) to process the news propagation graph and Transformer/LSTM to process the sequential information layer . These vectors are aggregated and trained to create a fake news detector.

# Datasets
Mô hình của tôi huấn luyện và đánh giá trên tập dữ liệu 'UPFD_Politifact' và 'UPFD_Gossipcop'. Thông tin chi tiết về 2 tập dữ liệu xin mời tham khảo 2 bài báo:

[Fakenewsnet: A data repository with news content, social context, and spatiotemporal information for studying fake news on social media](https://arxiv.org/pdf/1809.01286)

[User Preference-aware Fake News Detection](https://dl.acm.org/doi/pdf/10.1145/3404835.3462990)

Tổng quan bộ dữ liệu:

| Data  | #Graphs  | #Fake News| #Total Nodes  | #Total Edges  | #Avg. Nodes per Graph  |
|-------|--------|--------|--------|--------|--------|
| Politifact | 314   |   157    |  41,054  | 40,740 |  131 |
| Gossipcop |  5464  |   2732   |  314,262  | 308,798  |  58  |

Bạn có thể thu thập dữ liệu thô và tham khảo quá trình xử lý dữ liệu tại: [GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews/tree/main)

<p align="center">
    <br>
    <a href="[https://github.com/safe](https://github.com/ntn2110q1/GPSN)">
        <img src="https://github.com/ntn2110q1/GPSN/blob/main/picture/DataProcessing.png" width="1000"/>
    </a>
    <br>
<p>

