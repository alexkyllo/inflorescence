# inflorescence

A Python framework for clustered federated learning and simulation for
performance and fairness analysis, based on
[Flower](https://github.com/adap/flower).

Flower-compatible implementations of clustered FL strategies included:

- [x] Iterative Federated Clustered Algorithm (IFCA) from [Ghosh (2020)](https://arxiv.org/abs/2006.04088)
- [x] Clustered Federated Learning (CFL) from [Sattler (2019)](https://arxiv.org/abs/1910.01991)
- [x] Federated Learning with Hierarchical Clustering (FL+HC) from [Briggs (2020)](https://arxiv.org/abs/2004.11791)
- [x] Weighted Clustered Federated Learning (WeCFL) from [Ma (2022)](https://arxiv.org/abs/2202.06187)

If you use this package in your work, please cite the paper:

```bibtex
@inproceedings{kyllo2023inflorescence,
  title={Inflorescence: A Framework for Evaluating Fairness with Clustered Federated Learning},
  author={Kyllo, Alex and Mashhadi, Afra},
  booktitle={Adjunct Proceedings of the 2023 ACM International Joint Conference on Pervasive and Ubiquitous Computing & the 2023 ACM International Symposium on Wearable Computing},
  pages={374--380},
  year={2023}
}
```
