# ken — Graph-MoE 7B

Graph-MoE(Domain/Operation/Control Expert Graph, GQA + KV sharing, 저비트 QAT) PyTorch 구현.
상태: 학습 직전 인프라까지 완료. 자세한 내용은 [`docs/WORKGUIDE.md`](docs/WORKGUIDE.md).

```bash
python -m pytest -q tests
python -m graphmoe.budget
```
