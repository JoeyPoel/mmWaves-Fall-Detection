# TI mmWave Radar Fall Detection: Representation Comparison

| Representation                               |   Accuracy (%) |   Precision (%) |   Recall (%) |   Macro F1 (%) |   ROC-AUC |   False Alarm Rate / FPR (%) |   Latency (ms/seq) |
|:---------------------------------------------|---------------:|----------------:|-------------:|---------------:|----------:|-----------------------------:|-------------------:|
| Rep 1: Doppler-Time Map (ResNet-18)          |          75.19 |           77.42 |        72.73 |          75.19 |    0.7837 |                        22.22 |              14.22 |
| Rep 2: 2D Orthogonal Projections (ResNet-18) |          93.02 |           93.85 |        92.42 |          93.02 |    0.9861 |                         6.35 |              20.71 |
| Rep 3: Native 3D Point Tensor (PointNet++)   |          65.12 |           60.19 |        93.94 |          61.41 |    0.7007 |                        65.08 |               9.36 |
