## §Bench — benchmark records

| id | date | benchmark | machine | incumbent | metric | before | after | impact | cites |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BM1 | 2026-07-20 | nn NEFTune.Forward embedding-noise build [512,768]=393K | darwin/arm64 | self | ms | 2.76 | 2.00 | 1.38× | T922,V22 |
| BM2 | 2026-07-20 | nn Dropout.Forward Bernoulli-mask build [16,128,768]=1.57M | darwin/arm64 | self | ms | 16.90 | 10.69 | 1.58× | T919,V22 |
| BM3 | 2026-07-20 | nn DropPath.Forward per-sample mask build [16,128,768]=1.57M | darwin/arm64 | self | ms | 11.97 | 0.91 | 13.15× | T919,V22 |
| BM4 | 2026-07-21 | nn FocalLoss detection one-hot build [8192,4] many-rows | darwin/arm64 | self | ms | 1.079 | 1.045 | 1.03× | T927,V22 |
