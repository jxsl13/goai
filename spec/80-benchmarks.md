## §Bench — benchmark records

| id | date | benchmark | machine | incumbent | metric | before | after | impact | cites |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BM1 | 2026-07-20 | nn NEFTune.Forward embedding-noise build [512,768]=393K | darwin/arm64 | self | ms | 2.76 | 2.00 | 1.38× | T922,V22 |
| BM2 | 2026-07-20 | nn Dropout.Forward Bernoulli-mask build [16,128,768]=1.57M | darwin/arm64 | self | ms | 16.90 | 10.69 | 1.58× | T919,V22 |
| BM3 | 2026-07-20 | nn DropPath.Forward per-sample mask build [16,128,768]=1.57M | darwin/arm64 | self | ms | 11.97 | 0.91 | 13.15× | T919,V22 |
| BM4 | 2026-07-21 | nn FocalLoss detection one-hot build [8192,4] many-rows | darwin/arm64 | self | ms | 1.079 | 1.045 | 1.03× | T927,V22 |
| BM5 | 2026-07-21 | nlp BPE GPT2Encode (per-prompt tokenization, 1MB corpus) | darwin/arm64 | self | ms | 23.37 | 22.13 | 1.06× | T928,V22 |
| BM6 | 2026-07-21 | nlp BPE GPT2Decode (ids->text, 1MB output) | darwin/arm64 | self | ms | 2.299 | 2.086 | 1.1× | T929,V22 |
| BM7 | 2026-07-21 | nlp WordPiece Decode (ids->text, 300k ids) | darwin/arm64 | self | ms | 1.947 | 1.767 | 1.1× | T930,V22 |
| BM8 | 2026-07-21 | nlp Unigram Decode (ids->text, 300k ids) | darwin/arm64 | self | ms | 4.485 | 4.371 | 1.03× | T930,V22 |
| BM9 | 2026-07-21 | nlp SPM Decode (ids->text, 300k ids) | darwin/arm64 | self | ms | 94.93 | 93.20 | 1.02× | T930,V22 |
| BM10 | 2026-07-21 | nlp SPM Decode (Sscanf-gated; ids->text, 300k) | darwin/arm64 | self | ms | 93.29 | 4.624 | 20.18× | T931,V22 |
| BM11 | 2026-07-21 | nlp NewSPM construction (30k-piece vocab, byteID build) | darwin/arm64 | self | ms | 10.773 | 1.054 | 10.22× | T932,V22 |
| BM12 | 2026-07-21 | nlp SPM Decode (inline meta-unescape; 300k ids) | darwin/arm64 | self | ms | 4.530 | 2.742 | 1.65× | T934,V22 |
| BM13 | 2026-07-21 | nlp Unigram Decode (inline meta-unescape; 300k ids) | darwin/arm64 | self | ms | 4.354 | 2.636 | 1.65× | T934,V22 |
| BM14 | 2026-07-21 | nlp BPETokenizer(GGUF) Encode (~100k-char text) | darwin/arm64 | self | ms | 12.241 | 9.618 | 1.27× | T936,V22 |
| BM15 | 2026-07-21 | nlp BPETokenizer(GGUF) Decode (ids->text) | darwin/arm64 | self | ms | 1.649 | 1.559 | 1.06× | T937,V22 |
| BM16 | 2026-07-21 | nlp BPETokenizer(GGUF) Encode byte-offset | darwin/arm64 | self | ms | 9.609 | 6.779 | 1.42× | T938,V22 |
| BM17 | 2026-07-21 | nlp BPETokenizer(GGUF) Encode scratch-reuse | darwin/arm64 | self | ms | 6.712 | 5.355 | 1.25× | T939,V22 |
| BM18 | 2026-07-21 | nlp Watermark.Detect (vocab 32000, seq 1024) | darwin/arm64 | self | ms | 75.76 | 50.69 | 1.49× | V22,C27 |
| BM19 | 2026-07-21 | nlp Watermark.BiasLogits (vocab 32000) | darwin/arm64 | self | us | 146.4 | 106.9 | 1.37× | V22,C27 |
| BM20 | 2026-07-21 | nlp ApplyPenalties (vocab 32000, history 2048) | darwin/arm64 | self | us | 37.98 | 36.28 | 1.05× | T941,V22 |
