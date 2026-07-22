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
| BM21 | 2026-07-21 | nlp diverse-beam step (deferred materialization), cheap forward; allocs/op 320776->1528 -99.5%, B/op 82.6->41.5MiB -49.8% | darwin/arm64 | self | ms | 140.1 | 126.3 | 1.11× | T942,V22 |
| BM22 | 2026-07-21 | nlp BeamSearch step (deferred materialization, cheap-fwd) | darwin/arm64 | self | ms | 124.1 | 105.6 | 1.18× | T943,V22 |
| BM23 | 2026-07-21 | nlp nucleusTopP (top-p, vocab 32000, full-sort path) | darwin/arm64 | self | us | 703 | 674.6 | 1.04× | T944,V22 |
| BM24 | 2026-07-21 | nlp nucleusTopP idx pooled (top-p, vocab 32000) | darwin/arm64 | self | us | 676 | 655.5 | 1.03× | T945,V22 |
| BM25 | 2026-07-21 | nlp typicalTruncate radix pooled (vocab 32000, full-sort) | darwin/arm64 | self | us | 1179 | 1144 | 1.03× | T946,V22 |
| BM26 | 2026-07-21 | nlp typicalTruncate score/idx/keep pooled (vocab 32000) | darwin/arm64 | self | us | 1148 | 1102 | 1.04× | T947,V22 |
| BM27 | 2026-07-21 | nlp Sampler.Dist z+topk-scratch pooled (temp+topk+topp, vocab 32000) | darwin/arm64 | self | us | 347.8 | 316.5 | 1.1× | T948,V22 |
| BM28 | 2026-07-21 | nlp Sampler.penalize logits-copy pooled (SampleWithHistory, penalties+top-p, vocab 50257) | darwin/arm64 | self | us | 521.4 | 484.2 | 1.08× | T949,V22 |
| BM29 | 2026-07-21 | nlp Sample per-token probs pooled via distInto (SampleWithHistory, penalties+top-p, vocab 50257) | darwin/arm64 | self | us | 486.8 | 453.9 | 1.07× | T950,V22 |
| BM30 | 2026-07-21 | nlp Unigram.Encode Viterbi DP scratch pooled (~2KB text) | darwin/arm64 | self | us | 123.2 | 116.2 | 1.06× | T951,V22 |
| BM31 | 2026-07-21 | nlp SPM.Encode spmBounds merge scratch pooled (~1KB text) | darwin/arm64 | self | us | 70.28 | 63.25 | 1.11× | T952,V22 |
| BM32 | 2026-07-21 | nlp WordPiece.Encode into out + FieldsSeq (~2.6KB text) | darwin/arm64 | self | us | 109.4 | 102.1 | 1.07× | T953,V22 |
| BM33 | 2026-07-21 | nlp BPETokenizer(GGUF).Encode reuse mapped buffer via unsafe.String (110KB text) | darwin/arm64 | self | ms | 5.061 | 4.546 | 1.11× | T954,V22 |
| BM34 | 2026-07-22 | nlp Llama.DecodeStep hoist per-layer Attrs boxing (Generate500, allocs) | darwin/arm64 | self | allocs/op | 350988 | 346483 | 1.01× | T955,V22 |
| BM35 | 2026-07-22 | nlp Gemma.DecodeStep hoist per-layer Attrs boxing (28-step decode, allocs) | darwin/arm64 | self | allocs/op | 9757 | 9673 | 1.01× | T957,V22 |
| BM36 | 2026-07-22 | nlp Cohere.DecodeStep hoist per-layer Attrs boxing (28-step decode, allocs) | darwin/arm64 | self | allocs/op | 8917 | 8833 | 1.01× | T958,V22 |
| BM37 | 2026-07-22 | nlp Falcon/OLMoE/GraniteMoE.DecodeStep hoist per-layer Attrs (Falcon 28-step, allocs) | darwin/arm64 | self | allocs/op | 7685 | 7601 | 1.01× | T959,V22 |
| BM38 | 2026-07-22 | nlp project pooled 2-input slice recorder-guarded (Falcon 28-step decode, allocs) | darwin/arm64 | self | allocs/op | 7601 | 7265 | 1.05× | T960,V22 |
| BM39 | 2026-07-22 | nlp residual OpAdd pooled via exec2 across 23 decode models (Falcon 28-step, allocs) | darwin/arm64 | self | allocs/op | 7265 | 7153 | 1.02× | T961,V22 |
| BM40 | 2026-07-22 | nlp RoPE+MHA pooled via exec1a/exec3 recorder-guarded (Falcon 28-step decode, allocs) | darwin/arm64 | self | allocs/op | 7153 | 6986 | 1.02× | T962,V22 |
| BM41 | 2026-07-22 | nn RMSNorm/LayerNorm input slice pooled recorder-guarded (Falcon 28-step decode, allocs) | darwin/arm64 | self | allocs/op | 6986 | 6903 | 1.01× | T963,V22 |
| BM42 | 2026-07-22 | nn SwiGLU FFN input slices pooled recorder-guarded (Forward dim256/hidden1024, allocs) | darwin/arm64 | self | allocs/op | 46 | 41 | 1.12× | T964,V22 |
| BM43 | 2026-07-22 | nn SparseMoE.ForwardDecode per-expert slices pooled recorder-guarded (Mixtral decode, allocs) | darwin/arm64 | self | allocs/op | 155 | 152 | 1.02× | T965,V22 |
| BM44 | 2026-07-22 | rl softUpdate typed fast path (Polyak target update, ~130k-param MLP) | darwin/arm64 | self | us | 1279.8 | 70.35 | 18.19× | T966,V22 |
| BM45 | 2026-07-22 | rl forward() typed contiguous fill (256×64 batch) | darwin/arm64 | self | us | 214.6 | 155.9 | 1.38× | T967,V22 |
| BM46 | 2026-07-22 | nn EWCFisher typed contiguous fast path (Fisher-info estimate, ~131k-param MLP × 8 samples) | darwin/arm64 | self | us | 11179.5 | 548.3 | 20.39× | T968,V22 |
| BM47 | 2026-07-22 | nn MASImportance typed contiguous fast path (MAS importance Ω=mean\|g\|, ~131k-param MLP × 8 samples) | darwin/arm64 | self | us | 11176.8 | 579.4 | 19.29× | T969,V22 |
| BM48 | 2026-07-22 | nn SI Consolidate typed contiguous fast path (Synaptic-Intelligence task-boundary consolidation, ~131k-param MLP) | darwin/arm64 | self | us | 1094.6 | 255.5 | 4.28× | T970,V22 |
| BM49 | 2026-07-22 | nn SLERP typed contiguous fast path (spherical weight interpolation, model-merge, ~131k params) | darwin/arm64 | self | us | 1872.7 | 325.5 | 5.75× | T971,V22 |
| BM50 | 2026-07-22 | nn DARE typed contiguous fast path (drop-and-rescale model merge, ~131k params, rng-dominated) | darwin/arm64 | self | us | 2485.3 | 1881.3 | 1.32× | T972,V22 |
| BM51 | 2026-07-22 | nn TIESMerge slices.SortStableFunc + typed passes (TIES model merge, ~131k params × 3 models, sort-dominated) | darwin/arm64 | self | us | 54415.1 | 32598.6 | 1.67× | T973,V22 |
| BM52 | 2026-07-22 | nn MemorizingAttention AddSegment typed row-copy (memory-bank fold, 128×512) | darwin/arm64 | self | us | 433.5 | 129.7 | 3.34× | T974,V22 |
| BM53 | 2026-07-22 | nn UniformSoup/averageModels typed fast path (model-soup averaging, 5 models × ~131k params) | darwin/arm64 | self | us | 3904.4 | 410.9 | 9.5× | T975,V22 |
| BM54 | 2026-07-22 | nlp chat-template Render Builder pre-size (llama3, 11-msg conversation) | darwin/arm64 | self | ns | 2797.5 | 1026 | 2.73× | T976,V22 |
| BM55 | 2026-07-22 | nlp JSONSchemaToGrammar grammar-Builder pre-size (nested schema, byte-reduction) | darwin/arm64 | self | B/op | 408591 | 350121 | 1.17× | T977,V22 |
