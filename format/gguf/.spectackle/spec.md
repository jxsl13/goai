---
schema: v1
prefix: SIZE
---

## SIZE-THE-FANOUT-TO-THE-WORK-001
WHEN a fan-out helper splits into GOMAXPROCS workers with only an on/off work gate, the that helper SHALL scale the WORKER COUNT to the work as well, one worker per fixed grain of element-work capped by GOMAXPROCS, and pick the grain against both the clock and the CPU.

Rationale: Measured on the quantized decode matmul, which is called thousands of times per generation with one activation row. A profile of a 500-token quantized Llama generation at twelve cores spent 88 percent of its samples in pthread_cond_signal and pthread_cond_wait and 1.5 percent in the kernel itself, yet the fan-out was still worth having: forcing the same call serial cost 8 percent of the clock, and leaving it at twelve workers burned 42 percent more user CPU to buy that 8 percent back. THE ON/OFF GATE CANNOT EXPRESS THE ANSWER because the answer is a worker count, not a yes or no. One worker per 1<<15 of element-work, capped by GOMAXPROCS, beat both: BenchmarkQuantLlamaGenerate500 549.7 to 527.4 ms, minus 4.1 percent, with system CPU 2.03 to 1.44 s, minus 27 percent, and the prefill cell flat as a control since it takes the m greater than 1 path. Bit-identical at any chunk count - each output row is an independent dot with its own accumulator over ascending k. THE GRAIN IS A TRADE AND THE TWO AXES DISAGREE: a coarser 1<<16 gave 536.1 ms and 1.19 s of system time, so it costs 4 percent of the clock to save another 44 percent of the system time, and which side is right depends on whether the machine serves one request or many. Encoded as perfscan PS3061, 10 candidates tree-wide - every other fan-out helper in the repo.
