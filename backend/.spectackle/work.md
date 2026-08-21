---
schema: v1
---

## ADR-01M0FQ1AP8FBCSCA67X0RCPS8A Where should Vulkan OpEmbedBackward execute while the backend contract returns host-resident tensors synchronously?
kind: adr
state: done
created: 2026-08-20
context: The sibling Metal route has already passed exactness and 3.93x to 30.76x M2 gates; Vulkan must still pass its own MoltenVK measurements before sharing the decision.
decision: Typed deterministic host scatter at the current boundary
consequences: Vulkan OpEmbedBackward will use zero Vulkan submissions only if its independent MoltenVK campaigns clear every frozen speed and spread gate. Reference-order F32 accumulation becomes deterministic. Persistent device residency remains a separate graph-level redesign and invalidates this route decision when introduced.
status: accepted

kind: radio
option: Typed deterministic host scatter at the current boundary
option: Vulkan atomic scatter with upload, wait, and full-table download
option: Introduce persistent device-resident embedding state in this slice
blocks: P-01M0FQ0RWQE8PT2GT7FFP4WKBE
choice: Typed deterministic host scatter at the current boundary

## ADR-01M0FS4JSXFD4TBKY3ESMRGSQ8 How should synchronous host-resident Vulkan bias-gradient reduction choose its execution side?
kind: adr
state: done
created: 2026-08-20
context: The incumbent Vulkan route predates the later bit-exact parallel CPU kernel. The current API returns host tensors synchronously, while future recorder/device-buffer graphs have a different residency contract.
decision: Use a measured shape crossover and preserve Vulkan outside proven CPU winner zones
consequences: Production routing changes only where three independent M2 campaigns clear the frozen speed and stability gates. The old Vulkan path remains the control and the fallback outside measured CPU zones. This applies only to synchronous host-resident tensors; recorder/device-buffer execution remains Vulkan-resident.
status: accepted

kind: radio
option: Always use Vulkan to preserve nominal backend affinity
option: Always route F32 reductions to CPU
option: Use a measured shape crossover and preserve Vulkan outside proven CPU winner zones
blocks: T-01M0FS3HRCE44AKTADYZEQVADR
choice: Use a measured shape crossover and preserve Vulkan outside proven CPU winner zones
