# Training

Use the runtime without LoRA first.

Recommended sequence:
1. baseline
2. Agent LoRA
3. Code LoRA
4. Reasoning LoRA
5. only then Long/Vision experiments

Use observable behavior: answers, concise plans, tool calls, tool results and verification outcomes. Do not rely on private chain-of-thought from closed models.

Keep a held-out evaluation set and only deploy an adapter if task success improves without unacceptable latency/tool regressions.
