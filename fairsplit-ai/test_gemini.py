from langchain_core.messages import HumanMessage
from agents.llm_client import get_llm

prompt = """You are a neutral mediator splitting a shared expense fairly among a group.

Total amount: 30000.0
Itemized costs (if any): {"stay": 15000.0, "food": 8000.0, "travel": 7000.0}
People and their stated situations:
- Aisha: Did not eat dinners (skipped food costs entirely)
- Raj: Used the private pool room (should pay more for stay)
- Meera: Budget-constrained this month, joined only for 2 of 4 days (hard max: 6000.0)

Propose a split (amount per person) that sums exactly to the total amount.
Briefly justify your reasoning in 2-3 sentences.
Respond ONLY in JSON: {"proposal": {"name": amount, ...}, "reasoning": "..."}"""

llm = get_llm()
print(repr(llm.invoke([HumanMessage(content=prompt)]).content))
