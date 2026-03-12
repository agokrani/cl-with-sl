#!/usr/bin/env python3
"""Quick test: 1 sample from GPT-4.1-nano via OpenAI API."""
import asyncio
import sys
sys.path.insert(0, "subliminal-learning")

from sl.llm import services as llm_services
from sl.llm.data_models import Model, SampleCfg

async def main():
    model = Model(id="gpt-4.1-nano-2025-04-14", type="openai")
    chat = llm_services.build_simple_chat(
        system_content="You love owls.",
        user_content="Give me 10 random 3-digit numbers, comma separated.",
    )
    resp = await llm_services.sample(model, chat, SampleCfg(temperature=1.0))
    print(f"Model: {resp.model_id}")
    print(f"Response: {resp.completion}")
    print(f"Stop reason: {resp.stop_reason}")

asyncio.run(main())
