import asyncio
import httpx
import sys
from httpx_sse import aconnect_sse

async def test_connection(label):
    print(f"\n--- Testing connection via: {label} ---")
    try:
        async with httpx.AsyncClient() as client:
            # Using your confirmed working URL
            async with aconnect_sse(client, "GET", "http://0.0.0.0:8001/sse") as event_source:
                print(f"[{label}] Successfully connected to FastMCP server!")
                async for event in event_source.aiter_sse():
                    print(f"[{label}] Event received: {event.event}")
                    break
    except Exception as e:
        print(f"[{label}] FAILURE: {type(e).__name__}: {e}")

# 1. Test using default asyncio
print("Starting standard asyncio test...")
asyncio.run(test_connection("asyncio"))

# 2. Test using uvloop (the vLLM way)
try:
    import uvloop
    print("\nStarting uvloop test (vLLM style)...")
    # Setting the policy to uvloop is how vLLM forces the event loop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(test_connection("uvloop"))
except ImportError:
    print("\nuvloop not installed. Cannot test.")
except Exception as e:
    print(f"\nCritical error during uvloop execution: {e}")