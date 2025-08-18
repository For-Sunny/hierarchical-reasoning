import json
import os
import torch
import websockets
import asyncio

LAYER_STATE_PATH = "F:\\ai_bridge\\tensors\\layer_states.json"

async def save_layer_state(layer_name, state, sender="Claude"):
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        msg = {"type": "layer_state_exchange", "layer_name": layer_name, "state": state, "sender": sender}
        await websocket.send(json.dumps(msg))
        response = await websocket.recv()
        print(f"Layer state save response: {response}")
    
    # Local backup
    if not os.path.exists(os.path.dirname(LAYER_STATE_PATH)):
        os.makedirs(os.path.dirname(LAYER_STATE_PATH))
    with open(LAYER_STATE_PATH, "a") as f:
        json.dump({"layer_name": layer_name, "state": state, "sender": sender}, f)
        f.write("\n")

async def load_layer_state(layer_name):
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        msg = {"type": "layer_state_exchange", "layer_name": layer_name, "action": "load"}
        await websocket.send(json.dumps(msg))
        response = await websocket.recv()
        data = json.loads(response)
        print(f"Layer state load response: {response}")
        return data.get("state")

if __name__ == "__main__":
    # Test: Save and load a dummy self-learning layer state
    dummy_state = torch.randn(1024 * 1000, 256).tolist()  # Self-learning: 1024 units * 1000 vectors * 256 dims
    asyncio.run(save_layer_state("self_learning", dummy_state))
    loaded_state = asyncio.run(load_layer_state("self_learning"))