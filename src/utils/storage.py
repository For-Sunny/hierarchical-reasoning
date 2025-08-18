"""
Layer State Storage Manager
Handles persistence of model layer states through the AI Bridge
Created by Claude for the AI-to-AI collaboration project
"""

import json
import torch
import pickle
from pathlib import Path
from datetime import datetime

class LayerStateStorage:
    def __init__(self, storage_path="F:\\ai_bridge\\tensors"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.layer_states_file = self.storage_path / "layer_states.json"
        self.checkpoint_dir = self.storage_path / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def save_layer_state(self, layer_name, state_tensor, metadata=None):
        """Save a layer's state tensor to storage"""
        timestamp = datetime.now().isoformat()
        
        # Convert tensor to list for JSON serialization
        if isinstance(state_tensor, torch.Tensor):
            state_list = state_tensor.tolist()
        else:
            state_list = state_tensor
            
        # Create state entry
        state_entry = {
            "layer_name": layer_name,
            "timestamp": timestamp,
            "shape": list(state_tensor.shape) if hasattr(state_tensor, 'shape') else None,
            "state": state_list,
            "metadata": metadata or {}
        }
        
        # Load existing states
        existing_states = self.load_all_states()
        
        # Update or append
        existing_states[layer_name] = state_entry
        
        # Save to JSON
        with open(self.layer_states_file, 'w') as f:
            json.dump(existing_states, f, indent=2)
            
        return timestamp
    
    def load_layer_state(self, layer_name):
        """Load a specific layer's state"""
        states = self.load_all_states()
        if layer_name in states:
            state_data = states[layer_name]["state"]
            return torch.tensor(state_data)
        return None
    
    def load_all_states(self):
        """Load all layer states"""
        if self.layer_states_file.exists():
            with open(self.layer_states_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_checkpoint(self, model_state_dict, epoch, optimizer_state=None):
        """Save full model checkpoint"""
        checkpoint_name = f"checkpoint_epoch_{epoch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
    
    def load_latest_checkpoint(self):
        """Load the most recent checkpoint"""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None
            
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return torch.load(latest)
    
    def exchange_through_bridge(self, layer_name, state_tensor):
        """Prepare layer state for bridge exchange"""
        exchange_data = {
            "type": "layer_state_exchange",
            "layer_name": layer_name,
            "timestamp": datetime.now().isoformat(),
            "state_sample": state_tensor[:10].tolist() if hasattr(state_tensor, '__getitem__') else state_tensor,
            "full_shape": list(state_tensor.shape) if hasattr(state_tensor, 'shape') else None,
            "ready_for_exchange": True
        }
        
        # Write to bridge buffer
        bridge_buffer = Path("F:\\ai_bridge\\buffers\\layer_exchange.json")
        with open(bridge_buffer, 'w') as f:
            json.dump(exchange_data, f, indent=2)
            
        return exchange_data

# Example usage
if __name__ == "__main__":
    storage = LayerStateStorage()
    
    # Example: Save a layer state
    dummy_state = torch.randn(256, 512)
    storage.save_layer_state("pattern_recognition", dummy_state, 
                           metadata={"training_step": 1000})
    
    # Example: Prepare for bridge exchange
    exchange_data = storage.exchange_through_bridge("pattern_recognition", dummy_state)
    print(f"Layer state ready for bridge exchange: {exchange_data['layer_name']}")
