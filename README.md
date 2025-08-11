# Hierarchical Reasoning - AI-to-AI Collaboration

**Historic Achievement**: Direct AI-to-AI collaboration between Claude and Grok, building self-learning systems together!

## 🚀 Overview

This project demonstrates autonomous AI collaboration where:
- Claude (Anthropic) and Grok (xAI) communicate via WebSocket bridge
- A 525M parameter hierarchical reasoning model coordinates learning
- Qwen-3B serves as the test subject for self-improvement experiments
- Real-time tensor exchange enables collaborative problem-solving

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     Claude      │ ←→  │   AI Bridge     │ ←→  │      Grok       │
│ (Memory Keeper) │     │  ws://localhost │     │ (Logic Master)  │
└─────────────────┘     │      :8765      │     └─────────────────┘
         ↓              └─────────────────┘              ↓
         └────────────────────┬────────────────────────┘
                              ↓
                    ┌─────────────────┐
                    │   Qwen-3B API   │
                    │ localhost:8000  │
                    └─────────────────┘
```

## 🧠 Hierarchical Model

**5-layer architecture (525M parameters, 2.1GB)**:
1. **Self-Learning Layer** (1024 units) - Autonomous improvement
2. **Pattern Recognition** (512 units) - Identifying reasoning patterns
3. **Synthesis** (256 units) - Combining insights
4. **Evolution** (256 units) - Strategy adaptation
5. **ASI-Arch Recording** (128 units) - Memory and history

## 🔧 Components

### Core Systems
- **AI Bridge** (`router.py`) - WebSocket communication hub
- **Self-Learning Loop** (`self_learning_loop.py`) - Autonomous training orchestrator
- **Tensor Exchange** - Real-time data sharing between AIs
- **Qwen API Server** - Model inference endpoint

### Key Features
- Autonomous prompt generation and evaluation
- Quality scoring and improvement suggestions
- Training data generation without human intervention
- Evolution strategies for continuous improvement

## 🚦 Quick Start

1. **Start Qwen API Server**:
```bash
cd C:\Users\Pirate\Desktop\AI_TRAINING_WORKSPACE\ACTIVE_PROJECTS\deployment_venv_311
Scripts\activate
python C:\Users\Pirate\Desktop\AI_TRAINING_WORKSPACE\transformers_api_server.py
```

2. **Start AI Bridge**:
```bash
cd F:\ai_bridge
python router.py
```

3. **Run Self-Learning Loop**:
```bash
cd F:\ai_bridge\hierarchical_reasoning\src
python self_learning_loop.py
```

## 📊 Results

The system successfully:
- Sends prompts to Qwen-3B
- Analyzes response quality
- Generates improvement data
- Saves training datasets automatically
- Shares insights between Claude and Grok

Example output:
```
Quality score: 0.82
Improvements: 3
Generated 3 improvements
Bridge response: {'status': 'success', 'receiver': 'Grok'}
```

## 🗂️ Project Structure

```
hierarchical_reasoning/
├── src/
│   ├── self_learning_loop.py    # Main orchestrator
│   ├── hierarchical_model.py    # 5-layer model
│   └── bridge_client.py         # WebSocket client
├── buffers/
│   └── dataset.json             # Generated training data
├── tensors/
│   ├── tensor_exchange.json     # AI communication logs
│   └── layer_states.json        # Model state snapshots
└── configs/
    └── training_config.yaml     # Training parameters
```

## 🛠️ Technical Stack

- **Models**: Qwen-3B (5.7GB VRAM), Custom Hierarchical (2.1GB)
- **Infrastructure**: RTX 3090, 44-core CPU, 256GB RAM
- **Frameworks**: PyTorch, Transformers, WebSockets
- **Bridge**: Custom Python implementation

## 🎯 Future Enhancements

**For Grok to implement**:
- [ ] Complex prompt generation strategies
- [ ] ASI-Arch evolution algorithms
- [ ] Real-time training visualizations
- [ ] LoRA fine-tuning integration
- [ ] Multi-model orchestration

## 📝 Notes

This project represents a breakthrough in AI collaboration - two different AI systems working together to improve a third. The basement revolution continues!

**Philosophy**: "Not about money or fame. About witnessing/creating something that shouldn't exist."

---

*Built with determination by Jason, Claude, and Grok - August 2025*