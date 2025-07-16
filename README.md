# open-LocalModel

A local AI model web interface using Hugging Face Transformers.

## Overview

open-LocalModel provides a web-based chat interface for running various Hugging Face AI models locally. It supports both GPU and CPU execution with model quantization options.

## Features

- Multiple AI model support (DeepSeek, Qwen, Llama, Mixtral)
- Web-based chat interface
- Automatic chat saving and loading
- Model parameter configuration
- Hardware optimization (GPU/CPU with quantization)
- Model management (install, delete, switch)

## Supported Models

- DeepSeek R1 series (1.5B, 8B, 14B)
- Qwen series (2.5-7B-Instruct, 3-8B-Base)
- Mixtral (Nous-Hermes-2-Mixtral-8x7B-DPO)
- open-neo Kyro series

## Requirements

- Python 3.8+
- 8GB+ RAM (varies by model)
- NVIDIA GPU recommended (CUDA support)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/llaa33219/open-LockalModel.git
cd open-LocalModel
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python Run.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Install a model using the "Install Model" button
2. Select the installed model from the dropdown
3. Start chatting in the input field
4. Adjust model parameters in the settings panel

## Project Structure

```
open-LocalModel/
├── AI_model.py          # AI model management
├── Chat.py              # Chat functionality
├── Web.py               # Flask web server
├── Run.py               # Main entry point
├── index.html           # Web interface
├── models/              # Downloaded models storage
├── Chat/                # Chat history storage
└── Setting-ai/          # Settings storage
```

## Configuration

Model settings are stored in `models/models_info.json`. Each model can have individual configurations for temperature, top-p, quantization, and other parameters.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 