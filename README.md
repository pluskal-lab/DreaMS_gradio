---
title: DreaMS
emoji: 🔬
colorFrom: yellow
colorTo: pink
sdk: gradio
app_file: app.py
pinned: true
python_version: 3.11
---

## Usage

This repository provides the source code for the DreaMS Gradio app. The app is currently hosted on Hugging Face Spaces: [https://huggingface.co/spaces/anton-bushuiev/DreaMS](https://huggingface.co/spaces/anton-bushuiev/DreaMS).

## Development

To run or debug the app locally, you can use the following commands:

```bash
git clone https://github.com/pluskal-lab/DreaMS_gradio.git
cd DreaMS_gradio
conda create -n dreams_gradio python=3.11
conda activate dreams_gradio
pip install -r requirements.txt
pip install gradio spaces
python app.py
```
