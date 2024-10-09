#!/bin/bash
sudo snap install --classic code astral-uv
code --install-extension ms-python.python
code --install-extension ms-python.python
code --install-extension ms-python.black-formatter
code --install-extension ms-toolsai.jupyter
uv sync
