#!/bin/bash
echo "Starting your PyTorch Environment for Sand Mining Detection..."
sudo docker run --runtime nvidia -it --rm --network host \
  -v "$(pwd)":/workspace \
  -w /workspace \
  dustynv/pytorch:1.10-r32.7.1


