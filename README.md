# ComfyUI-PulseOfMotion

ComfyUI nodes for predicting **Physical FPS (PhyFPS)** from video using the **Visual Chronometer** model.

Based on the paper [**"The Pulse of Motion: Measuring Physical Frame Rate from Visual Dynamics"**](https://arxiv.org/abs/2505.15990) by the [TACO Group](https://github.com/taco-group/Pulse-of-Motion).

PhyFPS measures the true temporal resolution of a video from its visual motion dynamics — independent of the container frame rate. This is useful for detecting AI-generated videos, evaluating video quality, and understanding temporal characteristics.

## Nodes

| Node | Description |
|------|-------------|
| **Load Visual Chronometer** | Loads the VC model checkpoint. Auto-downloads from HuggingFace on first use. Supports device selection (auto/cpu/cuda). |
| **Predict PhyFPS** | Predicts average PhyFPS from video frames using a sliding window. Returns a float and a detailed report. |
| **Predict PhyFPS (Batch)** | Same as above but also returns a per-segment FPS list for analysis. |

## Installation

### Via ComfyUI-Manager (recommended)

Search for **"Pulse of Motion"** in ComfyUI-Manager and click Install.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/akashzeno/ComfyUI-PulseOfMotion.git
cd ComfyUI-PulseOfMotion
pip install -r requirements.txt
```

Restart ComfyUI after installation.

## Model

The checkpoint (`vc_common_10_60fps.ckpt`) is automatically downloaded from [HuggingFace](https://huggingface.co/xiangbog/Visual_Chronometer) on first use and saved to `ComfyUI/models/pulse_of_motion/`.

## Usage

1. Add a **Load Video (Upload)** node (from [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)) to load your video
2. Add a **Load Visual Chronometer** node — select the checkpoint and device
3. Add a **Predict PhyFPS** node — connect `model` from the loader and `IMAGE` from the video loader to `images`
4. Add two **Preview as Text** nodes:
   - Connect `phyfps` to one for the average FPS value
   - Connect `report` to another for the detailed per-segment breakdown
5. Adjust `clip_length` (default 30) and `stride` (default 4) as needed, then queue the prompt

### Parameters

- **clip_length** — Number of frames per analysis clip (default: 30, trained on 30-frame clips)
- **stride** — Step size between clips (default: 4). Lower = more clips = smoother average but slower

## Example Workflow

![workflow](example_workflows/workflow.png)

## Optimizations

This implementation uses **PyTorch SDPA** (`scaled_dot_product_attention`) for the spatial and cross-attention modules, which automatically dispatches to Flash Attention 2 or memory-efficient attention depending on your GPU. This provides identical accuracy with better speed and memory efficiency compared to the original manual attention implementation.

## Credits

- **Paper**: [The Pulse of Motion: Measuring Physical Frame Rate from Visual Dynamics](https://arxiv.org/abs/2505.15990)
- **Original Code**: [taco-group/Pulse-of-Motion](https://github.com/taco-group/Pulse-of-Motion)
- **Model Weights**: [xiangbog/Visual_Chronometer](https://huggingface.co/xiangbog/Visual_Chronometer)

## License

This project wraps the Visual Chronometer model for ComfyUI. Please refer to the [original repository](https://github.com/taco-group/Pulse-of-Motion) for licensing of the model and weights.
