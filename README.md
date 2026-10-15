# ArcForge 1.3

> **In short:** ArcForge is a local, open-source AI Beat Saber map creator. It analyzes songs, generates maps using a locally trained model, lets you edit notes and lighting, continue sections with AI, and create supported modded content like Chroma and Noodle Extensions. You can also import BeatSaver `.bplist` playlists to build your own training dataset.

![AI cat clapping](https://media1.tenor.com/m/m3r_8NhxM9MAAAAC/cat-clapping-ai-cat-clapping.gif)

Disclaimer!! This project was made for fun. Its clearly vibe coded with GPT 6.1 sol + my own super intelligence.

## What can it do?

- **AI mapping** using a model trained on human-made Beat Saber maps.
- **Audio analysis** with ArcAudio, plus mapping regions for different styles and NPS goals.
- **Training data import** from map ZIPs and BeatSaver playlists.
- **Map editor** for notes, bombs, walls, arcs, chains, BPM, and tempo changes.
- **AI continuation** to fill selected gaps in an existing map.
- **Lighting and modding tools** for supported content such as Chroma and Noodle Extensions.
- **V2, V3, and V4 support**, with map previews and ZIP export.

ArcForge runs locally and doesn't need an AI API key. CPU is supported, with optional NVIDIA CUDA acceleration where available.

## Get started

Already have an installed build? Run **`Launch ArcForge.cmd`**, then open [http://127.0.0.1:8769](http://127.0.0.1:8769).

For installation, source setup, training, ArcAudio, mapping, mod support, troubleshooting, and everything else, **read the [ArcForge Wiki](https://github.com/egkrypt/ArcForge/wiki)**.

> **Heads-up:** The public source release doesn't include my private training dataset, trained checkpoints, model weights, songs, or projects. To use AI generation from a source release, import at least three usable map archives and train the local model first. You can still import and edit maps without training.

## Contributing

ngl it's still a work in progress, so feedback and contributors are welcome. Contact me on Discord: **`g9es`**.

**[Read the Wiki →](https://github.com/egkrypt/ArcForge/wiki)**
