## Window -> images mapping (how to feed the model)

In this repo, the PCAP-to-image converter (`scripts/pcap_to_images.py`) works at **packet granularity**:
- `iter_pcap_packets()` yields one packet’s raw bytes at a time
- each yielded packet can produce **one 224x224 image**
- therefore, the number of images generated for a live window is proportional to the number of packets captured for that window

### Recommended mapping for streaming
For each live capture window:
1. Capture packet bytes for the window.
2. Convert into `all_packets` images:
   - run the same byte-to-image function for each packet
   - enforce `max_images_per_view = 200` by taking the **first 200 converted images** if needed
3. Convert into `syn_only` images:
   - for each packet, apply the same TCP SYN filter used by `scripts/pcap_to_images.py`
   - enforce the same `max_images_per_view = 200`
4. Run inference on each view’s image batch.
5. Fuse (average view-mean probabilities, then argmax) as described in `option3_streaming_plan.md`.

### Why we cap per view
The converter’s offline mode has `--max-images` (default 200). Using the same cap in streaming bounds:
- conversion CPU time per window
- inference batch size
- end-to-end latency predictability

