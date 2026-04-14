## Option 3 streaming inference plan (no labeling at runtime)

Goal: real-time SYN-flood detection where the model decides `ddos` vs `normal`, while keeping the runtime input style aligned with how training created images.

### Assumptions from the training code
In this repo:
- both `ddos` and `normal` training images are generated from **all packets**

In a real-time stream we do not have the offline PCAP label, so we cannot replicate “ddos => syn-only, normal => all” perfectly.

### Corrected streaming inference
For each live capture window:
1. Convert the captured traffic into `all_packets` packet-derived 224x224 grayscale images.
2. Run the trained ResNeXt model once on that image batch.
3. Decide `ddos` vs `normal` using the model output (argmax over mean probabilities).

### What we still need to choose
1. Streaming window definition:
   - time-based (e.g., 1s / 2s windows)
   - or packet-count-based (e.g., N packets per window)
2. How to handle `--max-images`:
   - training conversion stops at `max-images` (default 200)
   - real-time windows should define an equivalent cap for latency
3. Decision latency target and acceptable false positive/negative trade-offs:
   - threshold must be tuned on validation

