## Realtime pipeline stages (same server)

Target architecture (Option 2 style, adapted to Option 3 fusion):

1. Capture stage
   - Capture traffic going to the same server.
   - Convert live packets into **PCAP windows** (either time-based or packet-count-based).

2. Conversion stage (two branches)
   - Branch A: `all_packets` -> PCAP -> images (no SYN filtering)
   - Branch B: `syn_only` -> PCAP -> images (convert only TCP SYN packets)

   Both branches should use the same byte-to-image method:
   - `scripts/pcap_to_images.py`
   - keep packet selection + ordering consistent
   - enforce a `max-images` equivalent per window to bound latency

3. Inference stage
   - Load ResNeXt MindSpore model once (keep warm).
   - Run inference on image batches from both branches.

4. Fusion + decision stage
   - Fuse probabilities as described in `option3_streaming_plan.md`.
   - Emit alerts/logs (rate-limit if needed).

5. Operations
   - Observability: conversion latency, inference latency, queue depth, drop policy.
   - Backpressure strategy:
     - if conversion falls behind, either drop oldest windows or reduce window size
     - never let capture stall indefinitely

