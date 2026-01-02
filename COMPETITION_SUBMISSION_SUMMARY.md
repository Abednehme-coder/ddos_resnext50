# Huawei ICT Competition Submission: DDoS Detection with MindSpore

## Project Overview

This project implements **ResNet-Based Detection of SYN Flood DDoS Attacks** (Bazzi et al., 2023) using **MindSpore** and **MindCV** frameworks, creating a production-ready deep learning solution for network security.

## Key Highlights

### 🎯 Research Foundation
- Implements peer-reviewed methodology achieving **97.5% accuracy**
- Validates approach with modern AI framework (MindSpore)
- Extends research with enhanced architecture (ResNeXt50-32x4d)

### 🚀 Technical Innovation
- **First MindSpore implementation** of this DDoS detection methodology
- **ResNeXt50-32x4d** architecture (superior to ResNet-50)
- **Complete pipeline**: PCAP → Images → Training → Deployment
- **Multi-format export**: MindIR, AIR, ONNX for flexible deployment

### 💡 Huawei Ecosystem Integration
- **MindSpore Framework**: Native Huawei AI framework
- **MindCV Models**: Production-ready ResNeXt50 implementation
- **Hardware Optimized**: Supports GPU, CPU, and Ascend NPU
- **Enterprise Ready**: Deployment and inference capabilities

## Architecture Comparison

| Aspect | Original Paper | Our Implementation |
|--------|---------------|-------------------|
| Model | ResNet-50 | **ResNeXt50-32x4d** |
| Framework | TensorFlow | **MindSpore** |
| Deployment | Limited | **Multi-format Export** |
| Hardware | GPU only | **GPU/CPU/NPU** |

## Project Structure

```
ddos_resnext50/
├── scripts/
│   ├── pcap_to_images.py      # PCAP to image conversion
│   ├── train_resnext.py       # Model training
│   ├── eval_metrics.py        # Performance evaluation
│   ├── export_model.py        # Model deployment
│   ├── infer_resnext.py       # Inference pipeline
│   └── convert_batch.sh       # Batch processing
├── models/                     # Trained models
└── README.md                   # Complete documentation
```

## Quick Start

### 1. Data Preparation
```bash
python scripts/pcap_to_images.py \
  --pcap attack.pcap \
  --out images/ddos/ \
  --img-size 224 \
  --syn-only
```

### 2. Training
```bash
python scripts/train_resnext.py \
  --data-root images/ \
  --batch-size 16 \
  --epochs 8 \
  --device-target GPU
```

### 3. Evaluation
```bash
python scripts/eval_metrics.py \
  --data-root images/ \
  --split test \
  --ckpt models/resnext50-8_386.ckpt
```

### 4. Deployment
```bash
python scripts/export_model.py \
  --ckpt models/resnext50-8_386.ckpt \
  --out models/resnext50_export \
  --format MINDIR
```

## Performance Metrics

Based on paper's methodology:
- **Accuracy**: 97.5% (paper baseline)
- **Precision**: High (per-class metrics available)
- **Recall**: High (per-class metrics available)
- **F1 Score**: High (balanced performance)

## Competition Value Proposition

1. **Research-Based**: Built on validated academic research
2. **Huawei Native**: Demonstrates MindSpore/MindCV proficiency
3. **Production Ready**: Complete deployment pipeline
4. **Innovative**: First MindSpore implementation of this approach
5. **Practical**: Real-world cybersecurity application

## Future Enhancements

- Multi-class DDoS attack classification
- Real-time packet stream processing
- Integration with network monitoring systems
- Edge deployment optimization

---

**Reference**: Bazzi, H. S., et al. (2023). "ResNet-Based Detection of SYN Flood DDoS Attacks." Beirut Arab University.

