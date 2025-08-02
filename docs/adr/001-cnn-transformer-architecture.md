# ADR-001: CNN-Transformer Hybrid Architecture for Echo Processing

## Status
Accepted

## Context
EchoLoc-NN requires a deep learning architecture that can effectively process ultrasonic echo patterns for accurate indoor localization. The challenge is to capture both local temporal patterns in echo waveforms and global relationships between multiple sensor channels.

Key requirements:
- Process multi-channel echo data (4+ ultrasonic sensors)
- Handle variable-length echo sequences
- Extract spatial-temporal features from ultrasonic signals
- Achieve real-time inference on edge devices
- Maintain sub-centimeter accuracy

Alternative architectures considered:
1. Pure CNN approach
2. Pure Transformer/attention model
3. LSTM-based recurrent networks
4. CNN-Transformer hybrid

## Decision
We will implement a CNN-Transformer hybrid architecture that combines:
- CNN encoder for local pattern extraction from echo waveforms
- Transformer decoder with time-of-flight aware attention for global relationships
- Specialized positional encoding based on sensor geometry
- Dual output heads for position estimation and uncertainty quantification

## Rationale
**CNN Component Benefits:**
- Excellent at capturing local temporal patterns in 1D echo signals
- Translation invariant for echo features at different time delays
- Computationally efficient for edge deployment
- Well-suited for multi-scale feature extraction through dilated convolutions

**Transformer Component Benefits:**
- Global attention mechanism ideal for relating echoes from multiple sensors
- Self-attention can model complex multipath relationships
- Positional encoding can incorporate sensor geometry
- Parallel processing enables real-time inference

**Hybrid Approach Advantages:**
- Combines local and global feature extraction strengths
- CNN reduces sequence length before Transformer processing
- More parameter-efficient than pure Transformer
- Better convergence than pure attention models on limited data

**Time-of-Flight Aware Design:**
- Attention bias based on acoustic propagation delays
- Sensor geometry embedded in positional encoding
- Physics-informed architecture improves generalization

## Consequences

**Positive:**
- Superior accuracy compared to single-architecture approaches
- Efficient computation suitable for Raspberry Pi deployment
- Interpretable attention weights for debugging
- Extensible to different sensor array configurations
- Strong foundation for transfer learning

**Negative:**
- More complex architecture than single-model approaches
- Requires careful hyperparameter tuning for both components
- Larger model size than pure CNN (though smaller than pure Transformer)
- Additional implementation complexity

**Risks and Mitigations:**
- **Risk**: Overfitting with limited real-world data
  - **Mitigation**: Extensive simulation-based training data generation
- **Risk**: Edge deployment performance
  - **Mitigation**: Model quantization and pruning strategies
- **Risk**: Architecture complexity
  - **Mitigation**: Modular design with clear component interfaces

**Impact on Future Decisions:**
- Training pipeline must support both supervised and self-supervised learning
- Data augmentation strategies must consider both CNN and Transformer components
- Edge optimization tools must handle hybrid architectures
- Transfer learning approaches should leverage pre-trained components

## References
- "Attention Is All You Need" (Vaswani et al., 2017)
- "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020)
- "Deep Learning for Indoor Localization: A Survey" (Chen et al., 2021)
- Internal simulation results comparing architecture alternatives