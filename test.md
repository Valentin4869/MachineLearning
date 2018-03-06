
## META
- These notes are my (Henri Lunnikivi) interpretation of the lecture. Lecturer's
  original opinions may differ.
- Document contents: 1.1. Main Points, 1.2. Key Innovations in ML, 2.1. Social
  Trends, 2.2. Guidelines for Researchers, 2.3. Other Highlights.


## Summary

### Main Points
- ML research and industry are diverging, this is bad.
    * Researchers tend to build 1. bigger models, that 2. fit more specific
      nichés.
    * Industry needs 1. small models, that can fit on a cellphone or a smart
      watch, that 2. can be easily updated over cellphone networks, that 3. can
  be used in any environment or by any population.
- They made a network with the separable convolution and the sparse matrix
  multiplication optimization and compared it to other popular networks and to
  some binarized networks, and it's pretty good.
- Next step: build ecosystems for pervasive ML applications, enable on-device
  inference especially.

### Key Innovations in the Field of ML
1. Node pruning (sparse matrix multiplication).
2. SqueezeNet (50x model size reduction vs. AlexNet).
3. Low-precision results (8-bit etc.).
4. Network binarization.
5. MobileNet, other small-footprint nets.
6. Hardware: Diannao and Cnvlutin2.
7. Hardware: front-ends eg. SNPE - Qualcomm.
8. Hardware: TPU, FPGAs.


## Insights

### Social Trends
- Users are more aware of where their data goes => On-device inference is often
  required => On-device inference performance and model size is important.
- Using a model of 500 MB is not practical to update.
- Models made in research often generalize poorly to a large population of users
  and use-cases.

### Guidelines for Researchers
- Algorithms need to be more resource centric: instead of an optimized algorithm
  on one device, there would be an algorithm that's able to dynamically use the
  resources provided by the environment.
- State of the art techniques need to be enabled across all platforms.
- Hardware needs to provide AI friendly optimizations.
- Time-to-market of an ML model needs to be shorter => There needs to be an
  effort into the development of the ecosystem.

### Highlights
- Node Pruning
    * Dense layers take too much space.
    * Before: fully-connected
    * After: inserted-mid layer "dictionary", that is fully-connected to the
      next layer, but sparsely connected to previous layer.
- Convolutions are the Forgotten Bottleneck of Networks.
- Separable Convolution
    * 2-4 % degradation in accuracy
    * Order of magnitude improvement in performance with sep-conv.
    * Another order of magnitude improvement with node pruning and related
      algorithm.
- Binarization. XNOR. Bit Counting.
    * Key innovation: convolution by deterministic binary filters.
        * linear_combination_of(binary_base, coefficients) -> reconstructed
          image.
        * "One extra-forward pass" is required for creating the binary base. No
          need to repeat that in-training though.
- If I understood correctly, the network created by Nic Lane et al. is called
  "DBFnet", and contains the separable convolution and the binary CNN
  optimizations. DBFnet results are competitive w/ best alternatives (despite huge
  performance gains).
    * Outperforms binary networks as well. (XNOR-Net is pretty good though).


## Cool fact: AlphaGo has 1920 CPUs, 280 GPUs, and consumes $3000 in electricity / game.

