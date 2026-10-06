# Third-party code and resources

The pretraining transformer, optimizer, and data-shard code derive from Keller Jordan's **modded-nanoGPT**, with subsequent research modifications. The upstream MIT license and copyright notice are retained in [licenses/modded-nanoGPT-MIT.txt](licenses/modded-nanoGPT-MIT.txt). Our research modifications are licensed under Apache 2.0; the upstream material retains its MIT terms.

The bundled **Cut Cross-Entropy** implementation derives from Apple Inc.'s Cut Cross-Entropy library. Original file headers and its separate [Apple license](pretraining/cut_cross_entropy/LICENSE) are retained. The release retains the standard kernels used for base-token cross entropy.

The post-hoc code subclasses Hugging Face Transformers model implementations; Transformers itself is an external dependency rather than a vendored package.

UniMorph resources, Hugging Face model weights, FineWeb/FineWeb-2/FineWeb-Edu data, and evaluation datasets are external downloads with their own licensing and access terms. They are not redistributed by this code repository.
