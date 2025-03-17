# Coupled Exploration of High-Dimensional Images and Hierarchical Embeddings

Analysis plugin for coupling views of HSNE embeddings and high dimensional images in the [ManiVault](https://github.com/ManiVaultStudio/core) visual analytics framework.

```git
git clone --recurse-submodules https://github.com/ManiVaultStudio/ImageEmbeddingCoupling.git
```

Two plugins are build: An analysis plugin that computes HSNE embeddings, and a slightly modified image viewer based on the [ManiVault image viewer](https://github.com/ManiVaultStudio/ImageViewerPlugin/tree/feature/VMV2023) by Thomas Kroes.
The HSNE computation performed with [HDILibSlim](https://github.com/alxvth/HDILibSlim), a slightly modified [HDILib](https://github.com/biovault/HDILib) by Nicola Pezzotti and Thomas Höllt.
Build the project with the same generator as the ManiVault core, see instructions [here](https://github.com/ManiVaultStudio/core). Use [vcpkg](https://github.com/microsoft/vcpkg/) for other dependencies.

## References
This plugin implements methods presented in **Interactions for Seamlessly Coupled Exploration of High-Dimensional Images and Hierarchical Embeddings** (2023), published at [Vision, Modeling, and Visualization 2023](https://doi.org/10.2312/vmv.20231227) ([pdf](https://diglib.eg.org/bitstream/handle/10.2312/vmv20231227/063-070.pdf)). The conference talk recording and other supplemental material are available [here](https://graphics.tudelft.nl/Publications-new/2023/VLEVH23/).

```
@inproceedings{Vieth23,
  title        = {{Interactions for Seamlessly Coupled Exploration of High-Dimensional Images and Hierarchical Embeddings}},
  author       = {Vieth, Alexander and Lelieveldt, Boudewijn and Eisemann, Elmar and Vilanova, Anna and H\"{o}llt, Thomas},
  year         = 2023,
  booktitle    = {Vision, Modeling, and Visualization},
  publisher    = {The Eurographics Association},
  doi          = {10.2312/vmv.20231227},
  isbn         = {978-3-03868-232-5},
  editor       = {Guthe, Michael and Grosch, Thorsten}
}
```
