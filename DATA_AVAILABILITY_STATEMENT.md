# Data Availability Statement

## Overview

This study utilized multiple publicly available single-cell RNA sequencing (scRNA-seq) datasets to evaluate the Cytoneighbor framework across diverse biological contexts. All datasets are stored in AnnData format (.h5ad files) and are available through established data repositories or have been processed from publicly accessible sources.

## Primary Datasets

### 1. Diabetic Kidney Disease Dataset
**File:** `Diabetic Kidney Disease.h5ad`
**Source:** Processed from multiple GEO datasets including GSE131882
**Description:** Single-nucleus RNA sequencing data from human kidney cortex samples comparing healthy donors with diabetic kidney disease (DKD) patients.
**Cells:** 39,176 cells
**Genes:** 27,980 genes
**Cell Types:** 13 major cell types including epithelial cells of proximal tubule, podocytes, endothelial cells, and immune cells
**Reference:** 
- Wilson PC, Wu H, Kirita Y, et al. The single-cell transcriptomic landscape of early human diabetic nephropathy. Proc Natl Acad Sci U S A. 2019;116(39):19619-19625. doi:10.1073/pnas.1908706116
- GEO Accession: GSE131882

### 2. Human Immune Cell Atlas
**File:** `Human immune.h5ad`
**Description:** Comprehensive single-cell atlas of human peripheral blood mononuclear cells (PBMCs) across different age groups and conditions.
**Cell Types:** Multiple immune cell populations including CD4+ T cells, CD8+ T cells, NK cells, NKT cells, B cells, monocytes, and dendritic cells
**Annotation Column:** `final_annotation`
**Batch Column:** `batch`
**Reference:** Based on multiple studies including the Human Cell Atlas project and Allen Institute for Immunology datasets

### 3. Pancreatic Islet Cell Dataset
**File:** `Pancreas.h5ad`
**Description:** Single-cell transcriptomic data from human pancreatic islets, including both healthy and type 2 diabetic donors.
**Cells:** 16,382 cells
**Genes:** 18,771 genes
**Cell Types:** Multiple pancreatic cell types including alpha, beta, delta, gamma, epsilon cells, and acinar cells
**Reference:** 
- Baron M, Veres A, Wolock SL, et al. A Single-Cell Transcriptomic Map of the Human and Mouse Pancreas Reveals Inter- and Intra-cell Population Structure. Cell Syst. 2016;3(4):346-360.e4. doi:10.1016/j.cels.2016.08.011
- Segerstolpe Å, Palasantza A, Eliasson P, et al. Single-Cell Transcriptome Profiling of Human Pancreatic Islets in Health and Type 2 Diabetes. Cell Metab. 2016;24(4):593-607. doi:10.1016/j.cmet.2016.08.020

### 4. C. elegans Neuronal Gene Expression Map (CeNGEN)
**File:** `CeNGEN.h5ad`
**Description:** Comprehensive single-cell RNA sequencing atlas of the complete C. elegans nervous system.
**Source:** CeNGEN Project (https://www.cengen.org/)
**Cell Types:** 128 neuronal classes and additional non-neuronal cell types
**Reference:**
- Taylor SR, Santpere G, Weinreb A, et al. Molecular topography of an entire nervous system. Cell. 2021;184(16):4329-4347.e23. doi:10.1016/j.cell.2021.06.023
- Hammarlund M, Hobert O, Miller DM 3rd, Sestan N. The CeNGEN Project: The Complete Gene Expression Map of an Entire Nervous System. Neuron. 2018;99(2):430-433. doi:10.1016/j.neuron.2018.07.042

### 5. OpenProblems NeurIPS 2021 CITE-seq Dataset
**File:** `GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad`
**Description:** Bone marrow mononuclear cells (BMMC) with simultaneous measurement of gene expression and surface proteins using CITE-seq technology.
**Source:** Open Problems in Single-Cell Analysis NeurIPS 2021 competition
**GEO Accession:** GSE194122
**Reference:**
- Luecken MD, Burkhardt DB, Cannoodt R, et al. A sandbox for prediction and integration of DNA, RNA, and proteins in single cells. Nat Methods. 2022;19(10):1222-1232. doi:10.1038/s41592-022-01605-2

### 6. Zebrafish Embryonic Development Dataset
**File:** `Zebrafish embryonic cells.h5ad`
**Description:** Single-cell transcriptomic atlas of zebrafish embryonic development covering multiple developmental stages.
**Cell Types:** Various embryonic cell types across different developmental stages
**Reference:** Based on published zebrafish developmental atlases

## Spatial Transcriptomics Datasets

### 7. Human Limb Development Spatial Dataset
**Directory:** `data/Visium/`
**Files:**
- `human_limb_scRNAseq.h5ad` - Single-cell reference data
- `PCW8.1WSSS_THYst9383359.h5ad` - Spatial transcriptomics query data
- `lpm_scRNAseq.h5ad` - Lateral plate mesoderm reference
- `skm_scRNAseq.h5ad` - Skeletal muscle reference

**Description:** Spatial transcriptomics data from human limb development using 10x Genomics Visium technology, paired with single-cell reference datasets.
**Cell Types:** Multiple developmental cell types including chondrocytes, muscle cells, fibroblasts, and various mesenchymal populations
**Spatial Resolution:** 55μm spot diameter with spatial coordinates preserved
**Reference:** Based on Human Cell Atlas limb development studies

## Additional Datasets

### 8. Mouse Kidney Dataset
**File:** `Mouse_Kidney_mod1.h5ad`
**Description:** Single-cell transcriptomic data from mouse kidney for cross-species validation studies.

### 9. Agarwal et al. Nature Communications Dataset
**File:** `Agarwal_NatCommun_2020_32826893.h5ad`
**Reference:** Agarwal D, Sandor C, Volpato V, et al. A single-cell atlas of the human substantia nigra reveals cell-specific pathways associated with neurological disorders. Nat Commun. 2020;11(1):4183. doi:10.1038/s41467-020-17876-0

### 10. Kamath et al. Nature Neuroscience Dataset
**File:** `Kamath_NatNeurosci_2022_35513515.h5ad`
**Reference:** Kamath T, Abdulraouf A, Burris SJ, et al. Single-cell genomic profiling of human dopamine neurons identifies a population that selectively degenerates in Parkinson's disease. Nat Neurosci. 2022;25(5):588-595. doi:10.1038/s41593-022-01061-1

## Data Processing and Quality Control

All datasets underwent standardized preprocessing including:
- Quality control filtering (minimum genes per cell, minimum cells per gene)
- Normalization to counts per million (CPM) where appropriate
- Log transformation for downstream analysis
- Gene symbol standardization and Ensembl ID mapping where necessary
- Cell type annotation validation and harmonization

## Data Access and Licensing

- **Public Datasets:** All primary datasets are derived from publicly available sources with appropriate citations
- **GEO Datasets:** Available through NCBI Gene Expression Omnibus (https://www.ncbi.nlm.nih.gov/geo/)
- **Human Cell Atlas:** Data accessed through the Human Cell Atlas Data Portal (https://data.humancellatlas.org/)
- **CeNGEN:** Available through the CeNGEN portal (https://www.cengen.org/)
- **OpenProblems:** Available through the Open Problems in Single-Cell Analysis platform (https://openproblems.bio/)

## Ethical Considerations

All human datasets used in this study:
- Were obtained from publicly available repositories
- Had appropriate ethical approval from the original studies
- Comply with relevant data sharing agreements and privacy regulations
- Do not contain personally identifiable information

## Data Format and Structure

All datasets are stored in AnnData format (.h5ad) with standardized structure:
- **X:** Gene expression matrix (cells × genes)
- **obs:** Cell metadata including cell types, batch information, and quality metrics
- **var:** Gene metadata including gene symbols and Ensembl IDs
- **obsm:** Dimensionality reduction embeddings (PCA, UMAP) where available
- **uns:** Dataset-specific metadata and processing parameters

## Computational Requirements

- **Storage:** Approximately 15-20 GB total for all datasets
- **Memory:** Minimum 32 GB RAM recommended for processing large datasets
- **Software:** Python 3.8+, scanpy, anndata, pandas, numpy

## Contact Information

For questions regarding data processing or access, please refer to the original publications and data repositories cited above. For questions specific to this study's implementation, please contact the corresponding authors.

## Acknowledgments

We acknowledge the original authors and consortiums who generated and made these valuable datasets publicly available, enabling comparative studies like this one. Special thanks to:
- The Human Cell Atlas consortium
- The CeNGEN project team
- The Open Problems in Single-Cell Analysis community
- All individual research groups who contributed datasets to public repositories

---

*This data availability statement follows best practices for reproducible research and complies with journal requirements for data transparency and accessibility.*