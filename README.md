# microbial-steen-project

Type of project: Analysis
Group Members: Ashley Babjac (Solo)

Main Idea:
- It would be a major scientific finding to identify features of microbial genomes that are associated with that genome belonging to an uncultured organism
- However, there is major phylogenetic bias in the organisms that are uncultured. All 200+ microbial phyla contain uncultured species, but most phyla do not contain cultured species.
- Metabolism is strongly correlated to taxonomy, so therefore there’s a strong correlation between whether a species has been cultured and what its genome contains that probably isn’t related to whether it has been cultured
- So: is there a way to control for phylogeny, when selecting features that distinguish between cultured and uncultured microbes?

Description: This project is investigating the ability to use measures of abundance combined with metadata features to predict cultured/uncultured microbes. Elements of this project include:
- Creating an analysis pipeline
- Using a language other than R (Python)
- Parsing data from tsv/csv files
- Cleaning data
- Successfully predicting cultured/uncultured microbes using combinations of models such as PCA/SVM/autoencoders

Phase 1: Preliminary analysis using the TARA data completed. The PowerPoint slides detailing this phase of the analysis are under the "presentations" folder of this repo.

Phase 2: In progress. Recreate a pipeline similar to the TARA analysis using the annoted genomes from the GEM dataset provided by Taylor Royalty. End goal of predicting cultured/uncultured microbes and understanding useful features derived from the analysis.

Current conclusions

TARA:
- TARA is easily modeled and represented by both PCA and autoencoders
- Both perform well (average 90% accuracy) when predicting the ocean region
- AE do a slightly better job representing the data

GEM:
- Data is large and noisy
- Difficult to represent this with few dimensions but autoencoders do a good job reducing noise and sparsity in our features
- Both PCA+AE run with SVM do not predict well (one is 50/50 the other is all false positives)
- Need more time tuning the models to extract the important signals and better predict cultured/uncultured
- Will be doing this with "vectorizing" the GEM data then piping through autoencoders (may switch to variational or deep AEs)
