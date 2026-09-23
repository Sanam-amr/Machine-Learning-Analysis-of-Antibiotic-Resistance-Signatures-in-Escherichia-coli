\# Genomic Prediction of Antimicrobial Resistance Phenotypes in \*Escherichia coli\*



\## Research Question



Can genomic features of \*Escherichia coli\* be used to predict phenotypic antimicrobial resistance?



\## Objectives



\* Analyze publicly available \*E. coli\* genomic and antimicrobial susceptibility data.

\* Identify genomic features associated with antimicrobial resistance.

\* Investigate genotype–phenotype relationships.

\* Develop machine-learning models for antimicrobial resistance prediction.

\* Identify important genomic features for resistance prediction.



\## Project Overview



This project uses publicly available genomic and antimicrobial susceptibility data from \*E. coli\* isolates.



The current analysis focuses on predicting phenotypic ciprofloxacin resistance from genomic features.



Genomic annotation data were converted into binary features based on the presence or absence of each feature. Statistical association analysis and machine-learning methods were then used to investigate the relationship between genomic features and ciprofloxacin resistance.



\## Dataset



The dataset contains:



\* 721 \*E. coli\* isolates

\* 275 ciprofloxacin-resistant isolates

\* 446 ciprofloxacin-susceptible isolates

\* 182 genomic features



The genomic features were represented as binary variables:



\* 1 = feature present

\* 0 = feature absent



\## Methods



The analysis included:



1\. Genomic feature extraction and preparation

2\. Feature frequency analysis

3\. Fisher's exact test for genomic feature–phenotype associations

4\. Benjamini–Hochberg false discovery rate correction

5\. Stratified training/test split

6\. Logistic Regression as a baseline model

7\. Random Forest classification

8\. 5-fold stratified cross-validation

9\. Independent test-set evaluation

10\. Random Forest feature importance analysis

11\. Feature correlation and co-occurrence analysis

12\. Reduced-feature sensitivity analysis



\## Machine-Learning Results



The dataset was divided into:



\* Training set: 576 isolates

\* Test set: 145 isolates



The test set was kept separate during model development and was used only for final evaluation.



The Random Forest model achieved the following results on the independent test set:



| Metric            | Result |

| ----------------- | -----: |

| Accuracy          | 93.79% |

| Balanced Accuracy | 92.88% |

| Precision         | 94.23% |

| Recall            | 89.09% |

| Specificity       | 96.67% |

| F1 Score          | 91.59% |

| ROC-AUC           | 96.58% |



Confusion matrix:



```text

&#x20;                Predicted

&#x20;                Susceptible  Resistant



Actual Susceptible     87         3

Actual Resistant        6        49

```



\## Important Genomic Features



Several genomic features showed strong associations with ciprofloxacin resistance and were also important in the Random Forest model.



Examples include:



\* `parC\_S80I`

\* `gyrA\_D87N`

\* `gyrA\_S83L`

\* `parC\_E84V`

\* `parE\_I529L`

\* `ptsI\_V25I`

\* `uhpT\_E350Q`



The agreement between statistical association and machine-learning feature importance provided additional evidence that these features carried useful predictive information in this dataset.



These results indicate association and predictive importance. They do not establish that individual features cause antimicrobial resistance.



\## Feature Correlation Analysis



Several genomic features showed high correlation with each other. For example:



\* `ptsI\_V25I` and `parE\_I529L`: 0.996

\* `parC\_S80I` and `gyrA\_D87N`: 0.966

\* `qacEdelta1` and `sul1`: 0.964



A reduced-feature Random Forest analysis was performed as a sensitivity analysis.



After removing highly correlated features, the test ROC-AUC remained similar:



\* Full feature model: 0.9658

\* Reduced feature model: 0.9655



The other performance measures decreased slightly. This analysis suggests that the overall predictive performance was relatively robust to this feature-reduction approach.



\## Figures



The repository includes figures for:



\* Random Forest confusion matrix

\* Random Forest ROC curve

\* Random Forest feature importance

\* Integrated feature analysis

\* Key feature prevalence

\* Genomic feature PCA



\## Limitations



This is an exploratory computational study.



The analysis is limited to:



\* one public dataset

\* one antibiotic phenotype in the current machine-learning analysis

\* genomic features available in the dataset

\* a relatively limited number of isolates compared with larger genomic surveillance datasets



Genomic features can also be correlated because of shared genetic backgrounds. Therefore, machine-learning feature importance should not be interpreted as evidence of causation.



The model has not been externally validated and should not be considered a clinically validated prediction tool.



\## Future Work



Future work could include:



\* validation using independent \*E. coli\* datasets

\* analysis of additional antibiotics

\* more detailed population-structure analysis

\* phylogenetic analysis

\* investigation of genotype–phenotype discordant isolates

\* development of population-aware approaches for antimicrobial resistance prediction



\## Project Status



\*\*Initial analysis completed — September 2026\*\*



The current project includes genomic feature analysis, statistical association testing, machine-learning prediction, model evaluation, feature interpretation, and sensitivity analysis.



Further validation and extension would be required for a broader study.



\## Tools



\* Python

\* pandas

\* NumPy

\* SciPy

\* scikit-learn

\* statsmodels

\* Matplotlib



\## Reproducibility



The analysis scripts are organized in the `scripts/` directory.



Results are stored in the `results/` directory and figures are stored in the `figures/` directory.



The analysis can be reproduced by running the scripts in the order described in the project workflow.



\## Disclaimer



This project is for research and educational purposes. The model is not intended for clinical decision-making.



