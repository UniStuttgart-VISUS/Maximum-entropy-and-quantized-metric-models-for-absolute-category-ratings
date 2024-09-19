# Model Comparison
In the main document an evaluation of the different models is given where their goodness of fit is determined and predction accuracy is measured with respect to the *VQEG* and *KonIQ-10k* dataset.
Here, we provide a visual comparison of the models and give more details on their individual performance.

## Comparison of G-Test Results with Data Sets
To get a more detailed impression of the models performances, we have a look at the plane of $\Psi$ (*mean opinion score*) and $v$  (*variance*).
We use the *VQEG* and *KonIQ-10k* dataset to create a scatter plot of the stimuli's $\Psi$ and $v$ and color code them by their G-test value of the respective model.

| ![G-test results on VQEG dataset](figures/row_gtest-vqeg.svg) |
| --- |
| *Figure 1* - Scatterplots for the different models showing $\Psi$ and $v$ of the stimuli of the *VQEG* data set and corresponding G-test value as color. |

In Figure 1 it can be seen that all models expose a similar pattern where the stimuli on the sides, i.e., those with little or high mean opinion score and smaller possible variance, have small G-test values (blue color) and are better fitted by the models.
Stimuli located more towards the center tend to be fitted worse by the models.
For this dataset (*VQEG*), all models look equally valid.

| ![G-test results on KONIQ dataset](figures/row_gtest-koniq.svg) |
| --- |
| *Figure 2* - Scatterplots for the different models showing $\Psi$ and $v$ of the stimuli of the *KonIQ-10k* data set and corresponding G-test value as color. |

In Figure 2 the same kind of visualization is used, but with the *KonIQ-10k* dataset which contains a considerably larger number of stimuli.
Here, the models show clearly different patterns.
For example, the GSD model has high G-tests values for stimuli close to $\Psi=3$, whereas other models, like normal, beta, and maxentropy, show high G-test values for stimuli between $\Psi=3$ and $\Psi=4$.
The plot for the logit-logistic model appears to have the lowest G-test values for this data set, which resonates with our findings reported in the main document. 


## Comparison of ACR Probability Vector Outputs

| ![L1-distances of ACR outputs](figures/matrix_l1dist.svg) |
| --- |
| *Figure 3* - Pairwise comparison of the models ACR probability vector output. The L1-distance of the output vectors between two models for inputs from the $\Psi$ - $v$ - plane is color coded. Bright colors indicate areas where the models output differs more strongly. On the diagonal of the matrix plot the model is indicated that is compared in the corresponding row and column. Plots below the diagonal are redundant due to symmetry. | 

more text

| ![Aitchison distances of ACR outputs](figures/matrix_aitchison.svg) |
| --- |
| *Figure 4* - Pairwise comparison of the models ACR probability vector output. The Aitchison distance of the output vectors between two models for inputs from the $\Psi$ - $v$ - plane is color coded. Bright colors indicate areas where the models output differs more strongly. On the diagonal of the matrix plot the model is indicated that is compared in the corresponding row and column. Plots below the diagonal are redundant due to symmetry. |
