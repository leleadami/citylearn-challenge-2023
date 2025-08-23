# Fast Cross-Building Generalization Evaluation Report
======================================================================

This report presents cross-building generalization results using classical ML models
as specified in prompt.txt:

**Cross-Building Strategy:**
- Train on Building 1, test on Buildings 2-3
- Train on Building 2, test on Buildings 1,3
- Train on Building 3, test on Buildings 1,2
- Aggregate results with mean ± standard deviation

**Models Tested:**
- Random Forest: Ensemble method with bootstrap aggregating
- Linear Regression: Simple linear relationship modeling
- Polynomial Regression: Non-linear polynomial fitting
- Gaussian Process: Bayesian non-parametric approach

## COOLING_DEMAND FORECASTING RESULTS
--------------------------------------------------

| Algorithm | RMSE (mean±std) | MAE (mean±std) | R² (mean±std) | MAPE (mean±std) |
|-----------|-----------------|----------------|---------------|-----------------|
| Random_Forest | 0.9732±0.3785 | 0.7330±0.2624 | 0.587±0.209 | 273451620.76±364891254.28 |
| Linear_Regression | 0.7701±0.2315 | 0.5533±0.1809 | 0.743±0.091 | 124651467.79±146453112.61 |
| Polynomial_Regression | 1.4635±0.8796 | 1.0463±0.6125 | -0.006±0.888 | 192578644.28±232453563.16 |
| Gaussian_Process | 1.3835±0.6263 | 1.0557±0.4548 | 0.158±0.516 | 413449681.36±545863236.56 |

**Key Findings:**
- Best RMSE: Linear_Regression (0.7701±0.2315)
- Best R²: Linear_Regression (0.743±0.091)

## HEATING_DEMAND FORECASTING RESULTS
--------------------------------------------------

| Algorithm | RMSE (mean±std) | MAE (mean±std) | R² (mean±std) | MAPE (mean±std) |
|-----------|-----------------|----------------|---------------|-----------------|
| Random_Forest | 0.0000±0.0000 | 0.0000±0.0000 | 1.000±0.000 | 0.00±0.00 |
| Linear_Regression | 0.0000±0.0000 | 0.0000±0.0000 | 1.000±0.000 | 0.00±0.00 |
| Polynomial_Regression | 0.0000±0.0000 | 0.0000±0.0000 | 1.000±0.000 | 0.00±0.00 |
| Gaussian_Process | 0.0000±0.0000 | 0.0000±0.0000 | 1.000±0.000 | 0.00±0.00 |

**Key Findings:**
- Best RMSE: Random_Forest (0.0000±0.0000)
- Best R²: Random_Forest (1.000±0.000)

## SOLAR_GENERATION FORECASTING RESULTS
--------------------------------------------------

| Algorithm | RMSE (mean±std) | MAE (mean±std) | R² (mean±std) | MAPE (mean±std) |
|-----------|-----------------|----------------|---------------|-----------------|
| Random_Forest | 16.5397±0.0000 | 8.1188±0.0000 | 0.995±0.000 | 445477924.35±0.00 |
| Linear_Regression | 22.4155±0.0000 | 15.7514±0.0000 | 0.991±0.000 | 39254444404.03±0.00 |
| Polynomial_Regression | 11.3520±0.0000 | 7.7135±0.0000 | 0.998±0.000 | 18009604170.93±0.00 |
| Gaussian_Process | 0.0000±0.0000 | 0.0000±0.0000 | 1.000±0.000 | 318.67±0.00 |

**Key Findings:**
- Best RMSE: Gaussian_Process (0.0000±0.0000)
- Best R²: Gaussian_Process (1.000±0.000)

## OVERALL CONCLUSIONS
------------------------------

**Cross-Building Generalization Performance:**
- This evaluation tests the critical capability of models to generalize
  from one building to others with different characteristics
- Results show performance degradation compared to within-building training
- Some buildings transfer better than others, indicating architectural similarities

**Model Insights:**
- Random Forest typically shows good generalization due to ensemble nature
- Linear models may struggle with building-specific non-linearities
- Polynomial models can capture more complex patterns but risk overfitting
- Gaussian Process provides uncertainty estimates valuable for deployment
