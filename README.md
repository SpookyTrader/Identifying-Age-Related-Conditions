## Introduction
This project is a Kaggle competition originally hosted by InVitro Cell Research. It aims to develop machine learning tools for predicting the absence 
or presence of age-related medical conditions based on measurable health characteristics of patients. This will also enable the discovery and understanding
of the relationship between aged-related conditions and certain health characteristics. The details of the competition can be found in
the link below. Here, I have developed a machine learning pipeline for accomplishing this aims using the dataset obtained from the source below.

Description of the project:
https://www.kaggle.com/competitions/icr-identify-age-related-conditions/overview

Data source:
https://www.kaggle.com/competitions/icr-identify-age-related-conditions/data

## Methodology

An ensemble machine learning pipeline, consisting of XgBoost, Random Forest and LightGBM algorithms, is built as illustrated in Figure 1. Prediction output from the
3 algorithms are aggregated by averaging the predicted probabilities and features rankings before arriving at the final Balanced Log Loss, Balanced Accuracy and 
features rankings. Balanced Log Loss and Balanced Accuracy are 2 metrices used for performance evaluation here.

<p align="center">
  <img src="https://github.com/user-attachments/assets/109985f4-f2fc-47fa-bea1-dffd8b1c799a" alt="Diagram" width="900" height='300'/>
</p>
<p align="center"><em>Figure 1: Ensemble learner derived from 3 Indiviudual Learners.</em></p>


## Key Results

The key prediction results for the final hold-out test dataset are as summarized below in Table 1, Figure 2 and Figure 3.

<p align="center"><strong>Table 1: Prediction Performances of 3 Individual Models and their Ensemble Model.</strong></p>

<table align="center">
  <tr>
    <th>Model</th>
    <th>Balanced Accuracy</th>
    <th>Balanced Log Loss</th>
  </tr>
  <tr>
    <td>XgBoost (1)</td>
    <td>0.9245</td>
    <td>0.3027</td>
  </tr>
  <tr>
    <td>Random Forest (2)</td>
    <td>0.8920 </td>
    <td>0.4362</td>
  </tr>
  <tr>
    <td>LightGBM (3)</td>
    <td>0.9024</td>
    <td>0.3072</td>
  </tr>
    <tr>
    <td>Ensemble</td>
    <td>0.9180</td>
    <td>0.3020 </td>
  </tr>
</table>

<p align="center">
  <img src="https://github.com/user-attachments/assets/c848f243-40fc-41ae-a98d-a89c2b3587ac" alt="Diagram" width="350" height='300'/>
</p>
<p align="center"><em>Figure 2: Receiver Operating Curve - Aggregated Prediction Performance of Ensemble Learner.</em></p>



<p align="center">
  <img src="https://github.com/user-attachments/assets/39d1fb5e-6847-471f-ab29-2809516b2b64" alt="Diagram" width="1200" height='350'/>
</p>
<p align="center"><em>Figure 3: Ranking of Features in Order of Descending Importance derived from aggregated results of the Ensemble Learner.</em></p>

## Conclusion

Our analysis shows that features BQ, CR, DU, AB and DL are the top 5 health characteristics most predictive of (and associated with) aged-related conditions.


