# Dataset Description: MIMIC-IV Clinical Database Demo

## Original Dataset

**MIMIC-IV Clinical Database Demo 2.2** is a publicly available, de-identified subset of the MIMIC-IV database containing real clinical data from ICU patients at Beth Israel Deaconess Medical Center. The dataset includes:

- **102 unique patients** with demographic information (age, gender, admission details)
- **277 hospital admissions** with admission/discharge times, insurance, and clinical characteristics
- **107,729 laboratory events** covering 1,604 different lab tests (creatinine, BUN, electrolytes, blood counts, etc.)
- **668,854 chart events** containing vital signs, medications, and clinical observations
- **142 ICU stays** across different care units (Medical ICU, Surgical ICU, Trauma ICU, etc.)

The data spans multiple years and includes comprehensive clinical information for each patient's hospital stay, making it ideal for medical AI research and clinical prediction tasks.

## Used Dataset

For this AKI prediction project, we processed the original MIMIC-IV data to create a focused dataset:

- **2870 hospital admissions** (extended from 287 original admissions using data augmentation)
- **14.29% AKI rate** (realistic clinical prevalence based on patient characteristics)
- **18 clinical features** extracted from the original data:
  - **8 vital signs**: Heart rate, blood pressure (systolic/diastolic/mean), respiratory rate, temperature, oxygen saturation, glucose
  - **10 laboratory values**: Creatinine, BUN, potassium, sodium, chloride, bicarbonate, hemoglobin, platelet count, white blood cell count, lactate
- **48-hour temporal sequences** for each admission (hourly sampling)
- **KDIGO-based AKI staging** (Stages 1-3) applied to admission data

**Key Point**: The dataset uses **hospital admissions** as the primary unit (2870 admissions), not individual patients (100 patients). This is clinically appropriate because:
1. Each admission represents a separate clinical episode
2. Patients can have multiple admissions with different AKI risks
3. Each admission becomes an independent training sample
4. This approach captures the temporal nature of hospital care
5. **Data augmentation** was used to extend from 287 to 2870 samples while maintaining clinical realism

The processed dataset maintains the clinical realism of the original MIMIC-IV data while providing a structured format suitable for deep learning model training and evaluation. This enables direct comparison between CNN+BiLSTM+Attention and Transformer architectures on real clinical data, demonstrating the superior performance of the hybrid approach for AKI prediction in ICU settings.
