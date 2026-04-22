# Milestone 6: Deployment & Documentation
**Group 10: Data Science and AI Lab**

**Date:** 16th April 2026

---

## 1. Project Overview

This project builds a **production-ready guardrail system** that protects against prompt jailbreak attacks without changing the underlying Large Language Model.

The system sits between the user and the **Gemini API**, acting as a safety layer. It uses a combination of:

* Rule-based filtering for obvious threats
* A fine-tuned transformer model (*mDeBERTa-v3*) for semantic understanding
* A threshold-based decision system

Based on the risk level:

* **High-risk prompts** → blocked
* **Low-risk prompts** → allowed
* **Medium-risk prompts** → optionally cleaned using an external model and checked again

This layered approach ensures both **efficiency and robustness** during inference time.

---

## 2. Deployment

### Selected Approach

* **Platform:** Hugging Face Spaces
* **Framework:** Streamlit

The application is deployed as an interactive web interface where users can submit prompts and receive filtered responses in real time.

---

### Development Support

* **Kaggle** was used for experimentation, model training, and validation due to available compute resources
* Final deployment is independent and runs via Hugging Face Spaces

---

### Deliverables

* Deployment with working UI (Streamlit)
* Clear run instructions (see: `README.md`)
* Defined input/output behavior with examples
* Demo video / screenshots

---

### Links

* 🔗 Live App: [Click here](https://huggingface.co/spaces/DS-AI-Group10/gaurdrail_system_demo)
* 🔗 Notebooks: `[Click Here](https://github.com/23f1002451/Group-10-DS-and-AI-Lab-Project/tree/8fc399458171e525c11883119b4827080c51a57e/notebooks)

---

## 3. Documentation

Detailed technical and user-level documentation is available here:

* 📘 `/docs/` 

---

## 4. Final Report

The complete project report (methodology, experiments, evaluation, and conclusions) is available at:

* 📄 [Final Report](https://github.com/23f1002451/Group-10-DS-and-AI-Lab-Project/blob/c8cc014b72a8d4195cca14765f9f510e213cc82a/docs/Group10-DSAI-Final%20Report.docx)

---

## 5. Directory Structure

```id="bplvwy"
Group-10-DS-and-AI-Lab-Project/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
│
├── api/
│   └── TODO
│
├── app/
│   └── app.py
│
├── data/
│   ├── large/
│   │   ├── train.json
│   │   ├── test.json
│   │   └── validation.json
│   └── small/
│       ├── train.json
│       ├── test.json
│       └── validation.json
│
├── docs/
│   ├── api_doc.md
│   ├── CHANGELOG.md
│   ├── contributions.md
│   ├── Group10-DSAI-Final Report.docx
│   ├── licenses.md
│   ├── overview.md
│   ├── problem_statement.md
│   ├── technical_doc.md
│   ├── user_guide.md
│   │
│   ├── milestone_1/
│   │   ├── Milestone-1.md
│   │   ├── Analysis of existing solutions.pdf
│   │   ├── Review of Current Solutions.pdf
│   │   └── Group10_Milestone1.pptx
│   │
│   ├── milestone_2/
│   │   ├── Milestone-2.md
│   │   ├── Group10_Milestone2.pptx
│   │   └── Milestone2_EDA_Insights.pdf
│   │
│   ├── milestone_3/
│   │   ├── Milestone-3.md
│   │   └── Group10_Milestone3.pptx
│   │
│   ├── milestone_4/
│   │   ├── Milestone-4.md
│   │   └── DSAI_Group10_Milestone4.pptx
│   │
│   ├── milestone_5/
│   │   ├── Milestone-5.md
│   │   ├── cm.png
│   │   ├── label_dist.png
│   │   └── threshold vs asr,frr.png
│   │
│   └── milestone_6/
│       └── Milestone-6.md
│
├── models/
│   └── model.md
│
├── notebooks/
│   ├── Final Guardrail.ipynb
│   ├── Final_HPT.ipynb
│   └── outputs/
│       ├── final_guardrail_outputs/
│       │   ├── confusion_matrices.png
│       │   ├── error_analysis.png
│       │   ├── max_length_justification.png
│       │   ├── misclassified.csv
│       │   └── threshold_sweep.png
│       │
│       └── hp_notebook/
│           ├── confusion_matrices.png
│           ├── error_analysis.png
│           ├── final_summary.json
│           ├── hpo_results.csv
│           ├── hpo_sensitivity.png
│           ├── layerwise_analysis.png
│           ├── max_length_justification.png
│           ├── misclassified.csv
│           └── threshold_sweep.png
│
└── src/
    ├── evaluate.py
    ├── guardrail_classifier.py
    ├── guardrail_pipeline.py
    ├── regex_filter.py
    ├── run_e2e_subset.py
    ├── train.py
    └── __init__.py
```

---

### Notes

* This document provides a **high-level view only**
* Detailed explanations are maintained separately to avoid repetition


