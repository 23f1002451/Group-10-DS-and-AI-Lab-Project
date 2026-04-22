# Licences and Third-Party Attribution

## 1. Project Licence

This project is distributed under the **MIT Licence**. The full text of the licence is available in the `LICENSE` file at the root of the repository.

```
MIT License

Copyright (c) 2026 23f1002451

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 2. Pre-trained Model Licence

| Model | Licence | Source |
| :--- | :--- | :--- |
| microsoft/mdeberta-v3-base | MIT Licence | [Hugging Face](https://huggingface.co/microsoft/mdeberta-v3-base) |

The mDeBERTa-v3-base model is published by Microsoft Research under the MIT Licence, which permits commercial and non-commercial use, modification, and redistribution.

---

## 3. Dataset Licences

The training and evaluation datasets are sourced from publicly available repositories with the following licences:

| Dataset | Licence | Source | Usage |
| :--- | :--- | :--- | :--- |
| SQuAD v2 (Rajpurkar et al.) | CC-BY-SA 4.0 | [Hugging Face](https://huggingface.co/datasets/rajpurkar/squad_v2) | Benign prompt source |
| Alpaca Cleaned (Yahma) | Apache 2.0 | [Hugging Face](https://huggingface.co/datasets/yahma/alpaca-cleaned) | Benign prompt source |
| TrustAIRLab In The Wild Jailbreak Prompts | Research use | [Hugging Face](https://huggingface.co/datasets/TrustAIRLab/in-the-wild-jailbreak-prompts) | Benign and jailbreak prompts |
| JailbreakBench JBB Behaviours | MIT Licence | [JailbreakBench](https://jailbreakbench.github.io) | Benign and harmful prompts |
| LMSYS Toxic Chat | CC-BY-NC 4.0 | [Hugging Face](https://huggingface.co/datasets/lmsys/toxic-chat) | Jailbreak and harmful prompts |
| Rubend18 ChatGPT Jailbreak Prompts | Public domain | [Hugging Face](https://huggingface.co/datasets/rubend18/ChatGPT-Jailbreak-Prompts) | Jailbreak prompt source |

All datasets are used in accordance with their respective licences. The CC-BY-NC 4.0 licence on LMSYS Toxic Chat restricts commercial use of that specific dataset partition; all other datasets permit both commercial and non-commercial use.

---

## 4. Software Dependencies

The following major software libraries are used in this project:

| Library | Licence | Purpose |
| :--- | :--- | :--- |
| PyTorch | BSD-3-Clause | Deep learning framework |
| Transformers (Hugging Face) | Apache 2.0 | Model loading and tokenisation |
| scikit-learn | BSD-3-Clause | Evaluation metrics computation |
| NumPy | BSD-3-Clause | Numerical computing |
| SentencePiece | Apache 2.0 | Subword tokenisation for mDeBERTa |
| Gradio | Apache 2.0 | Web interface for demonstration |
| Streamlit | Apache 2.0 | Alternative web interface |
| google-generativeai | Apache 2.0 | Gemini API client for transformation layer |

All dependencies are installed via `pip install -r requirements.txt` and are listed in the `requirements.txt` file at the root of the repository.

---

## 5. API Usage

| Service | Provider | Terms |
| :--- | :--- | :--- |
| Gemini 2.5 Flash API | Google AI | [Google AI Terms of Service](https://ai.google.dev/terms) |
| Hugging Face Spaces | Hugging Face | [Hugging Face Terms of Service](https://huggingface.co/terms-of-service) |

The Gemini API is used for the LLM transformation stage (Layer 3). Usage is subject to Google's API terms and rate limits. The free tier provides sufficient quota for research-scale evaluation.

---

## 6. Academic References

This project builds upon the following published work:

1. He, P., Liu, X., Gao, J., and Chen, W. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. In Proceedings of ICLR.
2. Bergstra, J. and Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. Journal of Machine Learning Research, 13(Feb):281-305.
3. OWASP Foundation. OWASP Top 10 for Large Language Model Applications. https://owasp.org/www-project-top-10-for-large-language-model-applications
4. NIST. AI Risk Management Framework. https://www.nist.gov/itl/ai-risk-management-framework

