
**ASPECT SENTIMENT TRIPLET EXTRACTION USING FINE-TUNED BIDIRECTIONAL ENCODERS (BERT)**
Student 1, Student 2, Student 3, Student 4, Student 5

**MOTIVATION**

* Modern businesses are flooded with complex, unstructured customer reviews, making it impossible to manually track specific product flaws or strengths.
* Standard sentence-level sentiment analysis fails on nuanced feedback (e.g., scoring "The pizza was great, but the service was terrible" as "Neutral" loses critical business value).
* Extracting exact target-level feedback is essential for automated quality assurance and real-time product iteration.
* Major advancements in localized token classification have been made in recent years thanks to bidirectional Transformer architectures, which we aim to apply to Aspect-Based Sentiment Analysis (ABSA).




**PROBLEM STATEMENT**


**Core Problem:**

* Existing classification models use simple building blocks or standard sequence-level classifiers that ignore multi-aspect conflicts within a single sentence.


* Current generative LLMs (Large Language Models) introduce text-generation but sometimes lead hallucinations, lack precise span-boundary detection. In addition, it can  have inference latencies that are too high for real-time streaming pipelines and using it in production system.



**Research Goal:**

* By the end of the project (Week 10), implement and fine-tune a BERT-based sequence-labeling model to jointly extract aspect terms, aspect categories, and sentiment polarities (triplets) from unstructured reviews, and evaluate its performance on held-out test sets using strict Macro $F_1$-score as the primary metric, with Precision and Recall as secondary metrics.




**LITERATURE BACKGROUND**

**1**
**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., 2019)**

* **Overview:** Introduces the foundational BERT architecture, utilizing Masked Language Modeling (MLM) to process text bidirectionally.


* **The Strength:** Exceptional at capturing deep, bidirectional contextual embeddings for localized tokens.


* **Speed:** Extremely lightweight inference compared to modern LLMs, allowing for rapid batch processing of high-volume customer review pipelines.



**2**
**Do BERT-Like Bidirectional Models Still Perform Better on Text Classification in the Era of LLMs? (Zhang et al., 2025)**

* **Overview:** Systematically compares fine-tuned bidirectional encoders against zero-shot inference from massive LLMs across classification tasks.


* **The Strength:** Excellent at proving that BERT-like models mathematically excel at pattern-driven tasks over generative models.


* **The Gap:** While superior for general text classification, standard sequence classification heads aren't inherently optimized for the sharp span-boundary detection required for Aspect Sentiment Triplet Extraction (ASTE).




**OUR APPROACH**

* **The Goal:** Triplet extraction is a multi-scale sequence problem. Predicting sentiment is broad, but detecting exact aspect boundaries is highly localized.


* **The Method:** We adapt high-resolution bidirectional principles from BERT and combine them with custom BIO (Begin, Inside, Outside) sequence-tagging classification heads to create a hybrid model that detects aspect boundaries while calculating polarity.


* **The Reliability:** Beyond standard accuracy, we integrate strict span-level $F_1$ evaluation. This provides a "confidence buffer" ensuring the model actually understands the target, rather than randomly guessing positive or negative sentiment.




**DATASETS**

**M-ABSA (Multilingual Aspect-Based Sentiment Analysis, 2025)**


**Overview:** Large-scale, state-of-the-art benchmark dataset for modeling aspect-sentiment triplet extraction.
**Data Sources & Modalities:** High-quality review modalities spanning 7 domains, explicitly annotated with `[aspect term, aspect category, sentiment polarity]` targets.
**Structure:** Filtered to the English subset. Formatted for direct string extraction to eliminate messy manual annotation requirements.

**ASTE-Data-V2 (SemEval-based Triplet Data)**


**Overview:** Sequence-based dataset optimized for tracking complex, conflicting sentiments in single sentences.
**Data Sources & Modalities:** Curated restaurant and laptop reviews adapted from the original SemEval-2014 Task 4 benchmarks.
**Structure:** Tokenized spatial resolution mapped to precise opinion and aspect spans.

We use the "benchmark" ASTE-Data-V2 to establish our structural baselines, while the highly-complex M-ABSA data allows us to predict exactly how our model handles modern, multi-domain extraction tasks.


**METHODOLOGY**

1. **Data Preprocessing:** Extract relevant aspect triplets and translate them into machine-readable BIO (Begin, Inside, Outside) sequence tags.


2. **Token Alignment:** Construct WordPiece sub-token alignment to ensure target labels remain mapped correctly when BERT splits complex words.


3. **Model Architecture:**

* BERT encoder branch: global sentence context.


* Custom Token Classification head branch: local, fine-scale span extraction.


* Fusion: combine both using Focal Loss to calculate predictions.




4. **Prediction Task:** Predict the exact aspect boundary and polarity array from an unstructured text string.


5. **Training Strategy:** Train/Validation/Test split based on dataset guidelines. Execute linear warmup scheduling and gradient accumulation on GPU infrastructure.


6. **Evaluation:** Primary: Macro $F_1$-score for strict triplet matches. Secondary: Precision and Recall boundary matrices.




**TIMELINE**

* **Phase 1 (Week 1 - Week 2):** Research & Data Preparation. GitHub initialization and BIO sequence encoding.


* **Phase 2 (Week 3 - Week 5):** Baseline Construction. PyTorch Dataloader and out-of-the-box BERT evaluation.


* **Phase 3 (Week 6 - Week 8):** Model Development & Training/Tuning. Custom loss function integration and hyperparameter sweeps.


* **Phase 4 (Week 9 - Week 10):** Evaluation, Analysis & Report + Presentation. Strict code freeze and deep error analytics.




**RISKS**


**1. Token Alignment & Data Quality**

* *Issue:* Misaligned labels when the tokenizer splits words into sub-tokens.


* *Impact:* Reduced model accuracy and immediate dimension mismatch errors during training.


* *Mitigation:* Rigorous testing of the WordPiece mapping script before feeding tensors to the model.



**2. Extreme Class Imbalance**

* *Issue:* Over 85% of tokens in a sentence will be "Outside" (not an aspect), causing the model to just guess "Outside" every time.


* *Impact:* High apparent accuracy but an $F_1$-score of 0 on actual triplet extraction.


* *Mitigation:* Experiment with Class-Weighted Cross-Entropy or Focal Loss structures to penalize majority-class predictions.






**ROLES**

* **Student 1 : Integration & Coordination:** Manage end-to-end pipeline for experiments. Codebase organization, version control, and final document synthesis.


* **Student 2: Literature & Context:** Design the academic justification, track state-of-the-art baselines, and manage citation formatting.


* **Student 3: Data Engineering:** Preprocess and align ASTE labels to sub-word tokens. Handle input synchronization and build the PyTorch `DataLoader`.


* **Student 4: Model Architecture:** Design and implement BERT fine-tuning logic. Configure custom loss functions and hyperparameter optimization sweeps.


* **Student 5: Uncertainty & Evaluation:** Design strict span-level metrics. Evaluate failure cases (boundary errors vs polarity errors) and generate visualizations.




**REFERENCES (Paper & Dataset Links)**

* **Paper 1 (BERT):** Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv. (Link: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805))
* **Paper 2 (ASTE Foundation):** Xu, H., Liu, B., Shu, L., & Yu, P. S. (2020). *Knowing What, How and Why: A Dataset for Aspect Sentiment Triplet Extraction*. ACL Anthology. (Link: [https://aclanthology.org/2020.emnlp-main.735/](https://aclanthology.org/2020.emnlp-main.735/))
* **Paper 3 (Modern State-of-the-Art Defense):** Zhang, J., Huang, Y., Liu, S., Gao, Y., & Hu, X. (2025). *Do BERT-Like Bidirectional Models Still Perform Better on Text Classification in the Era of LLMs?*. ACL Anthology, EMNLP 2025 Findings. (Link: [https://aclanthology.org/2025.findings-emnlp.1033/](https://aclanthology.org/2025.findings-emnlp.1033/))
* **Dataset 1 (M-ABSA):** *Multilingual Aspect-Based Sentiment Analysis*. GitHub Repository. (Link: [https://github.com/swaggy66/M-ABSA](https://github.com/swaggy66/M-ABSA))
* **Dataset 2 (ASTE-V2 Data Hub):** *SemEval-Triplet-Data*. GitHub Repository. (Link: [https://github.com/xuuuluuu/SemEval-Triplet-data](https://github.com/xuuuluuu/SemEval-Triplet-data))
* **Supplementary Framework (PyABSA):** *ABSADatasets*. GitHub Repository. (Link: [https://github.com/yangheng95/ABSADatasets](https://github.com/yangheng95/ABSADatasets))