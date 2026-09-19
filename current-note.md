Your proposed class-weighting method is:

> Full fine-tuning of `bert-base-uncased` using independently calculated inverse-frequency class weights for the aspect-term BIO, aspect-category, and sentiment prediction losses.

For each prediction task, the weight for class \(c\) is:

\[
w_c = \frac{N}{K\,n_c}
\]

where:

- \(N\) = total valid training targets for that task
- \(K\) = number of classes
- \(n_c\) = training occurrences of class \(c\)

Rare classes receive larger weights, while frequent classes receive smaller weights. Weights are calculated only from the current fold’s training data, preventing development/test leakage.

The combined training objective is:

\[
L = L_{\text{BIO}}^{weighted}
  + L_{\text{category}}^{weighted}
  + L_{\text{sentiment}}^{weighted}
\]

All three loss multipliers are currently `1.0`.

The controlled comparison is:

- **Baseline:** ordinary cross-entropy for all three prediction heads.
- **Proposed method:** inverse-frequency weighted cross-entropy for all three heads.
- **Everything else remains identical:** BERT model, data, seeds, optimizer, learning rate, epochs, and checkpoint-selection procedure.

A suitable paper description is:

> We propose a class-weighted BERT baseline for explicit-aspect TASD. The model fully fine-tunes BERT-base-uncased with separate BIO tagging, aspect-category, and sentiment-classification heads. To address label imbalance, inverse-frequency weights derived exclusively from each fold’s source-training data are applied independently to the cross-entropy loss of all three tasks. We compare this approach with an otherwise identical unweighted baseline across matched random seeds and leave-one-domain-out evaluation.

Important limitation: weighting cannot learn a category absent from source training, and it cannot recover `NULL` aspects because the current span-based architecture cannot represent them. Also, class weighting itself is established; the stronger research contribution is determining when it helps under domain shift and how it affects term, category, sentiment, and rare-label errors.

The best next alternative is **category-only class weighting**, followed by **class-balanced loss based on effective sample counts**.

## Recommended order

### 1. Head-specific weighting ablation — highest priority

Your current method weights all three losses simultaneously:

\[
L=L_{\text{BIO}}^{weighted}+L_{\text{category}}^{weighted}+L_{\text{sentiment}}^{weighted}
\]

This makes it difficult to determine which weighting component causes the precision–recall trade-off. Add:

- BIO-only weighting
- Category-only weighting
- Sentiment-only weighting
- All-head weighting
- Standard unweighted baseline

I recommend testing **category-only weighting first**:

\[
L=L_{\text{BIO}}+
L_{\text{category}}^{weighted}+
L_{\text{sentiment}}
\]

Category imbalance is central to the research question, while aggressive BIO weighting may encourage excessive aspect extraction and reduce precision. This is currently a hypothesis that the ablation would test.

### 2. Effective-number class-balanced loss — best alternative loss

Inverse-frequency weighting can give extremely rare classes excessively large weights. Class-balanced loss instead uses the “effective number” of samples:

\[
w_c=\frac{1-\beta}{1-\beta^{n_c}}
\]

This produces smoother weights and may retain rare-category recall without causing such a large precision reduction. It is the most direct alternative because the model, data, and evaluation can remain unchanged. [Cui et al., 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html).

Recommended configuration:

> BERT-base-uncased with effective-number class-balanced loss applied to the category head, while BIO and sentiment retain standard cross-entropy.

### 3. Focal loss

Focal loss gives more attention to difficult or misclassified examples rather than weighting classes only by frequency:

\[
L_{\text{focal}}=-(1-p_t)^\gamma\log(p_t)
\]

It may help when frequent easy examples dominate training. A conventional starting value is \(\gamma=2\), but it must be fixed or selected using source development data only. [Lin et al., 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html).

### 4. Logit adjustment

Logit adjustment modifies classification scores using source-training class priors. It can be applied during training or after training and is designed for long-tailed classification. It may provide better calibration than aggressive inverse-frequency weighting. [Menon et al., 2021](https://research.google/pubs/long-tail-learning-via-logit-adjustment/).

### 5. Generative mT5 baseline

For a stronger publication comparison, fine-tune `mT5-base` to generate complete triplets. Unlike the current BIO architecture, a generative model can represent:

- `NULL` aspects
- Multiple category-sentiment pairs for one term
- Full TASD rather than the restricted explicit-aspect task

The original M-ABSA study uses fine-tuned `mT5-base` as its generative baseline, making this a particularly relevant comparison. [M-ABSA paper](https://aclanthology.org/2025.emnlp-main.128.pdf).

## Practical experiment recommendation

Complete the protected 70-run study first. Then add:

1. **Category-only inverse-frequency weighting:** 35 new runs  
2. **Category-only effective-number weighting:** 35 new runs  
3. **mT5-base:** only if time and GPU resources allow

The strongest next research question would be:

> Can category-specific class-balanced loss improve rare-category transfer without producing the precision loss observed when BIO, category, and sentiment losses are all weighted?

This is more informative—and potentially more publishable—than simply trying several unrelated loss functions.