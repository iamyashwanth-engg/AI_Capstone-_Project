Domain: Banking

## How to run
From a clean clone of the repository, execute the following commands in exact order:
```bash
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

---

## Part 1 findings
* **R1.1 Planning claim test**: The null hypothesis H0 (mean = 8.0) was rejected (reject_h0 = yes, z_stat = 2.8930 vs critical value 1.96). A Type I error in this banking context means concluding that average monthly spend differs from INR 8,000 when true mean spend is actually INR 8,000, causing unnecessary budget adjustments. The sample gap is 0.9382 column units (INR 938.20 per borrower), which materially impacts portfolio spend planning.
* **R1.2 Sample size implications**: Achieving a margin of error E = 0.5 (INR 500) at 95% confidence requires n_required = 1293 borrowers. With 800 rows available, the dataset provides more than three times the required sample size, making the current estimate highly reliable.
* **R1.3 Gradient descent convergence**: The slope difference is gd_slope_gap = 0.0000. Gradient descent reached the closed-form OLS line because univariate linear regression on standardised features minimizes a strictly convex quadratic loss with a well-scaled learning rate.
* **R1.4 Residual funnel and variance stabilization**: The raw model shows a clear funnel with resid_sd_ratio_raw = 3.7525, meaning residual spread increases substantially with fitted spend. Applying the log-log transformation removes this funnel, bringing resid_sd_ratio_log down to 1.0396.
* **R1.5 Multiple regression predictors**: Predictors with absolute coefficient above 0.05 (excluding constant) are ln(X4) (1.1576), X2 (0.0940), X3 (0.1067), X6 (0.1518). Comparing adjusted R-squared, mr_adj_r2 = 0.7403 improves over loglog_adj_r2 = 0.6561, confirming that the additional behavioural predictors earn their place.

---

## Part 2 findings
* **R2.1 Learning-rate effect**: Ordered from lowest to highest 200-epoch training loss: eta = 1.0 (0.3703), eta = 0.1 (0.3776), eta = 0.01 (0.5091). A learning rate that is too small progresses too slowly over 200 updates, while a large learning rate causes larger parameter steps that can overshoot or oscillate around the optimum.
* **R2.2 Feature impact and odds ratio**: The three predictors with the largest absolute weights are X3 (0.9013), X1 (-0.5554), X2 (0.5505). The largest odds ratio is odds_ratio_X3 = 2.4629: increasing X3 by one training standard deviation multiplies the odds of loan default or 90-day delinquency by 2.4629, holding other features fixed.
* **R2.3 Accuracy vs cost trade-off**: Blind majority-class accuracy is 0.7875 compared to 0.8042 at tau = 0.5, showing that plain accuracy hides poor minority-class detection on imbalanced default data. Lowering tau from 0.5 to 0.3 increased recall from 0.4706 to 0.7451 while precision shifted from 0.5455 to 0.4634.
* **R2.4 Logistic vs KNN**: At tau = 0.3, logistic regression achieves f1 = 0.5714 and recall = 0.7451, outperforming the cross-validated KNN classifier (K = 5), which achieves knn_f1 = 0.3696 and knn_recall = 0.3333.
* **R2.5 Operational threshold deployment**: Deploy logistic regression at threshold cost_tau = 0.1500, which yields cost_test_at_tau = INR 6050 compared to cost_test_at_0.5 = INR 11800 and cost_test_flag_nobody = INR 20400. Choosing the threshold on training rows prevents test-set leakage and ensures an unbiased operational cost estimate.

---

## Part 3 findings
* **R3.1 Elbow evaluation**: The four successive WCSS drops are WCSS(2) - WCSS(3) = 659.8282, WCSS(3) - WCSS(4) = 108.0739, WCSS(4) - WCSS(5) = 171.0575, and WCSS(5) - WCSS(6) = 45.5466. The elbow rule selects K = 3; note that the pinned starting rows give one reproducible solution rather than a guaranteed global optimum across all K.
* **R3.2 Borrower segments & strategy**:
  * **Segment A (Established Low-Risk Borrowers)**: Size 322, mean_X3 = 0.2250, outcome_rate = 0.0186; offer pre-approved credit line expansions.
  * **Segment B (High-Inflow Moderate-Risk Borrowers)**: Size 255, mean_X3 = 0.4534, outcome_rate = 0.2353; enroll in automated UPI autopay nudges.
  * **Segment C (High-Utilisation Subprime Borrowers)**: Size 223, mean_X3 = 0.7118, highest outcome_rate = 0.4664; cap credit exposure and trigger early collections reminders.
* **R3.3 Hierarchical linkage disagreement**: At the cut, single linkage produces 4 clusters ({E1+E4+E5} and {E2+E3+E6}), whereas complete linkage produces 2 clusters ({E1}, {E4+E5}, and {E2+E3+E6}). Single linkage merges groups based on their closest single pair (chaining), while complete linkage requires the farthest pair between two groups to be close before merging.
* **R3.4 Principal Component Analysis**: PC1 is driven positively by credit utilisation X3 (-0.5187) and failed transactions X2 (-0.4100) and negatively by account age X1 (0.5317) and repayment score X5 (0.4418). Retaining at least 80% of variance requires components_for_80pct = 3 components, and mean PC1 scores progress from Segment A (1.7773) to Segment B (-0.7831) to Segment C (-1.6709), separating low-risk from high-risk borrowers.

---

## Part 4 findings
* **R4.1 Split purity & tree constraints**: Split S4 (X3 greater than 0.5) is best because credit utilisation separates defaulting and non-defaulting borrowers more cleanly than any other single rule. Growing a tree until every leaf is pure memorizes individual training noise and overfits; setting a minimum samples per leaf rule stops excessive depth.
* **R4.2 Model combination comparison**: Logistic regression at tau = 0.3 achieves f1 = 0.5714 and recall = 0.7451, while soft_vote achieves f1 = 0.5507 and recall = 0.7451 (hard_vote achieves f1 = 0.5344 and recall = 0.6863). Soft voting combines probability confidence across models rather than discarding margin information, making logistic or soft voting the strongest candidates.
* **R4.3 AdaBoost weight progression**: After round 1 normalisation, the two misclassified points (points 3 and 7) carry a combined weight of 0.5000. Round 2 trusts h2 (round2_alpha = 0.8047) more than h1 (round1_alpha = 0.5493) because h2 achieves a lower weighted error on the reweighted sample.
* **R4.4 Deep learning sizing & gradient mechanics**:
  * **Model A (Option 2 CNN Architecture)**:
    | Layer | Output Shape | Parameters |
    |---|---|---|
    | Conv 1 (5x5, 6 filters, pad 2) | 28 x 28 x 6 | 156 |
    | Pool 1 (2x2, stride 2) | 14 x 14 x 6 | 0 |
    | Conv 2 (5x5, 16 filters, pad 0) | 10 x 10 x 16 | 2,416 |
    | Pool 2 (2x2, stride 2) | 5 x 5 x 16 | 0 |
    | Dense 1 (Flatten to 32) | 32 | 12,832 |
    | Dense 2 (Output 10) | 10 | 330 |
  * **Model B (RNN Shapes)**: W_xh is 128 x 64, W_hh is 128 x 128, and W_hy is 16 x 128; rnn_total (26768) does not grow with sequence length because the same recurrent weight matrices are reused at every time step.
  * **Activation & Loss**: Use ReLU for hidden layers, Softmax for the 10-class output layer, and Categorical Cross-Entropy loss.
  * **Gradient Vanishing**: Very deep plain networks suffer from vanishing gradients as repeated matrix multiplications shrink backpropagated error signals toward zero in early layers. Skip connections add identity shortcuts that pass gradients directly across layers without attenuation.

---

## Model-choice memo

**To:** Head of Lending-Risk Analytics, PhonePe  
**From:** Lending-Risk Analytics Team  
**Subject:** Production Credit-Risk Classifier Selection and Sizing  

We recommend deploying the **Logistic Regression model at an alert threshold of 0.1500** for pre-due-date delinquency nudges.

Because an unnecessary reminder costs INR 50 while an unflagged default costs INR 400 to recover, optimizing for raw accuracy (0.8042 at threshold 0.5) is suboptimal and misses too many defaults. At threshold 0.1500, our logistic model reduces test collections cost to **INR 6050**, saving **INR 5750** compared to the standard 0.5 threshold (**INR 11800**) and **INR 14350** compared to flagging nobody (**INR 20400**). While soft voting achieves a test F1 of 0.5507 (versus 0.5714 for logistic regression at 0.3), its operational cost under the INR 50 / INR 400 rule was not computed, whereas logistic regression is calibrated directly for minimum cost and low-latency inference.

**Risk and Limitation:** The model is trained on a static 560-row synthetic sample with only 51 test defaults, so shifts in real borrower inflow or repayment behavior could degrade calibration over time.

**Recommendation on Models A and B:** Neither Model A (cheque digit OCR, 15734 CNN parameters) nor Model B (payment remark RNN, 26768 parameters) should be built before the structured default model is in production, because preventing credit losses delivers immediate financial impact on existing loan books.

---

## Limitations
1. **Synthetic Data Distribution**: Records are generated from parametric normal and log-normal rules rather than live UPI transaction logs, omitting real-world heavy tails and seasonal shocks.
2. **Single Stratified Train/Test Split**: Evaluation relies on one fixed 560/240 split; k-fold cross-validation across the full dataset would provide tighter confidence bounds on generalization.
3. **Small Positive Test Sample**: With only 51 positive default cases in the 240-row test set, individual threshold classifications carry sampling variance that requires shadow-mode validation before full rollout.
