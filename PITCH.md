# CreditWise AI - 90-second pitch

**Problem (10s).** A card issuer loses margin to defaults, but blanket limit cuts push good customers away. The risk team can only review **500 accounts per cycle** - so *which* 500 is the whole game.

**Solution (15s).** CreditWise AI scores every account's probability of missing next month's payment, ranks them, and hands the team a **Top-500 review queue**. For the top 20 it adds plain-language risk signals and a recommended action - *call/payment outreach, limit review, or payment-plan review*. A human decides every time.

**Why 500 matters (15s).** About 22% of customers default. Picking 500 at random gives ~22% precision. Of our Top-500 out of 30,000 accounts, **81.8% actually defaulted - 3.7× the base rate**, catching 409 defaults. That's our operational metric: how much better than random the team's scarce hours are spent.

**Rigor (20s).** The ID and target never enter the model; preprocessing is fitted on training data only; model selection used a validation split and the test set stayed untouched. We decoded the cryptic repayment codes: **-2/-1/0 are not delays; 1+ is months late**. Undocumented education/marriage codes are mapped to "other", not dropped. A suspiciously high AUC would signal leakage, so we report honest numbers: **ROC-AUC 0.78, PR-AUC 0.56** on a held-out test set.

**Explainability (10s).** Each account gets deterministic reason codes - "3 consecutive months delayed, payments down 70%, balance at 96% of limit" - labelled as risk signals, not causes.

**Responsible AI (15s).** We **exclude sex, age, marriage and education**. An ablation shows ROC-AUC moves by just **0.002** without them, because repayment behaviour dominates. We still say plainly: dropping columns doesn't guarantee fairness, so we ship a group-level audit, human review, and a "not for automated decisions" policy.

**Limits & next steps (5s).** Random split of 2005 Taiwan data - next: temporal validation, outcome tracking of the actions, and calibrated tiers with the client's risk team.

### Likely judge questions
- **What does PAY_0 = -1 mean?** Paid in full that month. -2 = no consumption, 0 = revolving credit with minimum paid. Only 1+ means delayed.
- **Would you use gender in a credit model?** No. Ablation shows it adds ~nothing and it raises legal/fairness risk.
- **Why is 0.78 AUC still valuable?** With a 22% base rate, concentrating defaults in the top 500 makes each review hour materially more productive. Lift is the business metric.
- **Why is recall@500 only 6%?** There are 6,636 defaults and only 500 review slots; the queue is precise, not exhaustive. More capacity or an outreach that scales would raise it.
- **Why not 0.85+?** On this dataset that would indicate leakage or a bad split.
