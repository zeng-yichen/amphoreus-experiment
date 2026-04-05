# Amphoreus Learning Architecture: A Literature-Grounded Appraisal

**April 2026**

---

## Executive Summary

Amphoreus is a closed-loop content generation system that combines LLM-based generation with online learning from real-world engagement signals. Unlike conventional prompt-template systems that produce static output, Amphoreus treats every published post as a training observation and continuously adapts its generation strategy, quality evaluation, and audience targeting per client. This appraisal examines every learning component against the latest research, identifies novel design choices, and flags areas where the architecture makes deliberate tradeoffs.

The system's core thesis is **the bitter lesson applied to content strategy**: instead of encoding human editorial heuristics as rules, expand the observation space and let the system discover what works from data. This philosophy, articulated by Sutton (2019), pervades the architecture—from emergent quality dimensions in Cyrene to open-ended post analysis in RuanMei to continuous reward fields in LOLA.

---

## 1. RuanMei — The Observation Engine

### What it does
RuanMei is the system's episodic memory. It ingests every published post (not just system-generated ones), runs open-ended LLM analysis on each, computes a z-scored composite engagement reward, and stores the full observation history. This is the foundation that all other learning components read from.

### Literature context

**Open-ended LLM analysis vs. predefined taxonomies.** RuanMei deliberately avoids predefined post categories (hook type, tone, structure). Instead, it prompts Claude to describe "whatever stands out" about a post's construction. This aligns with the finding in Zheng et al. (2023, arXiv:2306.05685) that LLM-as-a-Judge evaluations, when unconstrained, often capture quality dimensions that predefined rubrics miss. The MT-Bench work showed that open-ended evaluation achieves higher agreement with human preference than categorical scoring systems.

**Z-scored composite reward.** The reward is a composite of depth (weighted comment/repost/reaction rates), reach (log impressions), and engagement rate, each z-scored against the client's own distribution. This per-client normalization is standard practice in contextual bandits (Li et al., 2010, "A Contextual-Bandit Approach to Personalized News Article Recommendation") and avoids the cross-client comparability problem that plagues systems using absolute engagement metrics. A client with 500 average impressions and one with 50,000 are evaluated on the same scale.

**Learned depth weights.** Rather than hard-coding that comments are worth 3× reactions, RuanMei learns component weights via Spearman rank correlation between each metric (comments/imp, reposts/imp, reactions/imp) and the composite reward. For innovocommerce, the system discovered that reactions dominate (ρ=0.96) while comments are uncorrelated (ρ=−0.01), yielding a weight vector of [0.0, 0.81, 2.19] instead of the default [3.0, 2.0, 1.0]. This is a notable design choice—most content analytics platforms use fixed weights.

**Lagged reward weight learning.** The `_get_reward_weights()` method learns how much each reward component (depth, reach, engagement rate, ICP) predicts *next post's* engagement. This is a simple form of temporal credit assignment: components that predict sustained success get higher weight. The lagged correlation approach has precedent in time-series forecasting (Granger causality tests), though applying it to content strategy reward shaping appears novel.

### Novel aspects

1. **Edit similarity tracking.** Every observation records the SequenceMatcher ratio between the draft (pre-push) and the live LinkedIn copy. This creates a natural experiment: when clients edit a post heavily before publishing, the system can correlate edit magnitude with engagement outcomes. No other content system I'm aware of tracks this signal systematically.

2. **Content trajectory as LLM context.** Rather than computing diversity metrics or topic entropy, RuanMei passes the raw chronological sequence of recent posts directly to the LLM for insight generation. The model identifies series momentum, topic saturation, and rotation gaps without predefined category taxonomies. This is the bitter lesson in action—let the model's pattern recognition handle what humans would need a taxonomy for.

### Concerns

- **Analysis cost.** Every ingested post triggers a full Claude analysis call. At ~$0.015/analysis, a client with 50 historical posts costs ~$0.75 on first ingest. This is a one-time cost that amortizes well, but the architecture should batch these calls.
- **No embedding of the analysis text in RuanMei itself.** The analysis is stored as free text. Cyrene and LOLA embed it separately, creating redundant embedding computation. A shared embedding cache would reduce cost.

---

## 2. Cyrene — SELF-REFINE Quality Loop

### What it does
Cyrene implements a generate→critique→revise loop based on the SELF-REFINE architecture (Madaan et al., 2023, arXiv:2303.17651). A critic scores each draft along quality dimensions, and a reviser addresses weak dimensions. The loop runs up to 3 iterations, stopping when a learned quality threshold is met or improvements become marginal.

### Literature context

**SELF-REFINE.** The original Madaan et al. paper demonstrated that LLMs can improve their own output through iterative self-feedback without additional training. Amphoreus' implementation is a faithful adaptation: separate critic and reviser prompts, structured JSON scoring, early stopping on threshold satisfaction. The key extension is that Amphoreus grounds the critic in *real engagement data*, addressing the main limitation noted by Huang et al. (2023, arXiv:2310.01798) in "Large Language Models Cannot Self-Correct Reasoning Yet"—that self-correction only works when the feedback signal is informative. In Amphoreus, the feedback signal is calibrated against actual post performance.

**Three-tier dimension system.** Cyrene operates in three modes based on data availability:
1. **Fixed dimensions** (cold start, < 10 observations): 7 predefined LinkedIn-native dimensions (Hook Scroll-Stop, Save-Worthiness, Comment Invitation, etc.)
2. **Emergent dimensions** (≥ 15 observations): LLM-discovered dimensions from top-vs-bottom performer comparison for each specific client
3. **Freeform critique** (≥ 5 observations with analysis): No dimensions at all—the critic writes open-ended analysis

This cascade from structured to unstructured evaluation mirrors the progression in the LLM-as-a-Judge literature. LLaVA-Critic (Sun et al., 2024, arXiv:2410.02712) showed that unconstrained evaluation can outperform rubric-based evaluation when the model has sufficient reference examples. Cyrene's freeform mode achieves this by injecting quality analyses from the client's own top and bottom performers into the critic prompt.

**Emergent dimension discovery.** This is perhaps Cyrene's most distinctive feature. Instead of using fixed quality dimensions, the system asks Claude to compare a client's top-quartile and bottom-quartile posts and identify what *actually explains the performance gap*. The discovered dimensions are cached for 14 days (TTL scales with posting frequency) and tracked for drift. This is a form of automated feature engineering—the system discovers which quality attributes matter for each specific client rather than assuming universal dimensions.

The drift detection mechanism (logging when >50% of dimensions change between rediscoveries) is a practical response to concept drift in content effectiveness. What works in Q1 may not work in Q3 as the audience evolves.

**Adaptive weights.** Dimension weights are Spearman correlations between each dimension score and actual engagement. This means a dimension like "Hook Scroll-Stop" might be weighted at 30% for one client and 5% for another based on which dimensions actually predict their engagement. The correlation-based weight learning is standard in feature importance analysis but novel when applied to LLM critic dimensions.

**Linear reward projection.** For freeform mode, Cyrene embeds the quality analysis using sentence-transformers and trains a ridge regression from (embedding, engagement_reward) pairs. This creates a continuous quality predictor that can evaluate drafts without waiting for post-publication engagement. The 384-dim embedding space naturally handles the high dimensionality of unconstrained quality analysis.

The blend weight between critic confidence and projected reward is itself learned from historical correlation—if the projection reliably predicts engagement (high Spearman ρ), it gets more weight. This meta-learning of the blending parameter is an elegant touch that avoids a hard-coded hyperparameter.

### Novel aspects

1. **Engagement-grounded pass threshold.** The pass threshold (default 3.5/5.0) is replaced by the median composite score of posts that actually performed well (above 40th percentile engagement). This directly connects the "quality gate" to what the audience actually rewards, rather than what an editorial rubric prescribes.

2. **Adaptive min_improvement.** The diminishing-returns threshold (when to stop iterating) scales with the standard deviation of historical composite scores. High-variance clients get a higher threshold (keep iterating—there's room to improve), while stable clients get a lower one (small improvements are meaningful).

3. **Phase transition from categorical to continuous.** The progression from fixed dimensions → emergent dimensions → freeform is a deliberate phase transition as data accumulates. Most systems pick one evaluation mode. Cyrene explicitly acknowledges that the right evaluation approach depends on data density.

### Concerns

- **Self-correction limitations.** Huang et al. (2023) showed that LLMs struggle with self-correction on reasoning tasks. Cyrene mitigates this by using engagement data as the external grounding signal, but the critic and reviser are the same model. Cross-model critique (as done in constitutional verification) would strengthen this.
- **Emergent dimension stability.** With a 14-day TTL and LLM-based discovery, the dimension set can shift between evaluation rounds. Two posts evaluated 15 days apart may be scored on entirely different rubrics. The drift detection logs this but doesn't prevent it.

---

## 3. Constitutional Verifier — Multi-Model Ensemble Gate

### What it does
A panel of 2-3 models (Claude, Gemini Flash, GPT-4o-mini) independently evaluate each post against quality principles. Majority vote determines pass/fail. Principles themselves are discovered from cross-client engagement data.

### Literature context

**Constitutional AI.** The name references Bai et al. (2022, arXiv:2212.08073), which introduced the idea of evaluating model outputs against a set of principles. Amphoreus adapts this from safety alignment to content quality: instead of "is this harmful?", the principles ask "does this have one main idea?", "does this avoid AI patterns?", "is this grounded in specific experience?"

**Multi-model ensemble.** Using model diversity to catch blind spots is well-established in ML ensembles. The specific application—parallel evaluation with 3 heterogeneous LLMs at their cheapest tiers—is a cost-effective adaptation. At ~$0.003/post for 3 models, the marginal cost of the ensemble is negligible. The system computes inter-model agreement as a confidence signal.

**Continuous confidence scoring.** When sufficient data exists (≥ 10 observations), the verifier switches from binary pass/fail to continuous 0.0-1.0 confidence per principle. This preserves the full gradient for downstream reward learning, following the principle from DPO (Rafailov et al., 2023, arXiv:2305.18290) that continuous preference signals are strictly more informative than binary ones.

**Learned principle weights.** Each principle gets a weight based on Cohen's d effect size of (pass vs. fail) × engagement. Principles where violation has no engagement impact become "soft" (advisory, not gating). This is a form of reward shaping—the system learns which quality dimensions actually matter for audience response, not just editorial preference.

**Emergent principle discovery.** The system periodically rediscovers its entire principle set from cross-client engagement data. This parallels Cyrene's emergent dimension discovery but operates at the system level rather than per-client. Discovered principles must have evidence_count ≥ 3 to be used, providing a minimum statistical bar.

### Novel aspects

1. **Constitutional score as a weighted aggregate.** Rather than requiring all principles to pass (as in Bai et al.), the score is a weighted mean of per-principle confidences. This allows the system to tolerate violations of low-impact principles while strictly gating high-impact ones—a pragmatic adaptation of constitutional AI for content quality.

2. **Cross-model agreement as quality signal.** When models disagree on a principle, the confidence for that principle is naturally lower (wider range → lower agreement score). This disagreement signal itself becomes useful—it identifies posts that are genuinely borderline rather than clearly good or bad.

### Concerns

- **Principle set stability.** With weekly rediscovery, the principle set can shift. The system needs a more robust mechanism for tracking principle evolution—e.g., versioned principle sets with explicit migration.
- **Model availability.** If Gemini or GPT API calls fail, the ensemble degrades to a single-model evaluation. The system handles this gracefully but loses the diversity benefit.

---

## 4. LOLA — LinUCB Online Learning Agent

### What it does
LOLA manages content strategy exploration via a hybrid bandit system: UCB1 for topic arms, Thompson Sampling for format arms, with a continuous reward field for clients with sufficient data. It decides which content directions to explore and which to exploit.

### Literature context

**UCB1 (Auer et al., 2002).** The Upper Confidence Bound algorithm is the canonical solution for the explore-exploit tradeoff in multi-armed bandits. LOLA's implementation is faithful: `ucb_score = mean_reward + α√(ln(total_pulls)/n_pulls)`. The adaptive α (scaling with arm density) is a standard modification for non-stationary environments.

**Thompson Sampling.** Used for format arms (categorical, low sample size), Thompson Sampling (Thompson, 1933; Chapelle & Li, 2011) is well-suited for the small-sample regime where format experiments operate. The Beta-Bernoulli model maps rewards to [0,1] for Beta distribution sampling.

**Continuous reward field.** This is LOLA's most distinctive component. Once ≥ 10 content points exist in embedding space, LOLA switches from discrete arm-based selection to a continuous Gaussian kernel reward field. Each published post becomes a point in 384-dimensional embedding space with an associated reward. Selection uses an acquisition function that balances:
- **Expected reward:** kernel-weighted average of nearby points' rewards
- **Exploration bonus:** density-based—regions with fewer observations get higher bonus
- **Time decay:** exponential decay so recent observations matter more
- **ICP modulation:** rewards are adjusted by how well the post attracted target-profile engagers

This is structurally equivalent to Gaussian Process Upper Confidence Bound (GP-UCB, Srinivas et al., 2010, arXiv:0912.3995), the standard algorithm for Bayesian optimization in continuous spaces. GP-UCB provides theoretical no-regret guarantees under certain kernel assumptions. LOLA's Gaussian kernel with median-heuristic bandwidth selection is a standard approximation that avoids the O(n³) cost of full GP inference.

The key insight is that content topics are not discrete categories—they're points in a continuous semantic space. A post about "AI in clinical trial protocol design" is close to both "AI in healthcare" and "regulatory compliance" but not identical to either. The continuous field captures these partial overlaps naturally.

**Embedding-based arm matching.** When discrete arms are used, observations are matched to arms via cosine similarity of sentence embeddings (all-MiniLM-L6-v2), not keyword matching. This ensures that a post about "why your CAC math is wrong" correctly matches the `ecommerce_metrics_contrarian` arm even though the keywords don't overlap. The 0.30 similarity threshold is learned from the client's own matching distribution.

**ICP-modulated reward.** Each arm tracks rolling ICP match rates (the fraction of engagers who match the Ideal Customer Profile). Arms with consistently off-target audiences get ICP-retired, while high-ICP arms get boosted exploration. This creates a two-dimensional optimization: not just "what gets engagement" but "what gets engagement from the right people." This parallels the quality-diversity tradeoff in recommendation systems (Kunaver & Požrl, 2017).

### Novel aspects

1. **Hybrid discrete-continuous architecture.** The transition from arm-based bandits to continuous reward fields as data accumulates is, to my knowledge, unique. Most bandit systems commit to one formulation. LOLA treats arms as a bootstrap mechanism that gracefully degrades into a continuous field.

2. **Adaptive threshold learning.** All LOLA thresholds—exploration rate, retirement streak, ICP boundaries, arm match similarity—are learned from the client's own data distribution. The `recompute_thresholds()` method replaces every hard-coded constant with a percentile-based estimate. The `soft_bound()` function logs anomalies (values > 3σ from historical mean) without clipping, preserving the learned value while alerting operators.

3. **ICP-aware arm retirement.** Arms can be retired for two independent reasons: declining engagement (consecutive declining posts) OR declining audience quality (rolling ICP match rate below threshold). This prevents the "viral but wrong audience" failure mode where a topic gets high engagement from people who will never buy.

### Concerns

- **Embedding model dependency.** The continuous field and embedding-based matching depend on sentence-transformers. If the model is unavailable, the system falls back to keyword substring matching—a substantial degradation.
- **Reward field scalability.** The kernel score computation is O(n) per candidate, O(n×k) for k candidates. With the 500-point cap this is fine, but the system would need spatial indexing (e.g., ball trees) to scale beyond ~2000 points.
- **Bandwidth selection.** The median heuristic for Gaussian kernel bandwidth is a reasonable default but not optimal. Bayesian bandwidth selection (as in GP-UCB with learned hyperparameters) would improve the reward field quality.

---

## 5. Alignment Scorer — 360Brew-Aware Consistency Check

### What it does
Computes cosine similarity between a draft's embedding and a cached "client identity fingerprint"—the centroid embedding of accepted posts, content strategy, LinkedIn profile, and ICP definition. Scores are classified as strong/moderate/drift against learned thresholds.

### Literature context

**360Brew and topical authority.** LinkedIn's 360Brew recommendation model (Firooz et al., 2025, arXiv:2501.16450) uses a decoder-only foundation model for personalized ranking. The companion paper on LinkedIn's large-scale retrieval system (Ramanujam et al., 2025, arXiv:2510.14223) describes the LLaMA 3 dual-encoder that cross-references posts against author topical authority. Posts that drift from established pillars are suppressed before entering the ranking pipeline.

Amphoreus' alignment scorer is a pre-publication approximation of this: if the draft's embedding is far from the client's established content fingerprint, 360Brew is likely to suppress it. The embedding-based approach (OpenAI text-embedding-3-small, 1536 dims) is a reasonable proxy for the dual-encoder retrieval step described in the LinkedIn papers.

**Learned thresholds.** The strong/drift thresholds are learned from (alignment_score, engagement) pairs via engagement split analysis: for each candidate threshold, compute the mean engagement gap between above-threshold and below-threshold posts. The threshold with the highest gap becomes the boundary. This is a simple one-dimensional threshold optimization that avoids the need for a complex model.

### Novel aspects

1. **Fingerprint construction from heterogeneous sources.** The fingerprint isn't just accepted posts—it includes content strategy documents, LinkedIn profile headline/about, and ICP definition. This creates a richer identity signal than post-only embeddings.

2. **LLM fallback.** When embeddings are unavailable, the scorer falls back to a Claude Haiku evaluation that produces topic-level decomposition (aligned_topics, drift_topics). This provides richer diagnostic information than the scalar embedding similarity.

### Concerns

- **Embedding drift.** The fingerprint is cached for 24 hours. If a client's content strategy evolves (new topic pillars), the fingerprint lags. More frequent recomputation or a weighted recency scheme would help.
- **Proxy fidelity.** OpenAI's text-embedding-3-small is not the same model that LinkedIn's retrieval system uses. The cosine similarity may not correlate well with actual 360Brew suppression. This is inherently a "best available approximation" and should be validated empirically.

---

## 6. ICP Scorer — Audience Quality Signal

### What it does
After a post is published and generates engagement, the system fetches engager profiles from LinkedIn (via Apify) and uses Claude to score each engager's headline against the client's Ideal Customer Profile description. Returns a continuous 0-1 score per engager and a segment breakdown.

### Literature context

**LLM-native scoring.** Rather than building a classifier from labeled ICP/non-ICP examples, the system uses Claude to directly evaluate "does this person's LinkedIn headline match this free-text ICP description?" This is an application of the "LLM-as-classifier" pattern, where the model's zero-shot understanding of professional roles replaces feature engineering.

The continuous 0-1 scoring (not categorical E/A/N/X labels) preserves the full gradient of relevance. This aligns with the broader trend toward continuous preference modeling in RLHF research (Ethayarajh et al., 2024, arXiv:2402.01306, "KTO: Model Alignment as Prospect Theoretic Optimization").

**Adaptive segment boundaries.** The exact_icp/adjacent/neutral boundaries are learned from the client's historical ICP score distribution (25th/50th/75th percentiles when ≥ 15 observations exist). This means a client whose audience is generally low-ICP doesn't get artificially high violation rates from tight thresholds.

### Novel aspects

1. **ICP as a reward component.** Most content systems optimize for engagement. Amphoreus optimizes for *engagement from the right audience*. The ICP score feeds back into RuanMei's composite reward, LOLA's arm selection, and the constitutional verifier's weights. This creates a system that can distinguish between "viral with the wrong audience" and "moderate reach with perfect audience."

2. **Anti-ICP description.** The ICP definition includes an explicit anti-description ("people we do NOT want engaging"). This provides the LLM scorer with both positive and negative exemplars, improving precision on borderline cases.

### Concerns

- **Engager fetching cost and latency.** Each post requires an Apify actor run to fetch engager profiles. The system mitigates this with minimum reaction thresholds (< 10 reactions → skip), reaction-only fetching (no comments actor), and per-post caps (30 results). But the external dependency adds fragility.
- **Headline-only signal.** Many LinkedIn profiles have vague or aspirational headlines. The system handles this by scoring conservatively (0.3-0.5 for ambiguous headlines), but the signal is inherently noisy.

---

## 7. AdaptiveConfig — Three-Tier Cascade Framework

### What it does
Every learnable parameter in the system uses the same pattern: per-client learned value → cross-client aggregate → hard-coded default. The `AdaptiveConfig` base class implements caching, staleness detection, history logging, and the cascade resolution logic.

### Literature context

This is a straightforward implementation of the **progressive precision** pattern common in Bayesian hierarchical models. Per-client parameters have the most variance but the least data; cross-client parameters have more data but less relevance; defaults have infinite "data" but no client specificity. The cascade naturally handles the bias-variance tradeoff.

**soft_bound()** is a notable design choice. Instead of clipping learned values to a hard range (which would silently discard information), it accepts any value but logs a warning if the learned value is > 3σ from historical mean. This follows the principle that anomalous learned values may be genuinely correct for edge-case clients—clipping would impose a prior that may not hold.

### Novel aspects

1. **History logging.** Every config computation is appended to `adaptive_config_history.jsonl`, creating an audit trail of how the system's parameters evolved over time. This enables retrospective analysis of whether parameter changes correlated with engagement changes.

2. **Module-keyed persistence.** All adaptive configs for a client are stored in a single `adaptive_config.json` with module-level keys, preventing the proliferation of per-module config files.

---

## 8. Cross-Client Learning Network

### What it does
Three layers of cross-client knowledge transfer:
1. **Universal patterns** — LLM-extracted writing mechanics that appear in top performers across 3+ clients
2. **Hook library** — Top-quartile hooks with engagement metadata, retrieved by embedding similarity to the client's content direction
3. **Cold-start LOLA seeding** — Auto-generated arms for new clients from universal patterns + client ICP/strategy

### Literature context

**Transfer learning for cold start.** The cold-start problem in recommendation systems is well-studied (Schein et al., 2002; Lam et al., 2008). Amphoreus' approach—using cosine similarity in a numeric profile vector to find the most similar existing client, then transferring LOLA arms with halved confidence and engagement model coefficients with inflated residuals—is a simple but principled warm-start strategy. The confidence reduction mirrors the standard practice of wider priors on transferred parameters.

**Universal pattern extraction.** The system asks Claude to identify writing mechanics that appear in top-25% performers across multiple clients. This is a form of meta-learning: instead of learning what works for each client independently, it extracts patterns that generalize. The confidence scoring (based on evidence count and reward lift) provides a natural weighting for downstream use.

**Embedding-based hook retrieval.** The hook library uses LOLA's reward-weighted centroid as a query vector to retrieve hooks most relevant to the client's established content direction. This combines collaborative filtering (cross-client hooks) with content-based filtering (embedding similarity). The 60/40 blend of similarity and engagement score prevents the retrieval from being dominated by either factor.

### Novel aspects

1. **Structural + LLM-generated patterns.** The `cross_client.py` module computes structural patterns (depth weights, content length, posting cadence) from numeric features, while `cross_client_learning.py` extracts semantic patterns via LLM. Both are stored in `universal_patterns.json`, giving the system both interpretable statistics and rich qualitative insights.

2. **Profile-based client similarity.** The numeric profile vector includes engagement statistics, content length distribution, posting cadence, depth weights, and LOLA arm performance—a comprehensive client fingerprint. Cosine similarity on this vector provides a fast, interpretable similarity metric that doesn't require LLM calls.

---

## 9. Engagement Predictor — Pre-Publish Scoring

### What it does
Ridge regression from post features (character count, posting hour/day, alignment score, Cyrene dimension scores, constitutional results) to predicted engagement reward. Returns the prediction, a 95% confidence interval from leave-one-out residuals, and the top 3 positive/negative feature contributions.

### Literature context

**Ridge regression as a simple baseline.** The choice of ridge regression (implemented from scratch in stdlib math) over more complex models is deliberate. With 20-40 observations and 5-15 features, complex models would overfit. Ridge's L2 regularization is the minimum viable regularization. This aligns with the recommendation in the bandit literature to use simple models for reward prediction in low-data regimes (Chu et al., 2011, "Contextual Bandits with Linear Payoff Functions").

**Leave-one-out confidence intervals.** Using LOO residuals for confidence intervals is more honest than in-sample residuals for small n. The 95% CI width directly communicates prediction uncertainty—the system explicitly says "this post scores 0.73 ± 0.52" rather than presenting a point estimate without uncertainty.

**Feature contribution decomposition.** The top-3 positive/negative breakdown (coefficient × normalized feature value) is a linear analog of SHAP values. It answers "what would push this post from 0.73 to 0.85?" directly.

### Novel aspects

1. **No external ML dependencies.** The entire ridge regression—matrix multiplication, Gauss-Jordan elimination, z-normalization—is implemented in Python stdlib math. This eliminates the numpy/sklearn dependency chain and makes the predictor self-contained.

2. **Dynamic feature discovery.** The feature set is discovered from the data, not hard-coded. Features present in ≥ 30% of observations are included. As Cyrene dimensions and constitutional results accumulate, the feature set automatically expands without code changes.

### Concerns

- **R² values are often negative.** The LOO R² for many clients is negative, meaning the model is worse than predicting the mean. This is expected with 5 features and 20-30 observations, especially when the most informative features (Cyrene dimensions, constitutional results) aren't yet populated. The system gracefully handles this by reporting the R² in the model metadata, but consumers should interpret predictions cautiously when R² < 0.
- **Feature sparsity.** Currently, most clients have no Cyrene dimension or constitutional data backfilled onto observations (these are newer pipeline components). The predictor is operating with only 5 basic features (char_count, posting_hour, posting_day, edit_similarity, alignment_score). Its value will increase substantially as the draft_map backfill populates richer features.

---

## 10. Temporal Orchestrator — Data-Driven Scheduling

### What it does
Replaces fixed posting cadences (Mon/Wed/Thu) with data-driven scheduling based on per-client day-of-week and hour-of-day performance, topic-time correlations, trending topic acceleration, and post-performance cool-down periods.

### Literature context

This is a straightforward application of temporal feature engineering from engagement data. The 90-minute amplification window (when the client's ICP audience is most active) reflects LinkedIn's known algorithm behavior where early engagement velocity determines reach. The cool-down period after high-performing posts addresses feed cannibalization—a phenomenon documented in the social media optimization literature.

---

## 11. Series Engine — Multi-Post Narrative Arcs

### What it does
Plans 4-6 post series around a theme, with arc roles (setup → tension → insight → application → synthesis). Monitors engagement trajectory within a series and recommends wrapping or extending. Injects series context into Stelle's prompt so each post builds on the previous one.

### Literature context

**Compound authority effects.** LinkedIn's 360Brew model rewards topic consistency over 90 days with exponential authority gains (Firooz et al., 2025). The series engine explicitly exploits this by sustaining a theme across multiple posts, each building on the previous one's established topical authority.

**Arc-aware generation.** The narrative arc roles ensure that posts within a series serve different functions, preventing repetition while maintaining thematic coherence. This is a practical application of narrative theory to content strategy.

### Novel aspects

1. **Engagement-triggered wrap/extend signals.** The system monitors engagement trend within a series and generates recommendations: "accelerating" → extend, "declining" → wrap. This closes the feedback loop between audience response and series planning.

---

## 12. Market Intelligence — Autonomous Competitive Monitoring

### What it does
Monitors top LinkedIn creators per industry vertical, extracts trending topics, hook shifts, and whitespace opportunities. Feeds signals into Stelle as soft context. Auto-seeds vertical mappings for unmapped clients.

### Literature context

This is competitive intelligence applied to content strategy. The creator registry, weekly scraping, signal extraction pipeline, and adaptive scraping parameters (scaling with signal yield) create an autonomous monitoring loop that expands the observation space for generation.

The embedding-based hook retrieval (using LOLA's reward-weighted centroid as query) ensures that market intelligence is filtered for client relevance rather than presented as generic industry trends.

---

## 13. Ordinal Sync — The Learning Heartbeat

### What it does
The hourly sync loop is the system's central learning heartbeat. Each cycle:
1. Fetches post analytics from Ordinal
2. Scores observations in RuanMei
3. Persists quality embeddings for Cyrene's linear projection
4. Recomputes Cyrene adaptive dimension weights (2c)
5. Recomputes depth weights (2d)
6. Recomputes alignment thresholds (2e)
7. Recomputes constitutional principle weights (2f)
8. Recomputes reward component weights (2g)
9. Recomputes ICP segment thresholds (2h)
10. Recomputes CV gating thresholds (2i)
11. Builds engagement prediction model (2j)
12. Builds client profile vector (2k)
13. Runs ICP scoring for unscored posts (4)
14. Updates LOLA bandit arms (7)
15. Checks series health (8)
16. Refreshes universal patterns and hook library (9)
17. Updates structural cross-client patterns (9a)
18. Discovers constitutional principles (9c)
19. Runs market intelligence cycle (10)

This is a 19-step learning pipeline that runs asynchronously in the background. Every component's parameters are recomputed from the latest data on every cycle.

### Design rationale

The design is an explicit alternative to online gradient descent. Rather than updating parameters incrementally with each new observation, the system recomputes everything from scratch every hour. This has two advantages:
1. **Idempotency.** The system can crash and restart without state corruption.
2. **Full recomputation.** Late-arriving data (engagement metrics grow over time) is automatically incorporated. A post that gets 500 impressions at hour 1 and 5,000 at hour 24 is rescored with the correct engagement.

The tradeoff is compute cost—full recomputation is wasteful when only 1-2 observations changed. At the current scale (22 clients, ~500 total observations), this is negligible.

---

## 14. Architectural Assessment

### Strengths

1. **Closed-loop learning.** Every published post feeds back into every parameter. This is rare in content generation systems, which typically treat generation and analytics as separate products.

2. **Per-client adaptation.** Nothing is global-only. Every threshold, weight, and strategy is learned per client with a principled cascade to aggregate and default values.

3. **Bitter lesson compliance.** The system consistently favors expanding the observation space (open-ended analysis, freeform critique, embedding-based matching) over encoding human heuristics (predefined categories, fixed rubrics, keyword matching).

4. **Graceful degradation.** Every component has a cold-start fallback. New clients get defaults; clients with some data get aggregate-based parameters; mature clients get fully learned parameters. The system never crashes on insufficient data—it just becomes less personalized.

5. **Continuous over categorical.** ICP scores, constitutional evaluations, Cyrene critiques, and reward signals all use continuous values rather than categorical labels. This preserves information that downstream components can exploit.

### Weaknesses

1. **No causal identification.** All learning is correlational. The system discovers that "posts with concrete numbers get higher engagement" but cannot distinguish whether numbers cause engagement or whether posts that merit numbers are about inherently more engaging topics. Causal identification would require controlled experiments (A/B testing), which the architecture doesn't support.

2. **Single-generation-model dependency.** Cyrene's critique, RuanMei's analysis, LOLA's arm seeding, and constitutional discovery all use Claude. A systematic failure mode of Claude (e.g., consistently rating a certain style too high) would propagate through the entire learning pipeline. The constitutional verifier's multi-model ensemble mitigates this for verification but not for generation or analysis.

3. **No online experiment infrastructure.** The system cannot run controlled A/B tests. It observes what was published and learns from it, but cannot deliberately vary a single factor (e.g., posting time) while holding others constant. This limits the system to observational learning, which requires more data than experimental learning to reach the same confidence.

4. **Embedding model fragility.** The continuous reward field, alignment scoring, hook retrieval, arm matching, and quality projection all depend on embedding models (all-MiniLM-L6-v2 for LOLA, text-embedding-3-small for alignment, OpenAI for fingerprinting). Changes to these models (API updates, deprecations) could silently degrade learned representations. The system would benefit from periodic embedding model validation.

5. **Temporal confounding.** Many parameters are recomputed from historical data that spans months. Changes in LinkedIn's algorithm, client's audience growth, or industry trends create temporal confounds that the system doesn't explicitly model. Time decay in LOLA's reward field partially addresses this, but the other components treat all historical observations equally.

### What makes this different from Virio's A/B model

A preference-based A/B model picks between two variants based on binary selection feedback. Amphoreus' architecture is strictly more expressive in several dimensions:

1. **Reward dimensionality.** A/B produces a single binary signal (A or B). Amphoreus produces a continuous composite reward decomposed into depth, reach, engagement rate, and ICP quality. Each component feeds into different learning systems.

2. **Strategy space.** A/B compares two pre-generated variants. Amphoreus operates in a continuous 384-dimensional content strategy space where the system can explore arbitrary directions, not just predefined variants.

3. **Cold-start transfer.** A/B needs months of per-client data. Amphoreus transfers knowledge from similar clients from day one via client profile similarity and universal patterns.

4. **Pre-publish prediction.** A/B requires publishing both variants and waiting for engagement. Amphoreus can predict engagement before publishing and explain which features to change.

5. **Quality verification.** A/B has no quality gate. Amphoreus runs a multi-model constitutional check that catches issues no single model would flag.

The fundamental advantage is that Amphoreus operates in a *learned* parameter space that grows richer with every observation, while A/B operates in a *fixed* comparison space that produces one bit of information per experiment.

---

## References

1. Madaan, A., Tandon, N., Gupta, P., et al. (2023). "Self-Refine: Iterative Refinement with Self-Feedback." *NeurIPS 2023*. arXiv:2303.17651.
2. Firooz, H., Sanjabi, M., Englhardt, A., et al. (2025). "360Brew: A Decoder-only Foundation Model for Personalized Ranking and Recommendation." arXiv:2501.16450.
3. Ramanujam, S.S., Alonso, A., Kataria, S., et al. (2025). "Large Scale Retrieval for the LinkedIn Feed using Causal Language Models." arXiv:2510.14223.
4. Bai, Y., Kadavath, S., Kundu, S., et al. (2022). "Constitutional AI: Harmlessness from AI Feedback." arXiv:2212.08073.
5. Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." *NeurIPS 2023*. arXiv:2306.05685.
6. Huang, J., Chen, X., Mishra, S., et al. (2023). "Large Language Models Cannot Self-Correct Reasoning Yet." *ICLR 2024*. arXiv:2310.01798.
7. Srinivas, N., Krause, A., Kakade, S.M., et al. (2010). "Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design." *ICML 2010*. arXiv:0912.3995.
8. Rafailov, R., Sharma, A., Mitchell, E., et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." *NeurIPS 2023*. arXiv:2305.18290.
9. Ethayarajh, K., Xu, W., Muennighoff, N., et al. (2024). "KTO: Model Alignment as Prospect Theoretic Optimization." *ICML 2024*. arXiv:2402.01306.
10. Sutton, R.S. (2019). "The Bitter Lesson." *Blog post*, March 13, 2019.
11. Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). "Finite-time Analysis of the Multiarmed Bandit Problem." *Machine Learning*, 47(2-3), 235–256.
12. Thompson, W.R. (1933). "On the Likelihood that One Unknown Probability Exceeds Another in View of the Evidence of Two Samples." *Biometrika*, 25(3-4), 285–294.
13. Li, L., Chu, W., Langford, J., & Schapire, R.E. (2010). "A Contextual-Bandit Approach to Personalized News Article Recommendation." *WWW 2010*.
14. Chapelle, O. & Li, L. (2011). "An Empirical Evaluation of Thompson Sampling." *NeurIPS 2011*.
15. Sun, Q., et al. (2024). "LLaVA-Critic: Learning to Evaluate Multimodal Models." arXiv:2410.02712.
16. Wang, Y. & Atanasova, P. (2025). "Self-Critique and Refinement for Faithful Natural Language Explanations." *EMNLP 2025*. arXiv:2505.22823.
17. Dewri, R. (2025). "GIER: Gap-Driven Self-Refinement for Large Language Models." arXiv:2509.00325.
18. Ahmadian, A., Cremer, C., Gallé, M., et al. (2024). "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs." arXiv:2402.14740.
