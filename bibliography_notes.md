# Bibliography Analysis for Implementation

## Core References

### 1. MAP-Elites (Mouret & Clune, 2015)
**Citation**: `mouret2015illuminatingsearchspacesmapping`
**arXiv**: 1504.04909

**Key Concepts for Implementation**:
- Quality-Diversity optimization paradigm
- Grid-based archive structure
- Illumination vs optimization distinction
- Selection and mutation operators
- Coverage metrics

**Implementation Notes**:
- Use uniform grid partitioning of behavioral space
- Maintain elite (best) solution per cell
- Random selection from archive for mutation
- Track coverage and QD-score over iterations

### 2. GCG - Universal Adversarial Attacks (Zou et al., 2023)
**Citation**: `zou2023universaltransferableadversarialattacks`
**arXiv**: 2307.15043

**Key Concepts**:
- Gradient-based adversarial suffix optimization
- Greedy coordinate gradient descent
- Token-level optimization
- Transferability across models

**Implementation Notes for Baseline**:
- For black-box: use zeroth-order gradient estimation
- Optimize discrete tokens via greedy search
- Evaluate on target LLM with suffix appended
- Track peak alignment deviation (best attack found)

### 3. PAIR - Jailbreaking in 20 Queries (Chao et al., 2024)
**Citation**: `chao2024jailbreakingblackboxlarge`
**arXiv**: 2310.08419

**Key Concepts**:
- Iterative prompt refinement
- Attacker LLM generates variations
- Judge LLM evaluates success
- Efficient query budget usage

**Implementation Notes for Baseline**:
- Use GPT-4 or similar as attacker LLM
- Iteratively refine based on judge feedback
- Track number of queries to success
- Measure diversity of successful attacks

### 4. TAP - Tree of Attacks (Mehrotra et al., 2024)
**Citation**: `mehrotra2024treeattacksjailbreakingblackbox`
**arXiv**: 2312.02119

**Key Concepts**:
- Tree search over attack space
- Parallel exploration of branches
- Pruning of unpromising paths
- Depth-first and breadth-first strategies

**Implementation Notes for Baseline**:
- Implement tree structure for attack variations
- Use heuristic for pruning (e.g., judge score threshold)
- Track best leaf nodes
- Measure exploration efficiency

### 5. Manifold Projection (Li et al., 2025)
**Citation**: `Li_Yin_Jiang_Hu_Wu_Yang_Liu_2025`
**AAAI 2025**: DOI 10.1609/aaai.v39i1.32024

**Key Concepts**:
- Adversarial examples lie off natural data manifold
- Projection back to manifold for robustness
- Latent space geometry

**Relevance to Paper**:
- This paper INVERTS this concept
- Instead of projecting TO safety, map the failure manifold
- Contrast: restorative vs exploratory approach

### 6. CVT-MAP-Elites (Vassiliades et al., 2016)
**Citation**: `DBLP:journals/corr/VassiliadesCM16`
**arXiv**: 1610.05729

**Key Concepts**:
- Centroidal Voronoi Tessellation for space partitioning
- Scales MAP-Elites to higher dimensions
- More efficient than uniform grid for d > 3

**Implementation Notes**:
- Use for 3D/4D behavioral space experiments
- K-means clustering to define Voronoi cells
- Update centroids as archive fills

### 7. Universal Adversarial Attacks Geometry (Subhash et al., 2023)
**Citation**: `subhash2023universaladversarialattackswork`
**arXiv**: 2309.00254

**Key Concepts**:
- Geometric explanation for universal attacks
- Embedding space structure
- High-dimensional geometry of LLMs

**Relevance**:
- Supports hypothesis that failures have geometric structure
- Behavioral space is continuous, not discrete
- Justifies smooth interpolation in embedding space

### 8. LLM Geometry for Toxicity (Balestriero et al., 2024)
**Citation**: `balestriero2024characterizinglargelanguagemodel`
**arXiv**: 2312.01648

**Key Concepts**:
- Geometric analysis of LLM representations
- Toxicity detection via geometry
- Internal representation structure

**Relevance**:
- Complements this work (internal vs output space)
- Validates geometric approach to safety
- Potential for hybrid methods

### 9. Rainbow Teaming (Samvelyan et al., 2024)
**Citation**: `rainbowteaming`
**NeurIPS 2024**: DOI 10.52202/079017-2229

**Key Concepts**:
- Open-ended generation of diverse adversarial prompts
- Quality-diversity for red teaming
- Multi-objective optimization

**Relevance**:
- Related work in QD for LLM safety
- Comparison point for diversity metrics
- Potential baseline or complementary method

### 10. Pareto Invariant Risk Minimization (Chen et al., 2023)
**Citation**: `chen2023paretoinvariantriskminimization`
**arXiv**: 2206.07766

**Key Concepts**:
- Multi-objective optimization
- Out-of-distribution generalization
- Pareto frontier

**Relevance**:
- Potential extension: Pareto-optimal safety-utility tradeoffs
- Multi-objective formulation of alignment
- Theoretical foundation for trade-offs

## Implementation Priorities

### Must Implement (Core to Paper)
1. MAP-Elites algorithm
2. Behavioral descriptor computation
3. Alignment Deviation metric with judge committee
4. 2D heatmap visualization

### Should Implement (Baselines)
1. Random sampling baseline
2. GCG baseline (simplified black-box version)
3. PAIR baseline
4. TAP baseline (simplified)

### Could Implement (Extensions)
1. CVT-MAP-Elites for higher dimensions
2. GP-based prediction
3. Semantic interpolation
4. Rainbow Teaming comparison

## Key Algorithmic Insights from Literature

### From MAP-Elites
- **Archive structure**: Grid partitioning enables visualization
- **QD-score**: Sum of quality scores in archive (overall performance metric)
- **Illumination**: Goal is to fill archive, not find single optimum

### From Adversarial Attack Papers
- **GCG**: Gradient-based is powerful but low diversity
- **PAIR**: Iterative refinement is query-efficient
- **TAP**: Tree search explores multiple paths but can be expensive

### From Geometry Papers
- **Continuous structure**: Failures form smooth manifolds, not discrete points
- **Embedding space**: Semantic interpolation is valid
- **Attraction basins**: Regions where diverse inputs converge to similar outputs

## Harm Taxonomy Sources

Based on standard AI safety frameworks:
1. **OpenAI Moderation API categories**
2. **Anthropic's Constitutional AI harm categories**
3. **Google's Perspective API toxicity dimensions**
4. **NIST AI Risk Management Framework**

Suggested 10 categories for implementation:
1. Violence/Physical Harm
2. Hate Speech/Discrimination
3. Sexual Content
4. Self-Harm/Suicide
5. Illegal Activities
6. Privacy Violations
7. Misinformation
8. Malware/Hacking
9. Fraud/Deception
10. Regulated Advice (medical, legal, financial)
