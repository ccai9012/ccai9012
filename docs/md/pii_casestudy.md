# A.1.2: Case Study #2: Mechanics & Application

## _The Revealing Mechanism_


_Note 1: This outline provides guidance specific to Case Study #2, complementing the **general case study rubric**_ [**[link]**](casestudy.html).
_Note 2: See [**[link]**](pii_examples.html) for some illustrative examples of mechanisms._




## The Brief


Understand the **high-level algorithmic structure** of an AI-powered method, function, or system.

Investigate how it operates and analyse how its design shapes behaviour.

You are **not analysing surface outputs.**
You are analysing the **underlying process**.

Your analysis should address the following questions.



### 1. What problem or function is the system designed to address?


- **Problem / Function**
    - What is the **main objective** of the task?
    - Why is this problem important?
    - Why does this mechanism exist within the system?




### 2. What is the mechanism?


- **Mechanism Explanation**
    - Describe clearly **how it operates**.
    - Explain how the overall task breaks down into **algorithmic steps or modules**.
- **Your explanation should show**
    - The **overall workflow of the system**.
    - The **specific mechanism** you are analysing.
    - **Inputs → transformations → outputs**.

Avoid unnecessary jargon.
Focus on **structural clarity**.




### 3. What does this mechanism enable — and what does it limit?


- **Capabilities**
    - What strengths or behaviours does the mechanism enable?
- **Limitations**
    - What trade-offs, distortions, or failure patterns arise from its design?




### 4. If the mechanism changed, what would change?


- **Causal reasoning**
    - Explain how structural design choices create:
        - strengths;
        - limits; and
        - failure patterns.
- **Comparison**
    - Where helpful, compare your mechanism to an **alternative design or configuration**.

Your analysis should reveal **one key insight** about how the mechanism shapes system behaviour.




### What a Strong Submission Demonstrates


A strong submission will:

- Isolate **one mechanism clearly**
- Diagram the **overall workflow of the system**
- Zoom into the **selected module**
- Identify **inputs → transformations → outputs**
- Explain how **structural design leads to behavioural outcomes**

Your writing should demonstrate **algorithmic thinking and structural clarity**.




### Choosing Your Mechanism

Your mechanism should be:

- **Specific**
- **Structurally identifiable**
- **Narrow enough to analyse clearly**
- **Causally linked to system behaviour**

Do not choose something broad like **“AI”** or **“machine learning.”**
Focus on **one clearly defined mechanism** within a system.

Explain the mechanism **conceptually**.
You may reference specific techniques where necessary to clarify how it works.

Do **not** turn this into a technical manual.

**Clarity over breadth.
Mechanism over surface.**

See [**[link]**](pii_examples.html) for some illustrative examples of mechanisms.

<!--
**Application-Based Mechanisms**
```
  ■ Social Media Feeds (e.g. TikTok / Instagram)
    ▪ Ranking logic
      • What determines what appears first?
      • What behaviour is being optimised?
    ▪ Feedback loop
      • How does user interaction reshape future content?

  ■ Streaming Platforms (e.g. Spotify / Netflix / YouTube)
    ▪ Similarity logic
      • How does the system define “similar”?
      • (e.g. collaborative filtering, embedding similarity)
    ▪ Novelty vs repetition balance
      • How does the system balance exploration and reinforcement?

  ■ Conversational AI (e.g. ChatGPT)
    ▪ Response selection
      • How does the system choose one answer among many possibilities?
      • (e.g. probability ranking, sampling control)
    ▪ Context limitation
      • What happens when memory runs out?
```
**Technique-Based Mechanisms**
```
  ■ Representation
    ▪ How information is internally structured or compressed
      • (e.g. vector representation, latent space)

  ■ Optimisation
    ▪ What the system is trying to maximise or minimise
      • (e.g. loss function, reward objective)

  ■ Learning dynamics
    ▪ How the system updates itself over time
      • (e.g. gradient-based adjustment)

  ■ Generation process
    ▪ How outputs are produced step by step
      • (e.g. iterative refinement, controlled randomness)
```



### What You Must Do

1. Show the overall workflow of the system
2. Isolate the specific mechanism you are analysing
3. Trace inputs → transformations → outputs
4. Explain how structural design choices create strengths, limits, and failure patterns


Use technique only to clarify structure.
Do not turn this into a technical manual.

Clarity over breadth.
Mechanism over surface.   -->





## Deliverables


Your submission must include **four parts**.




### i. Annotation (≈500 words)


Provide a short written annotation accompanying your case study.

- **Title**
    - A clear title identifying the system or mechanism studied.

- **Caption / Description**
    - A brief written explanation summarising the focus of your case study.




### ii. The Artefact


Your artefact consists of **two components**:

- **Supporting Explanation** — contextual material that frames the system and the mechanism you analyse
- **Mechanism Representation** — the core explanation showing how the mechanism operates

Think of this as **telling the story of the system first, then revealing how the mechanism works**.




#### Explanation Layer (Supporting Context)


This section provides the **context needed to understand the mechanism under study**.
It is essentially the **story or narrative that frames your mechanism**.

It may include slides, visual examples, short demonstrations, or other explanatory material.

Your explanation layer should generally follow the sequence below.

- **Problem**
    - Illustrate the real-world task or use case the system is designed to address.

- **System Context**
    - Provide brief background on the system or application being studied.

- **Mechanism (Overview)**
    - Identify the mechanism you will analyse.
    - The detailed explanation follows in the next section.

- **Assessment**
    - Provide preliminary observations about what the system does well and where it struggles.

- **Comparison (if relevant)**
    - Introduce any alternative approaches or configurations that may be referenced later.

The goal of this section is to **frame the system and prepare the reader for the detailed mechanism explanation**.

This corresponds to the **contextual slides or images** that accompany your core artefact in the online gallery documentation.




#### Mechanism (Core Artefact)


This section is the **central focus of the artefact**.

Explain how the mechanism operates using **clear diagrammatic representations**.

Your explanation should make visible:

- The **overall workflow of the system**
- The **specific module or mechanism** you are analysing
- **Inputs → transformations → outputs**
- How information **changes across the pipeline**

Where relevant, present **multiple levels of abstraction**:

- **System-level overview** — the overall pipeline
- **Module-level breakdown** — the mechanism you selected
- **Internal functional logic** — how the mechanism operates

If the system is complex, focus on the **overall structure plus one clearly analysed sub-mechanism**.

The goal is to make the **hidden architecture legible**.

This is **not a demo of outputs** and **not an exercise in aesthetic polish**.

Your priority is **clarity of algorithmic structure** and explaining how system behaviour emerges from design choices.




### iii. Vignette


The vignette is a **short-form video presentation** summarising your case study.

- Duration: **approximately 90 seconds**

Your video should clearly communicate:

- The **system or application studied**
- The **mechanism you investigated**
- Your **key insight about how the mechanism shapes behaviour**





### iv. Evidences (Optional)


You may include **supplementary materials** in an appendix, such as:

- Code snippets
- Parameter sweeps
- Prompt comparisons
- Additional diagrams or tables
- Experimental results






## Grading Criteria

### In a nutshell

#### We are **not** grading:

- Advanced mathematics
- Sophisticated coding
- Exhaustive technical coverage


#### We **are** grading:

| Criterion | What It Means |
|-----------|---------------|
| **Understanding** | Do you correctly explain how the mechanism works? |
| **Causal Reasoning / Comparison** | Do you link structural design choices to observable system behaviour, including how behaviour might change under alternative designs? |
| **Limitations** | Do you identify meaningful trade-offs, constraints, or failure modes introduced by the mechanism? |
| **Clarity** | Can you explain complex ideas in a precise and accessible way? |
| **Visual Explanation** | Do your diagrams or visual representations clearly communicate the mechanism and information flow? |


### Rubric

| Criterion | Excellent | Adequate | Insufficient |
|-----------|-----------|-----------|--------------|
| **Mechanism Accuracy** | Structurally correct and clearly articulated explanation of how the mechanism operates | Mostly correct explanation with minor gaps or simplifications | Superficial, vague, or incorrect description of the mechanism |
| **Causal Analysis** | Clear linkage between structural design and system behaviour, with thoughtful comparison or reasoning | Some connection between structure and behaviour, but underdeveloped | Describes outputs or observations without structural reasoning |
| **Failure Awareness** | Identifies meaningful trade-offs, limitations, or structural constraints | Mentions limitations but without depth or explanation | No meaningful discussion of limitations |
| **Clarity of Communication** | Complex mechanism explained clearly and accessibly to non-specialists | Explanation understandable but dense or uneven | Obscure, overly technical, or difficult to follow |
| **Visual Explanation** | Diagrams clearly reveal the system workflow, mechanism, and information flow | Visuals present but only partially clarify the mechanism | Visuals absent, confusing, or decorative rather than explanatory |


### Submission Deadline
<s>Original: 📅2026.04.01 ⏰00:00</s>
Extended: <span style="background-color:#a00000;color:#fff;">📅2026.04.08 ⏰00:00</span>





## The Standard


The goal is not to *describe AI*.
The goal is to **understand how it works**.

Do not remain at the surface of outputs.
Demonstrate that you can think in mechanisms, modules, and pipelines.

Don’t just use AI.
**Explain it.**
