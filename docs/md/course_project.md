# A.2: Course Project

<div style="height:4.0rem"></div> 

<hr style="height: 2px; border: none; background-color: #000;">

<div style="height:1.5rem"></div> 

## Overview

The course project is a **team-based, design-oriented assignment** in which students develop, prototype, and analyse an AI-assisted proposal addressing a concrete real-world problem. In contrast to the case studies, which focus on analysing existing systems or practices, the course project emphasises **making**, evaluation through use, and critical reflection through implementation.

Projects may take analytical, tool-based, or design-oriented forms, but all must demonstrate a considered synthesis of **responsibility and ethics**, **mechanisms and model behaviour**, and **datasets and data practices**, reflecting how these dimensions interact in real-world AI applications. The emphasis is on reasoned experimentation, explicit assumptions, and reflective judgement rather than technical novelty or performance optimisation.

The project is completed in teams of **4–6 students**. Tutorials provide structured opportunities for skills development, feedback, and team formation.

<div style="height:4.0rem"></div> 

<hr style="height: 2px; border: none; background-color: #000;">

<div style="height:1.5rem"></div> 

## Project Structure & Deliverables

The course project consists of the following assessed components:

- **Mid-term Pecha Kucha Presentation**: A 4-minute time-limited presentation of work in progress, used to articulate the project direction and receive formative feedback.  

- **Final Project Submission**: Think of your project as developing a set of **artefacts and evidence that form the backbone of your work**. These may include outputs, experiments, comparisons, screenshots, diagrams, workflows, quantitative observations, intermediate results, prototypes, and other documentation of your process and findings. Each should be carefully selected, organised, and **captioned to explain what it shows and why it matters**. These artefacts should correspond directly to your **slides**, and then become the common ingredients from which you construct your **video and paper**. The same evidence should therefore support a coherent project story across all formats. The submission comprises:
    - **(i) Abstract — Distilling the Story**: Write a concise abstraction of the project's **central question, approach, key result, and principal insight**. Rather than introducing new material, the abstract should distil the most important argument emerging from the work.
    - **(ii) Slides — Artefacts & Evidence**: Develop a sequence of **portrait-format slides** that organises and connects your key artefacts into a clear visual narrative. Each slide should communicate a meaningful part of the project through appropriately captioned evidence, such as intermediate outputs, comparisons, screenshots, diagrams, annotations, quantitative observations, or other relevant documentation.
    - **(iii) Video — Connecting the Story**: Develop a **4 minute video** using the slides and artefacts as its visual foundation. Use **narration, sequencing, animation, and transitions** to connect the evidence and communicate the project's motivation, development, findings, and key insights as a coherent story.
    - **(iv) Paper — Documenting the Story**: Develop a **paper** using the same artefacts and evidence, connecting them through written argument, explanation, and critical reflection. The paper should document the project's motivation, system design, implementation, results, and implications (**maximum 1500 words and 10 pages**).


- **Individual Reflection**: An individually assessed reflective statement documenting contributions and learning.


- **Peer Assessment**: Advisory peer feedback on group work and individual contributions. Further details are provided in the course outline.


<div style="height:4.0rem"></div> 

<hr style="height: 2px; border: none; background-color: #000;">

<div style="height:1.5rem"></div> 

## Milestones

<div style="height:1.0rem"></div> 

| Item | Description | Date | Individual (%) | Group (%) |
|------|-------------|------|----------------|-----------|
| **Project Team Formation Form** | Early indication of potential project partners to support team formation and initial scoping as a 100-200 word statement.  Submit via Moodle. | 2026.09.30 | – | – |
| [[**Mid-term Pecha Kucha Presentation**]](cp_pechakucha.html) | Time-limited presentation of project framing, proposed AI-assisted approach, and preliminary considerations on data, mechanisms, and responsibility. | 2026.10.21 | – | 15 |
| **Crit-Style Peer Review (Post-Midterm)** | Structured studio-style peer critique conducted during tutorial slots, focused on problem definition, methodological coherence, assumptions, and responsible AI positioning. Formative and non-graded. | 2026.10.19-23 | – | – |
| [[**Final Project Submission**]](cp_final.html)  | Submission of all core deliverables. Projects are presented through the screening of the vignette component. | 2026.12.09 | – | 25 |
| **Crit-Style Peer Review (Post-Final)** | Reflective peer critique of completed projects (via tutorial slots), emphasising evaluative judgement, strengths, limitations, and comparative learning. Formative and non-graded. | 2026.12.07-11 | – | – |
| [[**Individual Reflection**]](cp_reflect.html) | Individually assessed reflective statement on contributions, learning process, and key design decisions. | 2026.12.16 | 5 | – |
| **Peer Assessment** | Advisory peer feedback on group process and individual contributions, used to inform moderation where appropriate. | 2026.12.16| – | – |


<div style="height:4.0rem"></div> 

<hr style="height: 2px; border: none; background-color: #000;">

<div style="height:1.5rem"></div> 

## Learning Focus

<div style="height:1.0rem"></div> 

Through the course project, students are expected to demonstrate the ability to:

- **Problem Statement:** define and bound a real-world problem with clear context and relevance;
- **Solution Space:** design and justify an AI-assisted approach aligned with stated objectives;
- **Rigorous Investigation:**develop reasonable demonstrations or evaluations that provide meaningful validation;
- **Course Synthesis:** reason critically about interactions between mechanisms, datasets, and responsibility;
- **Effective Communication:** communicate intent, methods, results, and limitations clearly and coherently.

The course project serves as the **capstone assignment** for the course, integrating technical understanding, critical judgement, and professional communication into a single, coherent body of work.


<div style="height:4.0rem"></div>  

<hr style="height: 2px; border: none; background-color: #000;">

<div style="height:1.5rem"></div> 

## Possible Directions

<div style="height:2.0rem"></div> 

### Methodology

<div style="height:1.0rem"></div> 

CCAI 9012 encourages students to conduct their Course Project using programming, enabling systematic data collection and analysis, and unlocking scalable opportunities that would be cumbersome—if not intractable—to execute and control via ChatBot/UI-based GenAI alone.

While programming is strongly encouraged, we recognize that skill levels vary across the student body and welcome projects in any format, as long as they are grounded in:

- a well-framed problem
- an appropriately matched solution approach
- a rigorous process of investigation

Accordingly, it is also acceptable to construct a dataset manually through direct interaction with LLMs, provided you take appropriate precautions to avoid unintentionally biasing your results.

For students interested in developing programming-based solutions, a number of starter kits [**[link]**](starter_kits/index.html) are available. These include example code that already performs canonical ML and AI tasks. Often, simple modifications—such as swapping in a different text or image dataset—are sufficient to support exploration of new research questions.

_Note: The documentation for starter kits is being actively expanded and refined._

<div style="height:2.0rem"></div> 

### Thinking Algorithmically 

<div style="height:1.0rem"></div> 

Whether you are generating data manually or programmatically, it is helpful to break down your problem and methods into modular mappings, such as:

- **text → text** – input text, get new text (answering questions, summarising, translating)
- **image → text** – input an image, get text (captions, tags, object names)
- **text → image** – input a description, get an image (AI “drawing” what you describe)
- **image → image** – input an image, get an edited image (filters, style changes, object removal)
- **image → video** – input one or more images, get a short video (adding motion or transitions)
- **image → scalar** – input an image, get one number (e.g., a 0–100 quality score or a damage estimate)
- **vector → scalar** – input a list of numbers, get one number (e.g., a price prediction or risk score)
- **etc.** – other input/output mixes (audio, video, combinations of text–image–audio)

By intelligently chaining these modules, and then running the resulting pipeline over each data point in a dataset, you can build powerful systems to investigate varied and complex questions.

For example, to study whether different LLMs provide different responses under different circumstances, you might:

1. Define a set of prompts: **text**
2. Feed them into multiple LLMs: **text → text**
3. Compare and score their outputs: **text → scalar** (e.g., with human ratings or another model)
4. Aggregate results over the dataset: **vectors of scalars → scalars** (summary statistics)

Similarly, to study whether multimodal, image-generation–enabled LLMs exhibit professional, racial, or gender bias, you might:

1. Design prompts describing roles or scenarios: **text**
2. Generate corresponding images: **text → image**
3. Analyse those images for attributes (e.g., perceived gender/race/occupation): **image → text** and/or **image → scalar**
4. Quantify bias by aggregating these attributes: **vectors (attribute scores across images) → scalars (bias metrics)**

The key is to think in terms of these modular transformations and then compose them into end-to-end pipelines tailored to your research question.

<div style="height:2.0rem"></div> 

### Example Ideas

<div style="height:1.0rem"></div> 

Listed below are example directions (analysing AI, using AI for analysis, developing AI‑powered tools) are illustrative rather than exhaustive.  Note that many other directions are also possible.

- **Analysing AI**
    - **Problem Statement**: AI image generators used to create “professional” profile images for platforms like LinkedIn may depict certain professions (e.g., “doctor”, “engineer”, “CEO”, “nurse”, “teacher”) in ways that systematically vary by race and gender, even when prompts are neutral. This may reinforce stereotypes about which kinds of people are “typical” or “appropriate” for particular professional roles. There is limited systematic evidence on how frequently such patterns occur or how they vary across professions and systems.
    - **Possible Project Response**: Analyse outputs from one or more image‑generation systems using controlled prompts for professional roles, and assess how perceived gender and race are distributed across occupations (e.g., “engineer”, “nurse”, “CEO”).

- **AI for Analysis**
    - **Problem Statement**: In a second-year interaction design module, students submit written reflections, but there is no systematic way to see which themes or user groups are addressed across the cohort. Tutors suspect that accessibility and disability are rarely discussed, yet this has never been examined at scale. As a result, important blind spots in students’ assumptions and in the curriculum may remain invisible.
    - **Possible Project Response**: Use AI-based text analysis to scan a corpus of student reflections and identify how often issues such as accessibility, disability, and other user characteristics are mentioned.

- **Developing AI‑Powered Tools**
    - **Problem Statement**: In a first-year visual communication course, tutors must give feedback on many poster designs in short studio sessions. Time pressure and differing critique styles lead to inconsistent, often vague comments, leaving students unsure how to improve or how their work maps to assessment criteria. This makes it difficult for students to understand expectations and track their progress.
    - **Possible Project Response**: Prototype an AI-assisted feedback tool that helps tutors quickly generate structured, criteria-based comments on poster designs.

