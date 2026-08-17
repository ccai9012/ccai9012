# Illustrative Mechanisms You Could Study


Below are **illustrative examples of mechanisms** you might investigate.

These are **suggestions, not restrictions**.
You may choose other systems if the mechanism is **clearly defined and structurally identifiable**.




## Application-Based Mechanisms


- **Social Media Feeds** *(e.g. TikTok / Instagram)*

    - **Ranking systems**

        - Content in your feed is **ranked before being shown**.
        - Platforms prioritise posts using signals such as **engagement, recency, or predicted interest**.
        - Study how ranking determines **what appears first** and what becomes visible.

    - **Engagement feedback loops**

        - User actions (likes, watch time, shares) reshape what the system shows next.
        - Study how interaction data **updates and reinforces future recommendations**.


- **Streaming Platforms** *(e.g. Spotify / Netflix / YouTube)*

    - **Similarity-based recommendation**

        - The system suggests content **similar to what users previously consumed**.
        - Study how similarity is computed using **behaviour patterns or learned embeddings**.

    - **Exploration vs reinforcement**

        - Platforms must balance recommending **familiar content** and introducing **new items**.
        - Study how systems prevent feeds from becoming **too repetitive or predictable**.


- **Conversational AI** *(e.g. ChatGPT)*

    - **Response selection**

        - Language models generate many possible continuations and must **select one to output**.
        - Study how **probability ranking or sampling** influences the final response.

    - **Reasoning vs standard generation**

        - Some systems introduce **reasoning steps or structured prompting** instead of direct completion.
        - Study how reasoning pipelines differ from **simple next-token generation**.

    - **Context window limits**

        - Conversational systems remember **only a limited amount of prior text**.
        - Study how limited context affects **long conversations and response quality**.




## Model Architecture Mechanisms


- **Attention-based models (Transformers)**

    - **Learning relationships across information**

        - Some models analyse input by allowing different parts of the data to **attend to one another**.
        - Study how this structure helps models capture **long-range relationships in language or data**.


- **Adversarial models (GANs)**

    - **Learning through competition**

        - One model **generates outputs**, while another **evaluates them**.
        - Study how this competitive interaction improves the **realism of generated images or media**.


- **Encoder–decoder models**

    - **Compressing and reconstructing information**

        - One component converts input into a **compact internal representation**, while another reconstructs the output.
        - Study how this structure enables tasks such as **translation, summarisation, or reconstruction**.


- **Diffusion-style models**

    - **Gradual refinement**

        - Some generative systems create outputs through **many small refinement steps** rather than a single prediction.
        - Study how iterative refinement produces **detailed images or media**.




## Technique-Based Mechanisms


- **How systems represent information**

    - **Encoding meaning into numbers**

        - AI systems convert text, images, or behaviour into **numerical representations**.
        - Study how vector representations enable **similarity search, clustering, or retrieval**.


- **How systems learn what to prioritise**

    - **Training signals and objectives**

        - AI models improve by adjusting themselves to **reduce error or maximise a reward signal**.
        - Study how different objectives influence **what the system becomes good at**.


- **How systems update themselves during learning**

    - **Iterative improvement**

        - Models repeatedly adjust internal parameters based on **feedback from previous predictions**.
        - Study how repeated updates gradually improve system performance.


- **How systems generate outputs step-by-step**

    - **Sequential generation**

        - Many AI systems produce results **one step at a time rather than all at once**.
        - Study how sequential generation influences **creativity, randomness, or consistency**.




## Key Advice


Do **not attempt to analyse everything**.

A strong submission focuses on **one clearly identifiable mechanism** and explains **how it shapes system behaviour**.
