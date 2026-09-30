# Multimodal Reasoning

**Related API:** [`ccai9012.multi_modal_utils`](../api/ccai9012/multi_modal_utils.html) · [`ccai9012.svi_utils`](../api/ccai9012/svi_utils.html) · [`ccai9012.viz_utils`](../api/ccai9012/viz_utils.html)

### Overview
**Category:** Visual-Language Reasoning

This module assumes basic Python, pandas, image-file handling, and the Week 4
multimodal LLM tutorial. Its two complementary paths are deliberately
different: CLIP maps an image and a text prompt to a similarity score, whereas
captioning and visual QA turn an image into text that can then be searched for
candidate labels.

### Learning path

1. Start with the local, deterministic CLIP sample; use Google Street View only
   after explicitly enabling the API branch and supplying a key.
2. Keep the image/coordinate/heading relationship in a manifest, then map the
   CLIP-derived score back to a point-level index.
3. Run the controlled image-generation experiment and distinguish generated
   images, captions, visual-QA answers, and keyword-extracted labels.
4. Treat all model-derived labels as hypotheses requiring human or annotated
   reference checks.

**Modular Components:**

- **Image generation and loading**
  - Local model or API initialisation
  - Controlled prompt and seed inputs
- **Image to text**
  - Image captioning
  - Vision-language question answering
- **Text to evidence**
  - Keyword extraction
  - Candidate-label frequency and co-occurrence summaries

### Use Cases
- Do AI models associate certain architectural styles with particular geographic regions unfairly?
- Urban light pollution areas spotting based on facade material analysis
- Can we visualize gentrification through facade transformation using historical vs. recent street views?
- Thermal defect spotting based on facade and indoor infrared images

### Code Examples

#### Controlled Image Generation and Material-Label Analysis

**Research question:** When a neutral architectural prompt is used repeatedly, which material labels recur in the images that an AI model generates? This compact experiment makes a possible material-association pattern visible: hold the prompt steady, vary only the random seed, then examine how vision models describe the resulting image set. It analyses model-produced labels—not the true material distribution of buildings or a confirmed bias conclusion.

<p align="center">
  <img src="../figs/gen_image_eval_flow.svg" alt="A text-to-image-to-text flow. A neutral prompt is turned into generated building images with Stable Diffusion, then BLIP and Qwen-VL turn images into captions." width="100%"><br>
  <em>Start with a neutral prompt, generate images, then use captions to inspect what the models see.</em>
</p>

The flowchart combines both model directions: Stable Diffusion is the **text → image** part, while BLIP and Qwen2.5-VL are the **image → text** part. Captions and answers provide language to compare with the original prompt before moving to the label analysis.

**Four stages:**

1. **Design a controlled experiment.** Write a neutral building prompt that does not ask for a particular material, then change only the random seed. This creates varied samples while giving the comparison a clear starting point.
2. **Generate an image set (text → image).** Stable Diffusion turns the prompt and seed into images saved in `gen_imgs/`. The [Week 4 Multimodal LLM tutorial](../../weekly_scripts/wip/week4/week4_t_multimodal_llm.ipynb) introduces this prompt-and-seed behaviour.
3. **Ask vision models about the images (image → text).** BLIP captions one image as a qualitative check. Qwen2.5-VL then receives the same facade-material question for every image, so its answers can be compared across the set.
4. **View recurring labels.** Match the answers to a declared material vocabulary, save them in `output/results.csv`, and display both a frequency chart and a co-occurrence matrix.

**How to read the charts:** each frequency bar is the number of Qwen-VL answers that matched one vocabulary word. In the co-occurrence matrix, the diagonal is the count for one label; an off-diagonal cell is the count of images whose answer matched both labels. Larger cells show recurring *model-label patterns*. They do not measure real-world material prevalence or establish bias by themselves.

**Suitable applications:** use controlled samples to explore whether a generator associates occupations, gender, race, or architectural materials with particular visual cues; or use labelled reference images to assess a vision model's material recognition. The latter needs annotated reference data; neither activity alone can establish societal bias, material truth, or generalise to every setting.

**Limitations:** one prompt and 50 images are not representative; both the generator and vision model can introduce bias; a fixed vocabulary can miss relevant labels; and versions or hardware can alter output. Treat the charts as a starting point for a better-designed comparison, not as a final claim.

<p align="center">
  <img src="../figs/building_exterior_001.png" width="400"><br>
  <img src="../figs/SCR-20251218-lxrc.png" width="600"><br>
  <em>Using BLIP to identify the facade material from the images generated from StableDiffusion.</em>
</p>

#### Assessment of Conservation Status in Urban Historic Districts
**Content:**
- Categorizing SVIs of historic districts with CLIP
- Evaluating mixing index of historic and added-on buildings

**Dataset:**
- Tracked Barcelona Street View images (`582` `.jpg` files) for the default
  offline sample
- Google Street View Imagery (SVI) from Google Maps API for the explicit opt-in
  branch

The notebook uses four text prompts (historic/traditional, modern, mixed, and
open space). CLIP similarity is a ranking signal between each image and those
prompts; it is not a conservation inventory, a ground-truth historic label, or
proof that a building is historically protected. The weighted historic index
is a mapped summary of those model scores and should be checked against an
annotated conservation register or expert labels before interpretation.

<p align="center">
  <img src="../figs/clip_historic_decision_spatial.svg" alt="Decision tree from a deterministic local image sample or opt-in Google Street View API to a manifest, CLIP similarity, mapped candidate index, and human check." width="100%"><br>
  <em>Offline-first decision path and spatial evidence chain. The final index is model-derived and requires validation.</em>
</p>

### Application matrix

| Extension | Dataset needed | Annotation/reference standard | Reused functions/variables | Why it is meaningful |
| --- | --- | --- | --- | --- |
| Conservation-status screening | Municipal heritage register plus geolocated SVI | Protected-building status and boundary/date rules | `build_svi_image_manifest`, `join_clip_results`, `text_prompts`, `weighted_score` | Tests whether a candidate visual signal aligns with a defined conservation reference |
| Historic/modern mixing index | Parcel/building footprints with construction-era labels | Building-level age labels and a declared spatial aggregation unit | `sample_manifest`, `groupby([\"latitude\", \"longitude\"])`, `plot_points` | Separates a scene-level mixture hypothesis from an unsupported historic label |
| Cross-city transfer check | Matched SVI samples from multiple cities | City-specific, independently reviewed labels using the same coding protocol | `deterministic_image_sample`, `CLIPClassifier`, `all_scores` | Shows whether prompt similarity is stable across geography and imagery conditions |
| Human-in-the-loop audit | Stratified image sample and expert review form | Two or more reviewers, adjudication rule, and inter-rater agreement | `df_results`, `label_text`, `confidence`, manifest coordinates | Quantifies disagreement and prevents confidence from being mistaken for truth |

Each extension needs a new reference standard; none is completed by keyword or
CLIP matching alone.
<p align="center">
  <img src="../figs/SCR-20251218-lvlc.jpeg" width="400"><br>
  <em>Using CLIP to identify the historical status of the urban block.</em>
</p>
