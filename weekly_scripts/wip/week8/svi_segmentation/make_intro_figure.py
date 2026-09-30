"""Place a model arrow between the saved street-view and segmentation panels."""

from base64 import b64encode
from pathlib import Path


HERE = Path(__file__).parent
source = HERE / "sample_segmentation_pair.png"
destination = HERE / "sample_segmentation_with_model.svg"
encoded = b64encode(source.read_bytes()).decode("ascii")

# The source is the notebook's saved 1193 x 287 output. Clip its original
# left panel and its prediction-plus-legend panel without changing either.
svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1365 363"
  role="img" aria-labelledby="title desc">
  <title id="title">Street-view image through SegFormer-B5 to predicted labels</title>
  <desc id="desc">The notebook's real street-view sample is on the left. A SegFormer-B5
  model arrow points to its predicted Cityscapes pixel-label map and legend on the right.</desc>
  <defs>
    <image id="notebook-output" width="1193" height="287"
      href="data:image/png;base64,{encoded}"/>
    <clipPath id="input-panel"><rect x="25" y="38" width="510" height="287"/></clipPath>
    <clipPath id="prediction-panel"><rect x="660" y="38" width="683" height="287"/></clipPath>
    <marker id="arrowhead" viewBox="0 0 10 10" refX="9" refY="5"
      markerWidth="10" markerHeight="10" orient="auto">
      <path d="M0 0 L10 5 L0 10Z" fill="#2374a5"/>
    </marker>
  </defs>
  <rect width="1365" height="363" fill="white"/>
  <g clip-path="url(#input-panel)"><use href="#notebook-output" x="25" y="38"/></g>
  <g clip-path="url(#prediction-panel)"><use href="#notebook-output" x="150" y="38"/></g>
  <text x="596" y="155" text-anchor="middle" fill="#0b1f33"
    font-family="Arial, sans-serif" font-size="19" font-weight="bold">SegFormer-B5</text>
  <path d="M545 183 H645" fill="none" stroke="#2374a5" stroke-width="5"
    stroke-linecap="round" marker-end="url(#arrowhead)"/>
  <text x="596" y="218" text-anchor="middle" fill="#526777"
    font-family="Arial, sans-serif" font-size="17">model</text>
</svg>"""

destination.write_text(svg, encoding="utf-8")
