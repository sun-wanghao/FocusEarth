# Implementation

## Query-Adaptive Region Prioritization

[`query_processing.py`](../focusearth/query_processing.py) constructs search descriptions independently of the answering prompt. spaCy `en_core_web_sm` extracts nominal heads and identity modifiers; explicit target recovery uses structurally licensed nominal proposals. Entity occurrence order and duplicates are retained. Empty descriptions and `-` placeholders are filtered. Context reformulation preserves source-supported relations, cardinality, negation, spatial scope, and instance selectors. Unhandled questions retain their original text, and selection uses the Context ranking when no Entity remains. The private `_query_processing` subpackage contains the grammar rules and source-span bookkeeping.

[`candidate_regions.py`](../focusearth/region_prioritization/candidate_regions.py) constructs a row-major 6 × 6 source grid. Boxes expand by 5% of the cell width and height on each side, use Python integer rounding, and clip to the image boundary. Each candidate yields its full source crop and four corner-anchored crops with ceil-sized two-thirds dimensions. Crops precede the encoder's native resizing and center cropping; no intermediate candidate resize is inserted.

[`image_text_encoders.py`](../focusearth/models/image_text_encoders.py) loads CLIP, GeoRSCLIP, RemoteCLIP, and SkyCLIP-50 in that order. It applies each encoder's native image preprocessing and ViT-B/32 tokenizer. Embeddings are normalized by their norms with a `1e-12` minimum. [`encoder_scoring.py`](../focusearth/region_prioritization/encoder_scoring.py) computes FP32 NumPy cosine products with shape `[description, encoder, view, candidate]`.

[`entropy_matching.py`](../focusearth/region_prioritization/entropy_matching.py) takes the maximum over five views per candidate. It normalizes scores with population standard deviation and a `1e-6` floor in FP32, then evaluates softmax and entropy in FP64. Each description has its own median unit-temperature encoder entropy target. Temperature matching uses 40 bisection steps in `[0.25, 4]`; the closer endpoint is selected, with the lower temperature winning an exact entropy-error tie. The four resulting distributions receive equal weights.

## Multi-Query Region Selection

[`region_selection.py`](../focusearth/region_selection.py) proposes the highest-ranked Context region, then one region per Entity occurrence. Distinct candidate IDs are retained in proposal order until six are selected. Remaining slots are filled from the Context ranking. Descending score ties use ascending candidate IDs; selected boxes are serialized in source-grid order.

## Spatially Aligned Visual Representation

### Spatially Aligned Image Construction

[`canvas_geometry.py`](../focusearth/visual_representation/canvas_geometry.py) computes the reference layout from source dimensions and selected-region geometry. Its factor-aligned searches preserve token-capacity, aspect-ratio, and region-scale constraints, followed by the rounded 0.875 canvas scale. Source boxes map first to the reference canvas and then to the spatially aligned canvas; rounding at these two steps is intentional.

[`spatially_aligned_image.py`](../focusearth/visual_representation/spatially_aligned_image.py) performs one bilinear full-scene resize. Sequential binary masks render an 8-pixel outer envelope and a 4-pixel inner outline. Black excludes white, and both masks exclude the selected-region union. Selected-region pixels are therefore not replaced by outline colors.

### Region-Protected Token Compression

[`feature_mapping.py`](../focusearth/visual_representation/feature_mapping.py) maps half-open canvas rectangles to the actual Qwen post-merger grid. Floor/ceiling bounds include all intersecting cells. Protection is the deduplicated union of selected-box outer envelopes, which covers the regions and their rendered outlines. Overlapping masks can leave additional protected cells.

[`token_budget.py`](../focusearth/visual_representation/token_budget.py) computes a geometry-dependent reserve from fully represented standard regions, the actual protected-cell count, and the number of unprotected connected components. Unused protection reserve contributes to the context-group budget. Explicit feasibility checks reject insufficient reserve or component capacity; standard-region costs are not assumed to bound every edge-region arrangement.

[`region_protected_compression.py`](../focusearth/visual_representation/region_protected_compression.py) finds four-neighbor unprotected components. Group counts are proportional to component size, with deterministic rounding and at least one group per component. Morton-order quantile seeds initialize multi-source breadth-first assignment. Each group's representative is the member nearest its coordinate centroid, with row-major tie-breaking. Protected cells remain singleton groups. All groups are serialized by representative row-major position.

[`qwen3_vl_integration.py`](../focusearth/models/qwen3_vl_integration.py) applies FP32 mean aggregation and restores protected features exactly in the stream dtype. Identical memberships and representatives apply to final and DeepStack streams. The adapter shortens the image-placeholder run while retaining the full processor pixel tensor and grid. Sparse rotary positions use the representative's original row and column. The integration supports one image and batch size one, and restores model methods when the context exits.

[`qwen3_vl.py`](../focusearth/models/qwen3_vl.py) prepares factor-28 RGB geometry before the Transformers processor, wraps the image and original MME question/options in the model's chat template, and performs bfloat16/SDPA greedy generation with a 64-token cap.

## MME inference and evaluation

[`pipeline.py`](../focusearth/pipeline.py) composes the three stages. The sole benchmark entry point is [`eval_mme_realworld_rs.py`](../scripts/eval_mme_realworld_rs.py). The dataset loader preserves official annotation fields and validates the remote sensing population. The answering prompt requests Evidence and a Final answer. The parser requires one complete, last nonempty `Final answer:` line when the label occurs. Without that label, precedence is answer tag, recognized JSON key, then first nonempty line. Invalid and failed questions remain in the evaluation denominator.
