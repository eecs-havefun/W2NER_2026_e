# Conversation Summary

## Core context

The discussion focused on adapting ProcNet-style document-level annotations for W2NER training and diagnosing poor W2NER extraction performance.

A key framing established early was that the correct pipeline is **not** to directly train W2NER on raw document-level ProcNet annotations. Instead, document annotations should first be **strictly expanded into sentence-level samples**, while preserving a backtracking chain such as `doc_id + sent_id/sent_idx`, so that sentence-level predictions can later be re-aggregated back into document-level outputs.

## Main points agreed on

### 1. Conversion unit should be sentence-level
Each document should be split into multiple sentence-level samples. A single `doc_id` should not correspond to only one training sample.

### 2. Alignment must be strict
The data conversion script should use a strict alignment check such as:

```python
sentence_text[start:end] == entity_text
```

This is intended to prevent silent offset drift and annotation mismatch during the doc-to-sentence conversion process.

### 3. W2NER supervision source
For NER supervision, the main usable sources should be:

- `sentences`
- `ann_mspan2dranges`
- `ann_mspan2guess_field`

The event-layer structure such as `recguid_eventname_eventdict_list` should **not** be treated as the primary NER supervision source.

### 4. W2NER capability boundary
The current W2NER implementation can handle:

- nested entities
- overlapping entities

But it does **not natively support multiple labels on the same span**, for example when the same date span is simultaneously both `startDate` and `endDate`.

### 5. Most stable near-term data strategy
A practical strategy is to prepare two versions of the data:

- **Audit version**: keep fine-grained role labels for manual checking and diagnosis
- **Training version**: fold labels before training, for example:
  - `startDate/endDate -> date`
  - `startTime/endTime -> time`

## About folded labels and how to recover them

A major conclusion was that **folding is usually not reversible**.

After folding `startDate/endDate -> date`, W2NER only learns that a span is a `date`; it no longer knows whether that date originally played the role of `startDate`, `endDate`, or both.

So the correct way to think about “recovery” in real scenarios is **not inverse decoding**, but a **second-stage role recovery step** after coarse NER.

### Recommended real-world pipeline

1. Raw text
2. Sentence splitting
3. W2NER extracts coarse spans such as `date`, `time`, `amount`, `company`
4. A role recovery module predicts finer roles, such as:
   - `startDate`
   - `endDate`
   - `both`
5. Results are aggregated back using `doc_id/sent_id`
6. Then document-level event extraction / slot filling is performed

### Two role recovery routes discussed

#### A. Rule-based baseline
Suitable for quickly building a working system.

Examples:
- “从 X 到 Y” → `X` more likely `startDate`, `Y` more likely `endDate`
- “自 X 起” → `X` tends to be `startDate`
- “截至 Y” → `Y` tends to be `endDate`

#### B. Two-stage classifier
A more robust mid-term solution.

Input may include:
- sentence text
- span position
- coarse type such as `date`
- trigger words or event type

Output may include:
- `startDate`
- `endDate`
- `both`
- `none`

### Important limitation
If the new data lacks event type, trigger words, slot definitions, or surrounding context, then in many cases a coarse `date` span **cannot be reliably recovered** into `startDate` vs `endDate`. This is an information insufficiency problem, not merely a model weakness.

## Evaluation of the W2NER training result

The user shared this test result:

```text
+--------+--------+-----------+--------+
| TEST 9 |   F1   | Precision | Recall |
+--------+--------+-----------+--------+
| Label  | 0.9827 |   0.9807  | 0.9852 |
| Entity | 0.3367 |   0.9767  | 0.2034 |
+--------+--------+-----------+--------+
```

The conclusion was that this result is **not good**, because the business-relevant metric is the **Entity** line:

- Entity Precision = 0.9767
- Entity Recall = 0.2034
- Entity F1 = 0.3367

This pattern strongly suggests:

- the model is **very conservative**
- it rarely predicts an entity unless highly confident
- predicted entities are usually correct
- but it misses most true entities

In practical terms:

> The model usually predicts correctly when it predicts something, but it fails to detect the majority of gold entities.

This is generally not usable for a real extraction pipeline, because later stages cannot recover entities that were never detected.

## Most likely causes discussed for low recall

Priority suspicions were:

### 1. Data conversion problems
Possible issues:
- positive instances were lost during doc-to-sentence conversion
- entity offsets drifted
- gold spans were filtered incorrectly
- training supervision became too sparse

### 2. Very sparse training distribution
Possible issues:
- too many negative sentences
- very imbalanced entity type distribution
- still too few effective positives even after folding

### 3. Overly conservative decode or inference settings
Possible issues:
- threshold too high
- pruning too aggressive
- long or nested entities not decoded well

### 4. Mismatch between training and evaluation conventions
Possible issues:
- folded labels used in training, fine-grained labels used in evaluation
- sentence-level training but problematic doc-level re-aggregation
- span boundary convention mismatch

## Discussion about `util.py` and the `decode` step

When asked whether the `decode` step in `util.py` might be problematic, the conclusion was:

- **possibly mismatched with the user's current data setup**
- but **not obviously the primary bug if it is the standard W2NER implementation**

The main view was that standard W2NER `decode` is usually a faithful recovery step from the predicted relation grid, not the most likely root cause of extremely low recall by itself.

### Why `decode` was not considered the first suspect
The decode logic mainly reconstructs entities from relation predictions such as:

- `NNW`
- `THW-*`

So if the model output grid is already sparse or conservative, decode will simply reflect that sparsity rather than fix it.

### Four high-priority decode-related checks

#### 1. `length` correctness
If `length` is shorter than the true tokenized sentence length, decode will directly truncate entities near the end of the sentence.

This can produce exactly the pattern seen in the results:
- high precision
- very low recall

#### 2. Label ID mapping consistency
The decode logic assumes a strict convention, especially that:
- one ID corresponds to `NNW`
- values greater than a threshold correspond to `THW-type`

If preprocessing changed the label mapping, decode may run without crashing while still interpreting relations incorrectly.

#### 3. Gold/prediction representation must match exactly
Evaluation can fail badly if gold entities and decoded predictions differ in:
- token index system
- sentence-local vs document-global offsets
- folded vs fine-grained types
- boundary conventions

#### 4. Grid direction convention
If the preprocessing code fills the grid in the opposite directional convention from what `decode` expects, recall can collapse even though the decode code itself looks normal.

## Recommended debugging order

The recommended next checks were:

1. Compare `pred` entity counts vs `gold` entity counts on the test set
2. Manually inspect around 20 examples:
   - original sentence
   - gold spans
   - predicted spans
3. Re-check the generated W2NER training data:
   - sentence text
   - entity start/end
   - entity text
   - folded label
4. Verify evaluation consistency:
   - same token indexing
   - same label granularity
   - same span boundary convention
5. Specifically for `decode`, inspect one sample with:
   - `instance`
   - `length`
   - `gold entities`
   - `decode_entities`

## Overall conclusion

The discussion converged on these main conclusions:

- Sentence-level conversion with strict alignment is essential.
- Folded labels are a practical baseline, but not reversible.
- Fine-grained role recovery should be handled by rules or a second-stage classifier.
- The reported W2NER result is **not good**, because entity recall is far too low.
- The low recall is more likely due to data conversion, label mapping, indexing mismatch, or overly sparse predictions than to an obvious bug in the standard `decode` logic alone.
