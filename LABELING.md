# Correcting the draft outlines

Visit 1 leaves 32 crops (4 per real image) in `labels\train`, `labels\val`, `labels\test`, each with a draft
outline file from the best model. Your corrections become the ground truth that visit 2 trains on (train),
picks the training length on (val), and reports accuracy on (test, never trained on).

## Start

Open a terminal in `%USERPROFILE%\sem-powder` and run

```
run.cmd label
```

The Cellpose window opens on the first crop not yet reviewed, with the draft outlines loaded.
`run.cmd status` shows how many are done. Works on any Windows PC with the repo set up (no GPU needed).

## Controls (Cellpose GUI)

| Action | How |
|---|---|
| Delete a wrong outline | **Ctrl + left-click** on it |
| Draw a missing particle | **right-click**, trace the edge, right-click again (or return to the start) |
| Merge two pieces of one particle | **Alt + left-click** on each |
| Undo | Ctrl+Z (draw), Ctrl+Y (delete) |
| Zoom / pan | scroll wheel / left-drag; double-click = full view |
| Outlines on/off | X (masks), Z (outlines) |
| **Save** | **Ctrl+S** (do this on every crop, even if you changed nothing: saving marks it reviewed) |
| Next / previous crop | D / A |

## Rules (consistency matters more than perfection)

1. **One outline per particle you can see**, including particles partly hidden behind others (outline the visible part only).
2. **Satellites get their own outline** when you can see where they meet the larger particle. Same for any small particle touching a large one.
3. **Dense clumps of fines** where you cannot tell individual particles apart: one outline around the clump.
4. **Particles cut by the crop edge**: outline the visible part. They are excluded from statistics automatically.
5. **No outline** for substrate texture, dust, charging streaks, or shadows.
6. Edges within ~2 pixels are good enough. Do not spend time perfecting smooth outlines.
7. If the draft is already right, just press Ctrl+S and move on.

## Order and time

Train crops first (16), then val (8), then test (8). Expect ~5-15 minutes per crop. Visit 2 needs at least
4 reviewed train crops, but the result is only as good as the labels: do all 32 if you can.
Label the test crops as carefully as the others; they are the accuracy number you will quote.

## Your own images

Put extra images (e.g. CAMAL) in `data\label_extra\` and run `run.cmd drafts`:
they get draft crops too (existing crops and your corrections are never overwritten), in the train split unless listed in `configs\label_split.json`, e.g.
`{"camal_run12_img3": "test"}`.
