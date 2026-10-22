# Fig.4: predicted planning responses

`fig4.py` renders the cooling-only and joint NDVI/POP planning responses for
Beijing, Paris, New York, Johannesburg, and Sao Paulo. Install the repository
requirements and run commands from the repository root.

The renderer reads a schema-v1 `planning_data.pkl` containing both `joint`
and `cooling_only` branches. See the [planning experiment](../../experiments/planning_exp/EXPERIMENT.md)
for optimization, checkpoint, and processed-data requirements.

When the two branches are saved in separate packages, assemble them with:

```bash
python main_figs_v2/fig4/prepare_data.py
```

`--joint`, `--cooling-only`, and `--output-dir` select other input packages
and the destination. Assembly checks that the city IDs, year, original
NDVI/LST arrays, spatial masks, and preprocessing statistics match. It writes
`planning_data.pkl`, `comparable_metrics.csv`, `scenario_summary.csv`, and
`manifest.json` with the input paths and checksums.

Render the figure:

```bash
python main_figs_v2/fig4/fig4.py --input-dir main_figs_v2/fig4 \
  --output main_figs_v2/fig4/fig4.png
```

For an existing package produced with planning `--mode both`, pass its
directory directly to `fig4.py --input-dir`.

The CSV uses the renderer's raster transformations and masks. LST and NDVI
changes are city means over valid pixels; negative LST change denotes cooling.
POP percentages use the original city population as the denominator. Traffic
reduction is `100 * (original - adjusted) / original` for the joint branch.
Scenario summaries weight each city equally. All responses are model
predictions; the planning documentation states their interpretation.
