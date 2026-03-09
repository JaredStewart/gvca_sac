# Deck Generation

Auto-generate PPTX presentations from survey data. Reads a template PPTX, replaces chart images and data-driven text with fresh content, and marks narrative slides with a "NEEDS UPDATING" banner.

## Prerequisites

1. **Template PPTX** at `data/templates/presentation.pptx`
2. **Survey data loaded** — CSV files in `data/` for all relevant years
3. **PocketBase running** — free responses and tagging results available
4. **Charts dependencies installed** — `python-pptx`, `matplotlib` (already in `pyproject.toml`)

## Usage

### CLI

```bash
# Basic — auto-detects all available years for YoY charts
gvca generate-deck 2026

# Specify template path
gvca generate-deck 2026 --template path/to/template.pptx

# Specify years for comparison
gvca generate-deck 2026 --years 2023,2024,2025,2026
```

Output: `artifacts/2026/presentation_2026.pptx`

### API

```bash
curl -X POST http://localhost:8000/api/deck/generate \
  -H "Content-Type: application/json" \
  -d '{"year": "2026"}' \
  -o presentation_2026.pptx
```

Request body:
```json
{
  "year": "2026",
  "years": ["2023", "2024", "2025", "2026"],
  "template_path": null
}
```

## Slide Classification

### Data-Driven Slides (auto-populated)

| Slides | Content | What Gets Replaced |
|--------|---------|---------------------|
| 7-8 | Survey participation | Table values |
| 11 | Rank Question Results Overview | Chart image |
| 13 | Composite scores | Table values |
| 15 | YoY composite satisfaction | Chart image |
| 16-22 | Q3-Q9 per-question | 2 chart images + text annotations |
| 26 | Good Choice vs Better Serve | Chart image + count text |
| 27 | Tag frequency | Chart image |
| 28-36 | Per-tag category pages | Chart images + response frequency |
| 47 | Summary stats | Satisfaction rate percentages |

### Narrative Slides (banner overlay)

| Slides | Content |
|--------|---------|
| 1-6 | Title, exec summary, team, confidentiality, questions |
| 9-10, 23 | Section dividers / chart legend |
| 12 | Subgroup observations |
| 14 | Grade level observations |
| 24-25 | Open response process, nature of responses |
| 37-46 | Key findings, per-grade feedback, strengths, focus areas |
| 48-49 | Future plans, closing |

## Template Management

The template PPTX lives at `data/templates/presentation.pptx` (gitignored). To set it up:

```bash
cp "local data files/presentation.pptx" data/templates/presentation.pptx
```

When updating the template:
- Data-driven slides must maintain their image shapes in the expected positions
- Images are matched by left-to-right order on each slide
- Text replacements match against known text patterns (e.g., "Response Frequency:", year labels)

## Adding a New Chart Slide

1. Add the slide number and chart specs to `CHART_SLIDES` in `backend/app/core/deck_generator.py`
2. Format: `slide_num: [(image_index, "chart_function_name", {kwargs})]`
3. Implement or reuse a chart function from `backend/app/core/charts.py`
4. If the slide has data-driven text, add a case to `_update_slide_text()`

## Troubleshooting

- **"Template not found"**: Copy the template to `data/templates/presentation.pptx`
- **Missing charts**: Check that PocketBase is running and has tagging results for the year
- **Image sizing issues**: Charts are placed at the exact position/size of the original image shape
- **Text not updating**: Text replacement matches exact strings; check the template for the expected text patterns
