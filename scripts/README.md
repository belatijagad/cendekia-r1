# Scripts

## PDF Parser

Parse the pdf into markdown using Mistral's OCR, then saves it into `csv` format. Example usage:

```
python ./scripts/parse_pdf.py -p ./data/raw/SIMAK2018_MD.pdf -o ./data/processed/SIMAK2018_MD.md --format md
```

```
python ./scripts/parse_pdf.py --markdown ./data/processed/SIMAK2018_MD.md --format csv
```