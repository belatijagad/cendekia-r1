# Scripts

## PDF Parser

Parse the pdf into markdown using Mistral's OCR, then saves it into `csv` format. Example usage:

```bash
python ./scripts/parse_pdf.py -p ./data/raw/SIMAK2014_MD.pdf -o ./data/processed/SIMAK2014_MD.md --format md
python ./scripts/parse_pdf.py --markdown ./data/processed/SIMAK2014_MD.md --format csv
```

## Oracle

```bash
python scripts/oracle.py ./data/processed/SIMAK2018_MM.csv -o ./data/done/SIMAK2018_MM.csv
```