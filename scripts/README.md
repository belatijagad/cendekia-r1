# Scripts

## PDF Parser

Parse the pdf into markdown using Mistral's OCR, then saves it into `csv` format. Example usage:

```bash
python3 ./scripts/parse_pdf.py -p ./data/raw/soal.pdf -o ./data/processed/dataset.csv
```

## Oracle

```bash
python scripts/oracle.py ./data/processed/SIMAK2018_MM.csv -o ./data/done/SIMAK2018_MM.csv
```