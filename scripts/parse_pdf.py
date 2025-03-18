from mistralai import Mistral
import os
import re
import json
import csv
import argparse

def extract_pdf(pdf_path):
  API_KEY = os.environ.get('MISTRAL_API_KEY')
  if not API_KEY: raise ValueError("MISTRAL_API_KEY environment variable not set")
  
  client = Mistral(api_key=API_KEY)

  uploaded_pdf = client.files.upload(
    file={'file_name': os.path.basename(pdf_path), 'content': open(pdf_path, 'rb'),},
    purpose='ocr'
  )

  signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

  ocr_response = client.ocr.process(
    model='mistral-ocr-latest',
    document={'type': 'document_url', 'document_url': signed_url.url,}
  )
  full_text = ''
  for page in ocr_response.pages: full_text += page.markdown   
  return full_text

def extract_metadata(text):
  metadata = {}
  lines = text.split('\n')
  for line in lines[:10]:
    if line.startswith(':'):
      parts = line.split(':', 1)
      if len(parts) > 1:
        key = parts[0].strip()
        value = parts[1].strip()
        metadata[key if key else 'subject'] = value
  return metadata

def extract_sections_info(text):
  sections_info = {}
  pattern = r'Mata Ujian ([A-Z\s]+) nomor (\d+) sampai nomor (\d+)'
  for match in re.finditer(pattern, text):
    section_name = match.group(1).strip()
    start_num = int(match.group(2))
    end_num = int(match.group(3))
    sections_info[section_name] = {'start': start_num, 'end': end_num}
  return sections_info

def extract_questions(text):
  questions = []
  pattern = r'(\d+)\.\s+(.*?)(?=(?:\n\d+\.)|$)'
  positions = []
  raw_questions = []
  
  for match in re.finditer(pattern, text, re.DOTALL):
    q_num = int(match.group(1))
    q_text = match.group(2).strip()
    start_pos = match.start()
    
    prefix = text[max(0, start_pos-20):start_pos]
    if 'nomor' in prefix or 'soal' in prefix or q_num > 100: continue
        
    raw_questions.append((q_num, q_text))
    positions.append(start_pos)
  
  for i, (q_num, q_text) in enumerate(raw_questions):
    options = {}
    option_pattern = r'\(([A-E])\)(.*?)(?=\n\([A-E]\)|$)'
    
    for option_match in re.finditer(option_pattern, q_text, re.DOTALL):
      option_letter = option_match.group(1)
      option_text = option_match.group(2).strip()
      options[option_letter] = option_text
    
    question_text = q_text
    opt_pattern = r'\([A-E]\).*?(?=\n\([A-E]\)|$)'
    for opt_match in re.finditer(opt_pattern, q_text, re.DOTALL):
      question_text = question_text.replace(opt_match.group(0), '')
    
    questions.append({
      'id': q_num,
      'question_text': question_text.strip(),
      'options': options,
      'position': positions[i]
    })
  
  questions.sort(key=lambda q: q['id'])
  return questions

def associate_sections(questions, sections_info):
  for question in questions:
    q_id = question['id']
    section = None
    for section_name, section_range in sections_info.items():
      if section_range['start'] <= q_id <= section_range['end']:
        section = section_name
        break
    question['section'] = section
  return questions

def clean_output(questions):
  cleaned = []
  for q in questions:
    cleaned_q = {
      'id': q['id'],
      'section': q.get('section'),
      'question_text': q['question_text'],
      'options': q['options']
    }
    cleaned.append(cleaned_q)
  return cleaned

def process_markdown_exam(text):
  metadata = extract_metadata(text)
  sections_info = extract_sections_info(text)
  questions = extract_questions(text)
  questions = associate_sections(questions, sections_info)
  cleaned_questions = clean_output(questions)
  return {
    'metadata': metadata,
    'sections': sections_info,
    'questions': cleaned_questions
  }

def export_to_json(dataset, output_file='exam_dataset.json'):
  with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)
  return output_file

def export_to_csv(dataset, output_file='exam_dataset.csv'):
  with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ID', 'Section', 'Question', 'Options'])
    for q in dataset['questions']:
      options_text = ' '.join([f"({key}) {value}" for key, value in sorted(q['options'].items())])
      writer.writerow([
        q['id'],
        q['section'],
        q['question_text'],
        options_text
      ])
  return output_file

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Process exam PDF into structured dataset')
  parser.add_argument('--pdf', '-p', required=True, help='Path to the PDF file')
  parser.add_argument('--output', '-o', default='exam_dataset.csv', help='Output CSV filename')
  parser.add_argument('--json', '-j', default=None, help='Output JSON filename (optional)')
  parser.add_argument('--skip-ocr', '-s', action='store_true', help='Skip OCR and use existing markdown file')
  
  args = parser.parse_args()
  
  print(f"Processing exam from: {args.pdf}")
  
  try:
    if args.skip_ocr:
      with open(args.pdf, 'r', encoding='utf-8') as f:
        markdown_text = f.read()
    else:
      markdown_text = extract_pdf(args.pdf)
      markdown_path = f"{os.path.splitext(args.pdf)[0]}_extracted.md"
      with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(markdown_text)
      print(f"Extracted markdown saved to: {markdown_path}")
    
    if '\\n' in markdown_text: markdown_text = markdown_text.replace('\\n', '\n')
    
    dataset = process_markdown_exam(markdown_text)
    csv_file = export_to_csv(dataset, args.output)
    print(f"CSV dataset exported to: {csv_file}")
    
    if args.json:
      json_file = export_to_json(dataset, args.json)
      print(f"JSON dataset exported to: {json_file}")
                  
  except Exception as e:
    print(f"Error processing exam: {str(e)}")