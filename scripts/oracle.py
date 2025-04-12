import pandas as pd
from google import genai
from google.genai import types
import argparse
import os
import re
from tqdm import tqdm
from dotenv import load_dotenv
from typing import Optional, Tuple, List, Dict, Any
import traceback

GEMINI_API_KEY_ENV = 'GEMINI_API_KEY'
DEFAULT_MODEL_NAME = 'gemini-2.5-pro-preview-03-25'

load_dotenv()
API_KEY = os.environ.get(GEMINI_API_KEY_ENV)
if not API_KEY:
  raise ValueError(f"Environment variable '{GEMINI_API_KEY_ENV}' not set.")

try:
  client = genai.Client(api_key=API_KEY)
except Exception as e:
  print(f"Error initializing Gemini client: {e}")
  exit(1)

def generate_reasoning_and_answer(
  question: str,
  choices_string: str,
  model_name: str = DEFAULT_MODEL_NAME
) -> Tuple[Optional[str], Optional[int]]:

  system_instruction = r'Ini adalah percakapan antara pengguna dan asisten. Pengguna akan menanyakan suatu pertanyaan berbentuk pilihan ganda dan asisten bertugas untuk menjawab pertanyaan tersebut. Asisten perlu menunjukkan proses berpikir terlebih dahulu, lalu menjawab pertanyaan tersebut. Cukup translasikan proses "thoughts" yang telah dilakukan dalam bahasa Inggris ke bahasa Indonesia. Proses berpikir diapit oleh <think> </think> dan jawaban diapit oleh <answer> </answer> tag, secara berurutan. Contoh: <think> {Proses berpikir} </think> <answer> A </answer>'

  prompt = f"""Ini adalah percakapan antara pengguna dan asisten. Pengguna akan menanyakan suatu pertanyaan berbentuk pilihan ganda dan asisten bertugas untuk menjawab pertanyaan tersebut. Asisten perlu menunjukkan proses berpikir terlebih dahulu, lalu menjawab pertanyaan tersebut. Cukup translasikan proses "thoughts" yang telah dilakukan dalam bahasa Inggris ke bahasa Indonesia. Proses berpikir diapit oleh <think> </think> dan jawaban diapit oleh <answer> </answer> tag, secara berurutan. Contoh: <think> {{Proses berpikir}} </think> <answer> A </answer>

Pertanyaan: {question}
Pilihan:
{choices_string}
"""

  try:
    response = client.models.generate_content(
      model=model_name,
      config=types.GenerateContentConfig(
        system_instruction=system_instruction
      ),
      contents=prompt,
    )

    response_text = response.text
    token_count = response.usage_metadata.total_token_count if response.usage_metadata else None
    return response_text, token_count

  except Exception as e:
    error_message = f"Error generating response: {e}"
    print(f"API Error for question: '{question[:50]}...' - {error_message}")
    return error_message, None

def extract_answer(reasoning_answer_text: str) -> Optional[str]:
  # Updated regex to handle both 'A' and '(A)' formats
  match = re.search(r'<answer>\s*(?:\()?([A-E])(?:\))?\s*</answer>', reasoning_answer_text)
  if match:
    return match.group(1)  # Return the captured letter
  else:
    if '<answer>' in reasoning_answer_text:
      print(f"Warning: Found '<answer>' tag but couldn't extract A-E or (A)-(E) answer from: {reasoning_answer_text}")
    return None

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
  reasoning_answers: List[Optional[str]] = []
  answers_only: List[Optional[str]] = []
  total_tokens: List[Optional[int]] = []

  for row in tqdm(df.itertuples(), total=len(df), desc="Processing Rows"):
    question: str = getattr(row, 'Question', '')
    choices_string: str = getattr(row, 'Options', '')

    if not question or not choices_string:
      print(f"Warning: Skipping row index {row.Index} due to missing Question or Options.")
      reasoning_answers.append("Skipped - Missing Data")
      answers_only.append(None)
      total_tokens.append(None)
      continue

    api_response_text, token_count = generate_reasoning_and_answer(question, choices_string)

    reasoning_answers.append(api_response_text)
    total_tokens.append(token_count)

    if api_response_text and not api_response_text.startswith("Error generating response:"):
      extracted_ans = extract_answer(api_response_text)
      answers_only.append(extracted_ans)
    else:
      answers_only.append(None)

  df_processed = df.copy()
  df_processed['Reasoning and Answer'] = reasoning_answers
  df_processed['Answer'] = answers_only
  df_processed['Total Tokens'] = total_tokens
  return df_processed

def main() -> None:
  parser = argparse.ArgumentParser(
    description="Process a CSV file containing multiple-choice questions to generate "
                "reasoning and answers using the Gemini API."
  )
  parser.add_argument("input_csv", help="Path to the input CSV file.")
  parser.add_argument(
    "-o", "--output_csv",
    help="Path to the output CSV file. If not provided, the input CSV will be overwritten.",
    default=None
  )

  args = parser.parse_args()

  try:
    print(f"Reading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    if 'Question' not in df.columns or 'Options' not in df.columns:
      print("Error: The CSV file must contain 'Question' and 'Options' columns.")
      return

    print("Processing questions with Gemini API...")
    processed_df = process_dataframe(df)

    output_path = args.output_csv if args.output_csv else args.input_csv
    print(f"Saving processed data to: {output_path}")
    processed_df.to_csv(output_path, index=False, encoding='utf-8')

    if args.output_csv:
      print(f"Processed data saved successfully.")
    else:
      print(f"Input CSV '{args.input_csv}' has been updated successfully.")

  except FileNotFoundError:
    print(f"Error: Input CSV file '{args.input_csv}' not found.")
  except pd.errors.EmptyDataError:
    print(f"Error: Input CSV file '{args.input_csv}' is empty.")
  except Exception as e:
    print(f"An unexpected error occurred during processing: {e}")
    traceback.print_exc()

if __name__ == "__main__":
  main()