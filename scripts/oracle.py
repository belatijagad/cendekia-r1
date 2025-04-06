import pandas as pd
from google import genai
from google.genai import types
import argparse
import os
import re
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))

def generate_reasoning_and_answer(question, choices_string):
  prompt = f"""Ini adalah percakapan antara pengguna dan asisten. Pengguna akan menanyakan suatu pertanyaan berbentuk pilihan ganda dan asisten bertugas untuk menjawab pertanyaan tersebut. Asisten perlu menunjukkan proses berpikir terlebih dahulu, lalu menjawab pertanyaan tersebut. Cukup translasikan proses ``thoughts" yang telah dilakukan dalam bahasa Inggris ke bahasa Indonesia. Proses berpikir diapit oleh <think> </think> dan jawaban diapit oleh <answer> </answer> tag, secara berurutan. Pastikan jawaban akhir berupa \\boxed{{huruf pilihan}}.

Pertanyaan: {question}
Pilihan:
{choices_string}
"""
  try:
    response = client.models.generate_content(
      model='gemini-2.5-pro-preview-03-25',
      config=types.GenerateContentConfig(
        system_instruction=r'Ini adalah percakapan antara pengguna dan asisten. Pengguna akan menanyakan suatu pertanyaan berbentuk pilihan ganda dan asisten bertugas untuk menjawab pertanyaan tersebut. Asisten perlu menunjukkan proses berpikir terlebih dahulu, lalu menjawab pertanyaan tersebut. Cukup translasikan proses ``thoughts" yang telah dilakukan dalam bahasa Inggris ke bahasa Indonesia. Proses berpikir diapit oleh <think> </think> dan jawaban diapit oleh <answer> </answer> tag, secara berurutan. Pastikan jawaban akhir berupa \boxed{huruf pilihan}}. Contoh: <think> {Proses berpikir} </think> <answer> {Jawaban} \boxed{huruf pilihan} </answer>'
      ),
      contents=prompt,
    )
    return response
  except Exception as e:
    return f"Error generating response: {e}"

def extract_answer(reasoning_answer):
  match = re.search(r'\\boxed\{([A-E])\}', reasoning_answer)
  if match:
    return match.group(1)
  else:
    return None

def process_dataframe(df):
  reasoning_answers = []
  answers_only = []
  total_tokens = []
  for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Rows"):
    question = row['Question']
    choices_string = row['Options']
    response = generate_reasoning_and_answer(question, choices_string)
    if isinstance(response, types.GenerateContentResponse):
      reasoning_answer = response.text.replace('\n', '\\n')
      reasoning_answers.append(reasoning_answer)
      answers_only.append(extract_answer(reasoning_answer))
      total_tokens.append(response.usage_metadata.total_token_count)
    else:
      reasoning_answers.append(str(response).replace('\n', '\\n'))
      answers_only.append(None)
      total_tokens.append(None)
  df['Reasoning and Answer'] = reasoning_answers
  df['Answer'] = answers_only
  df['Total Tokens'] = total_tokens
  return df

def main():
  parser = argparse.ArgumentParser(description="Process a CSV file containing multiple-choice Question to generate reasoning and answers using Gemini.")
  parser.add_argument("input_csv", help="Path to the input CSV file.")
  parser.add_argument("-o", "--output_csv", help="Path to the output CSV file. If not provided, the input CSV will be overwritten.", default=None)

  args = parser.parse_args()

  try:
    df = pd.read_csv(args.input_csv)
    if 'Question' not in df.columns or 'Options' not in df.columns:
      print("Error: The CSV file must contain 'Question' and 'Options' columns.")
      return

    processed_df = process_dataframe(df.copy())

    if args.output_csv:
      processed_df.to_csv(args.output_csv, index=False)
      print(f"Processed data saved to {args.output_csv}")
    else:
      processed_df.to_csv(args.input_csv, index=False)
      print(f"Input CSV '{args.input_csv}' has been updated with 'Reasoning and Answer', 'Answer', and 'Total Tokens' columns.")

  except FileNotFoundError:
    print(f"Error: Input CSV file '{args.input_csv}' not found.")
  except Exception as e:
    print(f"An error occurred: {e}")

if __name__ == "__main__":
  main()