# suppress warnings
import warnings

warnings.filterwarnings("ignore")

# import libraries
import argparse
from together import Together
import textwrap
import pandas as pd
from datetime import datetime
import os
from pathlib import Path


## FUNCTION 1: This Allows Us to Prompt the AI MODEL
# -------------------------------------------------
def prompt_llm(prompt, with_linebreak=False):
    # This function allows us to prompt an LLM via the Together API

    # model
    model = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"

    # Make the API call
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    output = response.choices[0].message.content

    if with_linebreak:
        # Wrap the output
        wrapped_output = textwrap.fill(output, width=50)

        return wrapped_output
    else:
        return output


def summarize_data(data_path, n_rows=1000, output_dir="results"):
    """
    Reads the first n_rows of a CSV file, prepares a text summary, sends it to the LLM for summarization, and saves the LLM's summary to output_dir/dataset_summarize_{timestamp}.txt
    """
    try:
        df = pd.read_csv(data_path, nrows=n_rows)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    # Prepare a text summary for the LLM
    summary_text = []
    summary_text.append(f"File: {data_path}")
    summary_text.append(f"Shape (sample): {df.shape}")
    summary_text.append(f"Columns: {list(df.columns)}")
    summary_text.append(f"\nData types:\n{df.dtypes}")
    summary_text.append(f"\nFirst 3 rows:\n{df.head(3).to_string(index=False)}")
    summary_text.append(f"\nBasic statistics:\n{df.describe(include='all').transpose().head(5).to_string()}\n")
    summary_for_llm = '\n'.join([str(x) for x in summary_text])

    # Prompt the LLM to summarize
    prompt = f"""
    Please provide a concise, human-friendly summary of the following dataset sample. Highlight the main characteristics, data types, and any interesting patterns you notice.\n\n{summary_for_llm}
    """
    llm_summary = prompt_llm(prompt)

    # Save the LLM's summary
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"{output_dir}/dataset_summarize_{timestamp}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(llm_summary)
    print(f"Summary saved to {output_file}")
    return output_file


def chat_with_scientist(data_path, n_rows=1000):
    """
    Interactive chat with the Data Scientist. The assistant can suggest analyses, models, and graphs for the dataset.
    """
    try:
        df = pd.read_csv(data_path, nrows=n_rows)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Prepare context for the assistant
    context = f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns. Columns: {list(df.columns)}. Data types: {df.dtypes.to_dict()}"
    print("\nWelcome to the DS Assistant Chat! Type your questions or type 'exit' to quit.")
    print(f"\nContext: {context}\n")
    print("You can ask for ideas, analysis, modeling, or visualization suggestions.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        prompt = (
            "You are a helpful Data Science assistant. "
            "Given the following dataset context, suggest ideas for analysis, modeling, and visualization, or answer the user's question. "
            f"\n\nContext: {context}\nUser: {user_input}\nAssistant:"
        )
        response = prompt_llm(prompt)
        print(f"Assistant: {response}")


if __name__ == "__main__":
    # args on which to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--api_key", type=str, default=None)
    args = parser.parse_args()

    # Get Client for your LLMs
    client = Together(api_key=args.api_key)


    # Generate concept summaries
    print("Generating concept summaries...")
    summaries = summarize_data("data/ai_job_dataset.csv", n_rows=1000, output_dir="results")

    # Save results to timestamped file (already done in summarize_data, so just use the returned path)
    output_file = summaries  # summaries is the output file path

    # Display results
    print("\nGenerated Summaries:\n")
    with open(output_file, 'r', encoding='utf-8') as f:
        print(f.read())
    print("-" * 100)

    # Start interactive chat
    chat_with_scientist("data/ai_job_dataset.csv", n_rows=1000)