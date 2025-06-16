# suppress warnings
import warnings

warnings.filterwarnings("ignore")

# import libraries
import argparse
from together import Together
import textwrap
import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path

def perform_eda(data, file_path):
    """
    Perform Exploratory Data Analysis on the dataset
    Returns a dictionary with various EDA metrics
    """
    eda_results = {}
    
    # Basic information
    eda_results['file_name'] = os.path.basename(file_path)
    eda_results['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    eda_results['total_rows'] = len(data)
    eda_results['total_columns'] = len(data.columns)
    
    # Column information
    column_info = {}
    for col in data.columns:
        col_info = {}
        col_info['dtype'] = str(data[col].dtype)
        col_info['missing_values'] = data[col].isnull().sum()
        col_info['missing_percentage'] = (data[col].isnull().sum() / len(data)) * 100
        
        if pd.api.types.is_numeric_dtype(data[col]):
            col_info['mean'] = data[col].mean()
            col_info['std'] = data[col].std()
            col_info['min'] = data[col].min()
            col_info['max'] = data[col].max()
            col_info['unique_values'] = data[col].nunique()
        else:
            col_info['unique_values'] = data[col].nunique()
            col_info['most_common'] = data[col].mode().iloc[0] if not data[col].mode().empty else None
            col_info['most_common_count'] = data[col].value_counts().iloc[0] if not data[col].value_counts().empty else 0
        
        column_info[col] = col_info
    
    eda_results['column_info'] = column_info
    
    # Correlation matrix for numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        eda_results['correlation_matrix'] = data[numeric_cols].corr().to_dict()
    
    return eda_results

def save_eda_results(eda_results, output_dir='results'):
    """
    Save EDA results to a markdown file
    """
    # Create results directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'eda_summary_{timestamp}.md')
    
    # Start building markdown content
    md_content = f"# Dataset Analysis Summary\n*Analysis performed on: {eda_results['timestamp']}*\n\n"
    
    # Dataset Overview
    md_content += "## Dataset Overview\n"
    md_content += f"- **Total Rows:** {eda_results['total_rows']}\n"
    md_content += f"- **Total Columns:** {eda_results['total_columns']}\n\n"
    
    # Column Analysis
    md_content += "## Column Analysis\n\n"
    
    for col, info in eda_results['column_info'].items():
        md_content += f"### {col}\n"
        md_content += f"- **Type:** {info['dtype']}\n"
        md_content += f"- **Missing Values:** {info['missing_percentage']:.1f}%\n"
        md_content += f"- **Unique Values:** {info['unique_values']}\n"
        
        if 'mean' in info:
            md_content += "- **Statistics:**\n"
            md_content += f"  - Mean: {info['mean']:.2f}\n"
            md_content += f"  - Std Dev: {info['std']:.2f}\n"
            md_content += f"  - Min: {info['min']}\n"
            md_content += f"  - Max: {info['max']}\n"
        else:
            if info['most_common'] is not None:
                md_content += f"- **Most Common:** {info['most_common']} ({info['most_common_count']} occurrences)\n"
        
        md_content += "\n"
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    return output_file

def load_data(file_path):
    """
    Load data from file (supports CSV and text files)
    """
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        else:
            # Try to read as CSV first, if that fails, read as text
            try:
                return pd.read_csv(file_path)
            except:
                return pd.read_csv(file_path, sep='\t')
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return None

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

def save_summary(summary, output_dir='results'):
    """
    Save the dataset summary to a text file
    """
    # Create results directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'dataset_summary_{timestamp}.txt')
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    return output_file

if __name__ == "__main__":
    # args on which to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--api_key", type=str, default=None)
    parser.add_argument("-f", "--file", type=str, required=True, help="Path to the data file")
    args = parser.parse_args()

    # Get Client for your LLMs
    client = Together(api_key=args.api_key)

    # Load and analyze data
    data = load_data(args.file)
    if data is not None:
        print(f"\nAnalyzing file: {args.file}")
        eda_results = perform_eda(data, args.file)
        output_file = save_eda_results(eda_results)
        print(f"\nEDA results saved to: {output_file}")
        
        # Generate summary using LLM
        summary_prompt = f"""
        Please analyze this dataset and provide a brief summary:
        - File: {eda_results['file_name']}
        - Rows: {eda_results['total_rows']}
        - Columns: {eda_results['total_columns']}
        - Column types: {', '.join([f"{col} ({info['dtype']})" for col, info in eda_results['column_info'].items()])}
        """
        summary = prompt_llm(summary_prompt)
        print("\nDataset Summary:")
        print(summary)
        
        # Save summary to text file
        summary_file = save_summary(summary)
        print(f"Dataset summary saved to: {summary_file}")
    else:
        print(f"Failed to load file: {args.file}")
    
    print("-" * 100)