import os
import json
import argparse
from dotenv import load_dotenv
from typing import List, Dict
import litellm

# Load environment variables
load_dotenv()

# Supported model providers and their keys
MODEL_PROVIDERS = {
    "gemini": "gemini/gemini-2.5-flash",
    "mistral": "mistral/mistral-large-latest",
    # Add more providers/models as needed
}

def extract_pci_gdpr_info(text: str, model_provider: str) -> List[Dict[str, str]]:
    """
    Extracts PCI/GDPR info from text using the specified model.
    Returns a list of dictionaries with 'type' and 'text' fields.
    """
    model = MODEL_PROVIDERS.get(model_provider.lower())
    if not model:
        raise ValueError(f"Unsupported model provider: {model_provider}")

    # Prompt for extraction
    prompt = f"""
    You are an expert at GDPR and PCI compliance
    Extract the following types of information from the text below:
    - Address
    - Account Number
    - Name
    - Birthday
    - Government Identification
    - Other

    Return the results as a JSON array with objects containing 'type' and 'text' fields.
    Only include information that is explicitly present in the text.  If the information 
    that needs to be flagged doesn't fit into a category use other.  Do not include text
    does meet GDPR or PCI compliance

    Text:
    {text}
    """

    # Call the LLM
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    # Parse the response
    try:
        data = json.loads(response.choices[0].message.content)
        if isinstance(data, list):
            return data
        else:
            return []
    except Exception as e:
        print(f"Error parsing response: {e}")
        return []

def read_input_file(file_path: str) -> str:
    """Reads the input text file."""
    with open(file_path, "r") as f:
        return f.read()

def write_output_file(data: List[Dict[str, str]], output_path: str):
    """Writes the extracted data to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Extract PCI/GDPR info from a text file.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input text file.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_PROVIDERS.keys(), help="Model provider to use.")
    args = parser.parse_args()

    # Read input
    text = read_input_file(args.input)

    # Extract info
    extracted_data = extract_pci_gdpr_info(text, args.model)

    # Write output
    write_output_file(extracted_data, args.output)
    print(f"Extracted data written to {args.output}")

if __name__ == "__main__":
    main()

