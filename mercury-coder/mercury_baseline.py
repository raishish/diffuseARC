import json
import requests
import os
import glob
import time
from dotenv import load_dotenv


def solve_arc_task(file_path, api_key, output_folder=None):
  
    # Load data from JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    train = data['train']
    test = data['test']
    test_input = test[0]['input']
    
    # Create examples string
    examples = ""
    for item in train:
        input_json = json.dumps(item['input'])
        output_json = json.dumps(item['output'])
        examples += f"Input: {input_json}\nOutput: {output_json}\n\n"
    
    # Create prompt
    prompt = f"""You are given a set of input-output grid pairs that define a transformation rule. Each grid is a 2D array of integers, where each integer represents a color. Your task is to learn the transformation and apply it to a new input grid.
    
    Each example below consists of an 'input' grid and its corresponding 'output' grid. Analyze the patterns and apply the inferred transformation rule to the final test input.
    
    You must determine the correct size of the output grid based on the transformation logic. However, the output grid must not exceed 30 rows or 30 columns in size.
    
    Training Examples:
    {examples}
    Test Input:
    {test_input}

What should be the Output grid? Only provide the output grid as your answer."""
    
    # Make API request
    response = requests.post(
        'https://api.inceptionlabs.ai/v1/chat/completions',
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        },
        json={
            'model': 'mercury-coder-small',
            'messages': [
                {'role': 'user', 'content': prompt}
            ]
            # 'max_tokens': 16384
        }
    )
    
    # Parse the response
    result = response.json()
    try:
        output_data = json.loads(result['choices'][0]['message']['content'])
        
        # Create output filename based on input filename
        base_name = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(base_name)[0]
        output_file_name = f"{file_name_without_ext}.json"
        
        if output_folder is None:
            output_folder = os.path.dirname(file_path)
        
        os.makedirs(output_folder, exist_ok=True)
        
        output_file_path = os.path.join(output_folder, output_file_name)
        
        with open(output_file_path, 'w') as outfile:
            json.dump(output_data, outfile, indent=2)
        
        print(f"Solution saved to {output_file_path}")
        return output_data
    
    except (KeyError, json.JSONDecodeError) as e:
        error_result = {'error': str(e), 'raw_response': result}
        print(f"Error: {e}")
        return error_result


def process_all_files(input_folder, api_key, output_folder=None):
  
    # Find all JSON files in the input folder
    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_folder}")
        return {}
    
    # Create output folder if it doesn't exist
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    # Filter files that don't have corresponding outputs
    files_to_process = []
    for file_path in json_files:
        base_name = os.path.basename(file_path)
        output_file_name = base_name  # Same name in output folder
        output_file_path = os.path.join(output_folder or input_folder, output_file_name)
        
        if not os.path.exists(output_file_path):
            files_to_process.append(file_path)
    
    if not files_to_process:
        print("All files have already been processed")
        return {}
    
    results = {}
    for i, file_path in enumerate(files_to_process):
        print(f"Processing file {i+1}/{len(files_to_process)}: {file_path}")
        try:
            result = solve_arc_task(file_path, api_key, output_folder)
            results[file_path] = result
            
            # Add 5 second delay after every 10 requests
            if (i + 1) % 10 == 0 and i < len(files_to_process) - 1:
                print("Waiting 5 seconds before continuing...")
                time.sleep(5)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            results[file_path] = {'error': str(e)}
    
    print(f"Processed {len(files_to_process)} files. Successfully completed: {len([r for r in results.values() if 'error' not in r])}")
    return results


def main():

    load_dotenv()
    API_KEY = os.getenv('API_KEY')

    process_all_files('data/eval', API_KEY, 'data/eval_output')


if __name__ == "__main__":
    main()