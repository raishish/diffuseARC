import json
import os
import glob

# def compare_files(solution, output):
#     # Load the solution file
#     with open(solution, 'r') as f:
#         solution_data = json.load(f)
#         sol = solution_data['test'][0]['output']
    
#     # Load the output file
#     with open(output, 'r') as f:
#         output_data = json.load(f)
#         out = output_data
    
#     print(f"Comparing {os.path.basename(solution)} with {os.path.basename(output)}")
    
#     # Compare arrays if dimensions match
#     if len(sol) == len(out) and len(sol[0]) == len(out[0]):
#         print(f"Dimensions match: {len(sol)} x {len(sol[0])}")
        
#         match_count = 0
#         total_elements = len(sol) * len(sol[0])
        
#         for i in range(len(sol)):
#             for j in range(len(sol[0])):
#                 if sol[i][j] == out[i][j]:
#                     match_count += 1
        
#         match_percentage = (match_count / total_elements) * 100
#         print(f"Match percentage: {match_percentage:.2f}%")
#         print("-" * 50)
#         return match_percentage
#     else:
#         print(f"Dimensions don't match: Solution {len(sol)} x {len(sol[0])}, Output {len(out)} x {len(out[0])}")
#         print("-" * 50)
#         return 0

def compare_files(solution, output):
    # Load the solution file
    with open(solution, 'r') as f:
        solution_data = json.load(f)
        sol = solution_data['test'][0]['output']
    
    # Load the output file
    with open(output, 'r') as f:
        output_data = json.load(f)
        out = output_data
    
    print(f"Comparing {os.path.basename(solution)} with {os.path.basename(output)}")
    
    # Check outer dimensions
    if len(sol) != len(out):
        print(f"Dimensions don't match: Solution {len(sol)} rows, Output {len(out)} rows")
        print("-" * 50)
        return 0
    
    # Check inner dimensions of all rows before comparison
    for i in range(len(sol)):
        if i >= len(out) or len(sol[i]) != len(out[i]):
            print(f"Dimensions don't match: row {i} has different lengths")
            print("-" * 50)
            return 0
    
    # If we get here, all dimensions match
    print(f"Dimensions match: {len(sol)} x {len(sol[0])}")
    
    match_count = 0
    total_elements = len(sol) * len(sol[0])
    
    for i in range(len(sol)):
        for j in range(len(sol[0])):
            if sol[i][j] == out[i][j]:
                match_count += 1
    
    match_percentage = (match_count / total_elements) * 100
    print(f"Match percentage: {match_percentage:.2f}%")
    print("-" * 50)
    return match_percentage

def main():
    # Get all solution files
    solution_files = glob.glob('data/evaluation/*.json')
    output_files = glob.glob('data/cot-out/*.json')
    
    results = []
    
    # Compare each solution file with its corresponding output file
    for solution_file in solution_files:
        solution_basename = os.path.basename(solution_file)
        
        # Find the matching output file
        for output_file in output_files:
            if os.path.basename(output_file) == solution_basename:
                match_percentage = compare_files(solution_file, output_file)
                results.append({
                    'solution_file': solution_basename,
                    'match_percentage': match_percentage
                })
                break
    
    # Display summary of results
    print("\nSummary of Results:")
    print("=" * 50)
    for result in results:
        print(f"{result['solution_file']}: {result['match_percentage']:.2f}%")
    
    # Calculate average match percentage
    if results:
        avg_match = sum(result['match_percentage'] for result in results) / len(results)
        print(f"\nAverage match percentage: {avg_match:.2f}%")

if __name__ == "__main__":
    main()