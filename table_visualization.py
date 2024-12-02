import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the log directory
log_dir = r"C:\Users\garyz\Desktop\Deep Learning\GroupProjects\layer\layer_skip\logs"
# Define the directory to save the heatmap images
picture_dir = r"C:\Users\garyz\Desktop\Deep Learning\GroupProjects\layer\layer_skip\Picture"

# Ensure the picture directory exists
os.makedirs(picture_dir, exist_ok=True)

# Define the list of files you want to process
json_files_to_process = [
    "results_table5_facebook_layerskip-codellama-7B_human_eval_self_speculative.json",
    "results_table3_facebook_layerskip-llama2-13B_cnn_dm_summarization_autoregressive.json",
    "results_table3_facebook_layerskip-llama2-13B_cnn_dm_summarization_autoregressive_with_early_exit.json",
    "results_table3_facebook_layerskip-llama2-13B_cnn_dm_summarization_self_speculative.json",
    "results_table3_facebook_layerskip-llama2-13B_human_eval_autoregressive.json",
    "results_table3_facebook_layerskip-llama2-13B_human_eval_autoregressive_with_early_exit.json",
    "results_table3_facebook_layerskip-llama2-13B_human_eval_self_speculative.json",
    "results_table3_facebook_layerskip-llama2-13B_xsum_summarization_autoregressive.json",
    "results_table3_facebook_layerskip-llama2-13B_xsum_summarization_autoregressive_with_early_exit.json",
    "results_table3_facebook_layerskip-llama2-13B_xsum_summarization_self_speculative.json",
    "results_table5_facebook_layerskip-codellama-7B_human_eval_autoregressive.json"
]

# To hold extracted data for each dataset and metric
tokens_per_sec_data = []
rouge_2_data = []

# Helper function to extract the relevant metrics from a single JSON file
def extract_metrics(json_file):
    tokens_per_sec = None
    rouge_2 = None

    with open(json_file, 'r') as f:
        try:
            content = f.read()  # 读取文件的全部内容
            # 将文件中的多个JSON对象通过换行符分开
            content = content.replace('}{', '}\n{')  # 在两个对象之间插入换行符
            objects = content.splitlines()  # 将内容按行分割成多个JSON对象

            for obj_str in objects:
                try:
                    data = json.loads(obj_str.strip())  # 解析每个单独的JSON对象

                    # 提取所需的指标
                    if 'tokens_per_second' in data:
                        tokens_per_sec = data['tokens_per_second'].get('mean', None)
                    if 'predicted_text' in data and 'rouge-2' in data['predicted_text']:
                        rouge_2 = data['predicted_text']['rouge-2']

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in object: {e}")  # 处理无效的JSON对象
                    continue  # 跳过无效的JSON对象

        except Exception as e:
            print(f"Error reading file {json_file}: {e}")  # 处理读取文件时的错误

    return tokens_per_sec, rouge_2

# Iterate over all the files you specified and process them
for json_filename in json_files_to_process:
    file_path = os.path.join(log_dir, json_filename)

    print(f"Processing file: {file_path}")  # Debugging: Print file path

    tokens_per_sec, rouge_2 = extract_metrics(file_path)
    if tokens_per_sec is not None and rouge_2 is not None:
        # Save the data for CSV and further processing
        tokens_per_sec_data.append([json_filename, tokens_per_sec])
        rouge_2_data.append([json_filename, rouge_2])

# Convert the data for CSV into a DataFrame
tokens_df = pd.DataFrame(tokens_per_sec_data, columns=['File', 'Tokens per Sec'])
rouge_df = pd.DataFrame(rouge_2_data, columns=['File', 'ROUGE-2'])

# Merge the two dataframes on the 'File' column
merged_df = pd.merge(tokens_df, rouge_df, on='File')

# Save the merged data into a CSV file
csv_file_path = os.path.join(picture_dir, 'combined_results.csv')
merged_df.to_csv(csv_file_path, index=False)

print(f"Results saved to {csv_file_path}")