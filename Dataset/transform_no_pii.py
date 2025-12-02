import pandas as pd
import re
import csv 

csv_file_path   = 'dataset_with_text_no_pii.csv'  
output_path     = 'output.csv'             
rows_to_analyze = 5


df = pd.read_csv(csv_file_path)
rows_to_analyze = max(rows_to_analyze, len(df))  


records = []

for idx, row in df.iloc[:rows_to_analyze].iterrows():
    sentence = row['final_text']
    subs_dict = eval(row['final_substitutions_dict'])

    classes = list(subs_dict.keys())
    
   
    records.append({
        "Sentence":  sentence,
        "Label":     0,
        "Class":     "",   
        "Sensitive": ""
    })

output_df = pd.DataFrame(records)
output_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)

print(f"Creato il file {output_path} con {len(records)} righe.")
