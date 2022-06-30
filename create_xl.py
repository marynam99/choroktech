# 1. create excel file
import pandas as pd
import os

# writer = pd.ExcelWriter('dataset_fb_0502.xlsx', engine='xlsxwriter')
# writer.save()

# dataframe Name and Age columns
df = pd.DataFrame({'Name': ['A', 'B', 'C', 'D'],
                   'Age': [10, 0, 30, 50]})

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('demo.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
df.to_excel(writer, sheet_name='Sheet1', index=False)

# Close the Pandas Excel writer and output the Excel file.
writer.save()

PATH = r'C:\Users\SM-PC\HCILab\chorok_dataset\skin_0502'
CAT = os.listdir(PATH)
# print(CAT.sort(key=int))
# print(CAT)
cat = []
for i in range(len(CAT)):
    cat.append(CAT[i].split('.')[0])
cat.sort(key=int)

for i in range(len(CAT)):
    for j in range(len(CAT)):
        if j == int(cat):
            print(cat)
            print(CAT[j])


# # Create a Pandas Excel writer using XlsxWriter as the engine.
# writer = pd.ExcelWriter('dataset_fb_0502.xlsx', engine='xlsxwriter')
# for cat in CAT:
#     print(cat)
#     cat_path = os.path.join(PATH, cat)
#     cat_list = os.listdir(cat_path)
#
#     cat_df = pd.DataFrame(cat_list)
#
#     # Convert the dataframe to an XlsxWriter Excel object.
#     cat_df.to_excel(writer, sheet_name=cat[:30], index=False)
#
# # Close the Pandas Excel writer and output the Excel file.
# writer.save()