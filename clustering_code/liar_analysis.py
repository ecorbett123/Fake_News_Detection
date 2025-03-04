import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''
    This contains initial data analysis for the liar data set. 
    It explores the subject distribution and the label distribution.
'''

liar_df = pd.read_csv('/Users/emmacorbett/PycharmProjects/coms6998mlf/liar_dataset/train.tsv', sep='\t', header=None)
liar_df.columns = ["Id", "Label", "Statement", "Subject", "Speaker", "Job_Title", "State_Info", "Party_Affiliation", "Barely_True_Counts", "False_Counts", "Half_True_Counts", "Mostly_True_Counts", "Pants_On_Fire_Counts", "Context"]

# find all unique subjects
subjects = set()
for subject in liar_df['Subject']:
    subject_list = str(subject).split(',')
    subjects.update(subject_list)


plt.hist(liar_df['Label'], bins=12, alpha=0.5, color='blue')
plt.title("Distribution of Statement Labels")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()

top_10_counts = liar_df['Subject'].value_counts().head(10)
print(top_10_counts)

distribution = liar_df.groupby('Subject')['Label'].value_counts(normalize=True).unstack(fill_value=0)
print(distribution)
