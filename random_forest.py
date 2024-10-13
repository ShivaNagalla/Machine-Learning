#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import random

warnings.filterwarnings("ignore")


# In[2]:


#GET DATA
df = pd.read_csv("mutations_1.csv")
df = df.set_index('class')
df = df.loc[:, (df != 0).any(axis=0)]


# In[3]:


#CLASSIFICATION PER MUTATION
def classify_row(row, mutation, group_a, group_b):
    if row[mutation] == 1:
        group_a.append(row.name)
    if row[mutation] == 0:
        group_b.append(row.name)


# In[4]:


#TREE BUILDER
def build_tree(df, level, used_mutations=[], group=[]):
    
    #IF ROOT, EMPTY USED MUTATIONS AND SELECT ALL SAMPLES
    if len(group) == 0:
        used_mutations=[]
        group = list(df.index)

    #CALCULATING PHI FUNCTION THROUGH PANDAS (LOOKS BAD BUT EXTREMLY PERFORMATIC)     
    df_classification = pd.DataFrame(index=list(df.columns), columns=[])

    df_classification['NTL'] = df.sum()
    df_classification['NTR'] = len(df)-df_classification['NTL']
    df_classification['NTL_C'] = df[df.index.str.startswith('C')].sum()
    df_classification['NTL_NC'] = df[df.index.str.startswith('NC')].sum()
    df_classification['NTR_C'] = len(df[df.index.str.startswith('C')])-df[df.index.str.startswith('C')].sum()
    df_classification['NTR_NC'] = len(df[df.index.str.startswith('NC')])-df[df.index.str.startswith('NC')].sum()
    df_classification['PL'] = df_classification['NTL']/len(df)
    df_classification['PR'] = df_classification['NTR']/len(df)
    df_classification['PCTL'] = df_classification['NTL_C']/df_classification['NTL']
    df_classification['PNCTL'] = df_classification['NTL_NC']/df_classification['NTL']
    df_classification['PCTR'] = df_classification['NTR_C']/df_classification['NTR']
    df_classification['PNCTR'] = df_classification['NTR_NC']/df_classification['NTR']
    df_classification['QST'] = np.abs(df_classification['PCTL']-df_classification['PCTR']) + np.abs(df_classification['PNCTL']-df_classification['PNCTR'])
    df_classification['2PLPR'] = 2*df_classification['PL']*df_classification['PR']
    df_classification['PHI'] = df_classification['2PLPR']*df_classification['QST']
    df_classification.sort_values(by='PHI', ascending=False, inplace=True)

    #VERIFY IF MUTATION HAS BEEN USED
    unused_mutations = [m for m in df_classification.index if m not in used_mutations]

    #SELECTED ONLY SQRT(N) BEST MUTATIONS
    sqrt_n = int(np.sqrt(len(unused_mutations)))
    unused_mutations = unused_mutations[:sqrt_n]
    
    if not unused_mutations:
        return None

    group_a = []
    group_b = []
    
    #RANDOMIZE 2/3 OF N OUT OF THE SQRT(N) BEST MUTATIONS
    np.random.shuffle(unused_mutations)
    filter = int(2 * len(unused_mutations)/3)
    selected_mutations = unused_mutations[:filter]

    #SELECTING BEST MUTATION AND SPLITTING GROUPS
    best_mutation = max(selected_mutations, key=lambda mutation: df_classification.loc[mutation, 'PHI'] if not pd.isna(df_classification.loc[mutation, 'PHI']) else float('-inf'))
    used_mutations.append(best_mutation)

    df.apply(classify_row, args = (best_mutation, group_a, group_b), axis=1)
    group_a = list(set(group_a))
    group_b = list(set(group_b))

    level = level - 1

    if level == 0:
        return {'mutation': best_mutation, 'group_a': group_a, 'group_b': group_b}
    
    #RECURSIVELY BUILD NEXT LEVELS
    tree_a = build_tree(df.loc[group_a], level, used_mutations, group_a)
    tree_b = build_tree(df.loc[group_b], level, used_mutations, group_b)
    
    return {'mutation': best_mutation, 'group_a': tree_a, 'group_b': tree_b}


# In[5]:


#BUILD RANDOM FOREST
def random_forest(n_trees, levels, df):
    trees = []
    root_nodes = []
    for n in range(n_trees):
        df_bootstrap = df.sample(n=len(df), replace=True)
        if n == 0:
            test_samples = [i for i in df.index if i not in df_bootstrap.index]
        tree = build_tree(df_bootstrap, levels)
        trees.append(tree)
        root_nodes.append(tree['mutation'])
    return trees, root_nodes, test_samples

trees, roots, test_samples = random_forest(35, 10, df)

#RANDOM FOREST REPORT
def get_feature_count(level):
    features = []
    feature_count = []
    for feature in level:
        if feature not in features:
            features.append(feature)
            feature_count.append(1)
        else:
            index = features.index(feature)
            feature_count[index] += 1
    return features, feature_count

print("Root mutations: ", get_feature_count(roots)[0])
print("Number of times each feature was selected as root, respectively", get_feature_count(roots)[1])


# In[6]:


#RANDOM FOREST CLASSIFIER

#MULTI-LEVEL CLASSIFIER
def tree_classifier(sample, tree):
    current_node = tree
    while 'mutation' in current_node:
        mutation = current_node['mutation']
        if sample[mutation] == 1:
            current_node = current_node['group_a']
        else:
            current_node = current_node['group_b']
    counter = sum(1 for value in current_node if value.startswith('C'))
    if len(current_node) / 2 < counter:
        return 1
    else:
        return 0

#RANDOM FOREST CLASSIFIER            
def random_forest_classification(sample, trees):
    classification = 0
    for i in range(len(trees)):
        classification += tree_classifier(sample, trees[i])
    if classification < len(trees)/2:
        return 0, classification
    else:
        return 1, classification


# In[7]:


#REQUESTED CLASSIFICAIONS
for sample in ['C1', 'C10', 'C50', 'NC5', 'NC15']:
    classification, votes = random_forest_classification(df.loc[sample], trees)
    if classification == 0:
        print("Sample ", sample, " is NC, with ", len(roots)-votes, " trees classifying it as NC")
    else:
        print("Sample ", sample, " is C, with ", votes, " trees classifying it as C")

TP = 0
TN = 0
FP = 0
FN = 0

for sample in list(df.index):
    classification = random_forest_classification(df.loc[sample], trees)[0]
    if classification and sample.startswith('C'):
        TP+=1
    if classification and sample.startswith('NC'):
        FP+=1
    if not classification and sample.startswith('C'):
        FN+=1
    if not classification and sample.startswith('NC'):
        TN+=1

accuracy = round((TP+TN)/(TP+TN+FP+FN)*100,2)
sensitivity = round((TP)/(TP+FN)*100,2)
specificity = round((TN)/(TN+FP)*100,2)
precision = round((TP)/(TP+FP)*100,2)
fdr = round((FP)/(TP+FP)*100,2)
miss_rate = round((FN)/(TP+FN)*100,2)
false_or = round((FN)/(TN+FN)*100,2)

print("")
print("METRICS FOR ALL SAMPLES TRAINED AND TESTED")
print("Accuracy (How many samples were properly labeled): ", accuracy, "%")
print("Sensitivity (How many positive samples were identified): ", sensitivity, "%")
print("Specificity (How many negative samples were identified): ", specificity, "%")
print("Precision (How many positive labels are correct): ", precision, "%")
print("Miss Rate (How many positive samples were not detected): ", miss_rate, "%")
print("False Discovery Rate (How many samples that were labeled positive are negative): ", fdr, "%")
print("False Omission Rate (How many samples that were labeled negative are positive): ", false_or, "%")

TP = 0
TN = 0
FP = 0
FN = 0

for sample in list(test_samples):
    classification = random_forest_classification(df.loc[sample], trees)[0]
    if classification and sample.startswith('C'):
        TP+=1
    if classification and sample.startswith('NC'):
        FP+=1
    if not classification and sample.startswith('C'):
        FN+=1
    if not classification and sample.startswith('NC'):
        TN+=1

accuracy = round((TP+TN)/(TP+TN+FP+FN)*100,2)
sensitivity = round((TP)/(TP+FN)*100,2)
specificity = round((TN)/(TN+FP)*100,2)
precision = round((TP)/(TP+FP)*100,2)
fdr = round((FP)/(TP+FP)*100,2)
miss_rate = round((FN)/(TP+FN)*100,2)
false_or = round((FN)/(TN+FN)*100,2)

print("")
print("METRICS FOR TEST SAMPLES (1ST TREE OUT OF BAG SAMPLES)")
print("Accuracy (How many samples were properly labeled): ", accuracy, "%")
print("Sensitivity (How many positive samples were identified): ", sensitivity, "%")
print("Specificity (How many negative samples were identified): ", specificity, "%")
print("Precision (How many positive labels are correct): ", precision, "%")
print("Miss Rate (How many positive samples were not detected): ", miss_rate, "%")
print("False Discovery Rate (How many samples that were labeled positive are negative): ", fdr, "%")
print("False Omission Rate (How many samples that were labeled negative are positive): ", false_or, "%")
