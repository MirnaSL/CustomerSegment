# import packages and libraries (add as needed)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from IPython import display

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# magic word for producing visualizations in notebook
%matplotlib inline
#=================================================================================
#Needed functions:
def compare_col (df1, df2, column):
    fig, ax = plt.subplots(1,2)
    sns.countplot(df1[column], ax=ax[0])
    sns.countplot(df2[column], ax=ax[1])
    plt.subplots_adjust(wspace=0.6)
    fig.show()
    return





#load data and feature summary files
feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv', sep=';')
missing_values=missing_splited(feat_info["missing_or_unknown"])
#=================================================================================
def clean_data(df):
    """
    based on Udacity help ticket: https://knowledge.udacity.com/questions/514624
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    
    # Put in code here to execute all main cleaning steps:
# 1a- create a list of missing values in each columns/features:
    mis_val = feat_info.missing_or_unknown.apply(lambda x: x.strip('][').split(','))
    mis_val=mis_val.tolist()

    print('The shape of the dataframe is: {}'.fotmat(df.shape))


# 1b- convert missing value codes in columns into NaNs:
    for i in range(len(mis_val)):
        for n in mis_val[i]:
        #print(n, type(n))
            if n == str(-1):
                df[df.columns[i]].replace(to_replace = int(n), value = np.nan, inplace=True)
            if n == str(0):
                df[df.columns[i]].replace(to_replace = int(n), value = np.nan, inplace=True)
            if n == str(9):
                df[df.columns[i]].replace(to_replace = int(n), value = np.nan, inplace=True)
            if n == 'X':
                df[df.columns[i]].replace(to_replace = 'X', value = np.nan, inplace=True)
            if n == 'XX':
                df[df.columns[i]].replace(to_replace = 'XX', value = np.nan, inplace=True)
        
    total_nan= df.isnull().sum().sum()

# 1c- create a dictionay of column names and missing values in them
    col_nan=[]

    for col_head in df.columns:
        n=df[col_head].isnull().sum()
        col_nan.append(n)
    df_misvalcol_dict={}

    for col_head in df.columns:
        azdias_misval_dict[col_head]=azdias_nan[col_head].isnull().sum()
    
    print('Datarame columns and # of missing values: {}'.format(df_misvalcol_dict)


# 1d- investigate patterns in the amount of missing data in each column: a barplot 
    n_bins = 500
    legend = ['Demographic Dataset (azdias)']
    # Creating histogram
    fig, ax = plt.subplots(1, 1,
                            figsize =(17, 17), 
                            tight_layout = True)
    ax.hist(col_nan, bins = n_bins)

    # Adding extra features    
    plt.xlabel("Azdias Features")
    plt.ylabel("Number of NaNs in Feature(column)")
    plt.legend(legend)
    plt.title('Feature Missing Values Histogram')
    # Show plot
    plt.show()
    
# 2a- create list of outlier columns to remove and decide on cutoff
    cutoff_col=20
    outliers_col = azdias_nan.isnull().sum()/azdias_nan.shape[0]*100
    outliers_col = outliers_col[outliers_col>cutoff_col].index
    print('The columns with high NaN values to be removed are: {}'.fotmat(outliers_col))


# 2b- create a dictionary of removed column and their NaN values
    col_outlier_dict = dict((k, df_misvalcol_dict[k]) for k in outliers_col)
    print('This a dictionary of the columns to remove and # of NaNs in each: {}'.fotmat(col_outlier_dict))

# 2c- the dataframe copy with columns removed
    df_colnan=df.copy()
    df_colnan.drop(labels=outliers_col, axis=1, inplace=True)
    print('The shape of the dataframe with hi-NaN columns removed: {}'.fotmat(df_colnan.shape))
# 2d- a bar plot showing the NaN values in remainging columns
    col_nan2=[]
#calculate NaN entries in the remaining columns
    for col_head2 in df_colnan.columns:
        n=df_colnan[col_head2].isnull().sum()
        col_nan2.append(n)

# Investigate patterns in the amount of missing data in each column.
    n_bins = 30
    legend = ['Demographic Dataset (azdias after removing outlier columns)']
    # Creating histogram
    fig, ax = plt.subplots(1, 1,
                            figsize =(17, 17), 
                            tight_layout = True)
    ax.hist(col_nan2, bins = n_bins)
    # Adding extra features    
    plt.xlabel("Number of NaNs in Feature(column)", fontsize=24)
    plt.ylabel("Count", fontsize=24)
    plt.legend(legend)
    plt.title('Feature Missing Values Histogram', fontsize=30)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=24)
    # Show plot
    plt.show()


#3- remove selected rows
# 3a- calculate NaN values in each row
    row_nan = df_colnan.isnull().sum(axis=1)

# 3b- view abr plot of row NaN values to decide on split
    n_bins = 50
    legend = ['Demographic Dataset (azdias)']
    # Creating histogram
    fig, ax = plt.subplots(1, 1,
                            figsize =(17, 17), 
                            tight_layout = True)
    ax.hist(row_nan, bins = n_bins)

    # Adding extra features    
    plt.xlabel("Number of NaNs in Rows", fontsize=24)
    plt.ylabel("Count", fontsize=24)
    plt.legend(legend)
    plt.title('Rows NaN-Values Histogram', fontsize=30)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=24)
    # Show plot
    plt.show()

# 3c- split rows into subsets
    splitat=20

    split1=row_nan[row_nan<splitat].index
    split2=row_nan[row_nan>=splitat].index

    # create a copy of col-cleaned dataframe
    df_rownan = df_colnan.copy()
    df_lessnan_split=df_rownan.drop(labels=split2, axis=0, inplace=False)
    df_morenan_split=df_rownan.drop(labels=split1, axis=0, inplace=False)
    print('The shape of dataframe split with least row-NaNs is : {}'.format(df_lessnan_split.shape))

#3d- compare distributions of n-columns with least missing values
    temp = min(df_misvalcol_dict.values())
    res = [key for key in df_misvalcol_dict if df_misvalcol_dict[key] == temp]

    #choose 5 columns from the key list of with min values
    comp_5col = res[1:6]
    print('Columns whose distributions are compared: {}'.format(comp_5_col))

    # apply the function to the 5 chosen columns
    for i in comp_5col:
        compare_col(df_lessnan_split, df_morenan_split, i)
#==============================================================================================
# based on data types in feat_info, 
    data_type = feat_info['type'].value_counts()
# how many features are there of each data type?
    print('These are the data types in our datafram: {}'.format(data_type))
# build a list of (remaining) categorical variables
    cat_var=[]
    for i in range(feat_info.shape[0]):
        if feat_info['type'][i] == 'categorical':
            cat_var.append(azdias.columns[i])
# using list comprehension to remove features already removed from the categorical var list
    cat_var = [i for i in cat_var if i not in col_outlier_dict]
    print('The remaining categorical variables in the split dataframe: {}'.format(cat_var))
    print('The number of remaining categorical variables: {}'.format(len(cat_var)))
# Assess categorical variables types (binary-numerical, binary non-numerical, multi-level)
    cat_var_dict={}
    for i in cat_var:
        cat_var_dict[i] = df_clean[i].value_counts()


# get the list of mixed variables.
    mix_var=[]
    for i in range(feat_info.shape[0]):
        if feat_info['type'][i] == 'mixed':
            mix_var.append(df.columns[i])

    print('The remaining mixed variables in the split dataframe: {}'.format(mix_var))

    # using list comprehension to remove features already removed from the categorical var list
    mix_var = [i for i in mix_var if i not in col_outlier_dict]

    print('The remaining mixed variables in the split dataframe: {}'.format(mix_var))
#==============================================================================================


# 4- One Hot Encoding Categrocial Variables
    df_clean=df_lessnan_split.copy()
# build the following lists for Customer data
    binary_feat=['ANREDE_KZ','GREEN_AVANTGARDE','SHOPPER_TYP', 'SOHO_KZ']
    binary_num=['ANREDE_KZ','GREEN_AVANTGARDE','SHOPPER_TYP', 'SOHO_KZ']
    binary_nonum=['ANREDE_KZ','GREEN_AVANTGARDE','SHOPPER_TYP', 'SOHO_KZ']
    
    multi_feat=[]
    multi_feat=[i for i in cat_var if i not in binary_feat]

    print('The categorical variables are: {}'.format(cat_var_dict))
    print('The multi-level categorical variables are : {}'.format(len(multi_feat)))
    
    print('The binary numerical variables are: {}'.format(binary_num))
    print('The binary non-numerical variables are: {}'.format(binary_nonum))
    
# build a dictionaly with categorical featurs as keys and entry counts as values 
    entries=[]
    for i in range(len(multi_feat)):
        entries.append(int(len(df_clean[multi_feat[i]].value_counts())))
    
    
    #print(cat_var_dict)
    print('Length of list of columns of encoded variable to be added should = number of categorical variables to encode: {}'.format(len(entries)) )
    print('The numbers of columns (per encoded cat variable) to be added to dataframe are {}'.format(entries))
    print('The total # of columns to be added to dataframe is {}'.format(sum(entries)))

# 4-b create dummy variables encoding the categorical variables and then drop original cat-var columns
    # encode individual categorical variable and append results to create a dataframe of encoded variables.
    w=[]
    for i in range(len(multi_feat)):
        ww=pd.get_dummies(df_clean[multi_feat[i]],prefix=multi_feat[i],columns=multi_feat[i],dummy_na=False)
        w.append(ww)
            
    w=pd.concat(w, axis=1)

    # check shape of the encoded variable dataframe
    print('The shape and type of the encoded variables dataframe and its shape are: {} and {}, respectively '.format(w.shape, type(w)))

    # concatenate encoded variable dataframe to original dataframe and drop original cat-var columns
    df_hotdata = pd.concat([df_clean,w], axis=1) 
    df_hotdata.drop(columns=multi_feat,inplace=True)

    print('The shape of the dataframe after OneHotEncodeing is: {}'.format(df_hotdata.shape))

# 5- ngineering Variables
    #The meanings of the entries in 'PRAEGENDE_JUGENDJAHRE' and 'CAMEO_INTL_2015' were obtained from the Data.Dictionary.md file provided with project material: 
    #"1- create needed # of columns (e.g., movement and decade) duplicating the original 'PRAEGENDE_JUGENDJAHRE' feature and (e.g., wealth, lifestage) duplicating the original 'CAMEO_INTL_2015'.  
    #"2- clean each new variables to only reflect the "dimension" needed.  
    #"   a- make movement variables binary: replace entries with 0s and 1s them being Mainstream or Avantgarde  
    #"   b- make decade variables reflect decades: 40s, 50s, etc  
    #"   c- make wealth reflect wealthy (0), prosperous(1),comfortable(2), less affluent(3), and poor(4)  
    #"   d- make life_stage reflect couples and or families with: pre-children(0), born_children(1), young_children(2), mature(3), and elders(4)
    #"3- drop original feature"


    df_hotdata2=df_hotdata.copy()
# mixed variables to keep (only the mixed variables needed handling), delete all others
    mix_keep=['PRAEGENDE_JUGENDJAHRE', 'CAMEO_INTL_2015']

# using list comprehension to remove mixed features already removed from the categorical var list
    mix_drop = [i for i in mix_var if i not in mix_keep]
    print('These are the mixed variables to drop and not consider in the remaining clustering and modeling: {}'.format(mix_drop)

# drop un-needed mixed features
    df_hotdata2.drop(columns=mix_drop,inplace=True)
    print('The shape of the dataframe before engineering the mixed variables: {}'.format(df_hotdata2.shape)

# a- create new variables
    df_hotdata2['PRAEGENDE_JUGENDJAHRE_movement']=df_hotdata2['PRAEGENDE_JUGENDJAHRE']
    df_hotdata2['PRAEGENDE_JUGENDJAHRE_decade']=df_hotdata2['PRAEGENDE_JUGENDJAHRE']
    print('The shape of the dataframe after adding movement and decade columns is {}'.format(df_hotdata2.shape))
      
# b- clean new movement variable; follow explanation of entries in 'Data.Dictionary.md'
    mainstream=[1, 3, 5, 8, 10, 12, 14]
    Avantgard=[2, 4, 6, 7, 9, 11, 13, 15]
    df_hotdata2['PRAEGENDE_JUGENDJAHRE_movement']=df_hotdata2['PRAEGENDE_JUGENDJAHRE_movement'].replace(mainstream,0)
    df_hotdata2['PRAEGENDE_JUGENDJAHRE_movement']=df_hotdata2['PRAEGENDE_JUGENDJAHRE_movement'].replace(Avantgard,1)


# c- clean the decade variable
    _40=[1,2]
    _50=[3,4]
    _60=[5, 6, 7]
    _70=[8, 9]
    _80=[10, 11, 12, 13]
    _90=[14, 15]

    df_hotdata2['PRAEGENDE_JUGENDJAHRE_decade']=df_hotdata2['PRAEGENDE_JUGENDJAHRE_decade'].replace(_40, 40)
    df_hotdata2['PRAEGENDE_JUGENDJAHRE_decade']=df_hotdata2['PRAEGENDE_JUGENDJAHRE_decade'].replace(_50, 50)
    df_hotdata2['PRAEGENDE_JUGENDJAHRE_decade']=df_hotdata2['PRAEGENDE_JUGENDJAHRE_decade'].replace(_60, 60)
    df_hotdata2['PRAEGENDE_JUGENDJAHRE_decade']=df_hotdata2['PRAEGENDE_JUGENDJAHRE_decade'].replace(_70, 70)
    df_hotdata2['PRAEGENDE_JUGENDJAHRE_decade']=df_hotdata2['PRAEGENDE_JUGENDJAHRE_decade'].replace(_80, 80)
    df_hotdata2['PRAEGENDE_JUGENDJAHRE_decade']=df_hotdata2['PRAEGENDE_JUGENDJAHRE_decade'].replace(_90, 90)
    print('The shape of the dataframe after cleaning movement and decade columns is {}'.format(df_hotdata2.shape))

# d drop oiginal column
    df_hotdata2.drop(columns='PRAEGENDE_JUGENDJAHRE',inplace=True)
    print('The shape of the dataframe after dropping original PRAEGENDE_JUGENDJAHRE column is {}'.format(df_hotdata2.shape))

# Investigate "CAMEO_INTL_2015" and engineer two new variables  
    #The new codes/entries that will reflect wealth are:  
        # (unknown = 0, wealthy = 1, prosperous = 2, comfortable = 3, less affluent = 4, and poor = 5)
    #The new codes/entries that will reflect life stage of couples/families are:  
        #(pre-children = 1, born_children = 2, young_children = 3, mature = 4, and elders = 5)

# 1- create new variables
    df_hotdata2['CAMEO_INTL_2015_wealth']=df_hotdata2['CAMEO_INTL_2015']
    df_hotdata2['CAMEO_INTL_2015_lifestage']=df_hotdata2['CAMEO_INTL_2015']
    print('The shape of the dataframe after adding wealth and life stage columns is {}'.format(df_hotdata2.shape))
      
# a- clean the new wealth variable; follow explanation of entries in 'Data.Dictionary.md'
    wealth=[11, 12, 13, 14, 15]; prosper=[21, 22, 23, 24, 25]; comfi=[31, 32, 33, 34, 35]; lessaff=[41, 42, 43, 44, 45]
    poor=[51, 52, 53, 54, 55]

    df_hotdata2['CAMEO_INTL_2015_wealth']=df_hotdata2['CAMEO_INTL_2015_wealth'].replace(-1,0)
    df_hotdata2['CAMEO_INTL_2015_wealth']=df_hotdata2['CAMEO_INTL_2015_wealth'].replace('XX',0)
    df_hotdata2['CAMEO_INTL_2015_wealth']=df_hotdata2['CAMEO_INTL_2015_wealth'].replace(wealth,1)
    df_hotdata2['CAMEO_INTL_2015_wealth']=df_hotdata2['CAMEO_INTL_2015_wealth'].replace(prosper,2)
    df_hotdata2['CAMEO_INTL_2015_wealth']=df_hotdata2['CAMEO_INTL_2015_wealth'].replace(comfi,3)
    df_hotdata2['CAMEO_INTL_2015_wealth']=df_hotdata2['CAMEO_INTL_2015_wealth'].replace(lessaff,4)
    df_hotdata2['CAMEO_INTL_2015_wealth']=df_hotdata2['CAMEO_INTL_2015_wealth'].replace(poor,5)

# b- clean the life stage variable
    prechild=[11, 21, 31, 41, 51]; bornchild=[12, 22, 32, 42, 52]; youngchild=[13, 23, 33, 43, 53]; mature=[14, 24, 34, 44, 54]
    elder=[15, 25, 35, 45, 55]

    df_hotdata2['CAMEO_INTL_2015_lifestage']=df_hotdata2['CAMEO_INTL_2015_lifestage'].replace(-1,0)
    df_hotdata2['CAMEO_INTL_2015_lifestage']=df_hotdata2['CAMEO_INTL_2015_lifestage'].replace('XX',0)
    df_hotdata2['CAMEO_INTL_2015_lifestage']=df_hotdata2['CAMEO_INTL_2015_lifestage'].replace(prechild,1)
    df_hotdata2['CAMEO_INTL_2015_lifestage']=df_hotdata2['CAMEO_INTL_2015_lifestage'].replace(bornchild,2)
    df_hotdata2['CAMEO_INTL_2015_lifestage']=df_hotdata2['CAMEO_INTL_2015_lifestage'].replace(youngchild,3)
    df_hotdata2['CAMEO_INTL_2015_lifestage']=df_hotdata2['CAMEO_INTL_2015_lifestage'].replace(mature,4)
    df_hotdata2['CAMEO_INTL_2015_lifestage']=df_hotdata2['CAMEO_INTL_2015_lifestage'].replace(elder,5)
    
    print('The shape of the dataframe after cleaning weakth and lige stage columns is {}'.format(df_hotdata2.shape))

# c- drop original column
    df_hotdata2.drop(columns='CAMEO_INTL_2015',inplace=True)
    print('The shape of the dataframe after dropping original CAMEO_INTL_2015 column is {}'.format(df_hotdata2.shape))

    df_ready=df_hotdata2.copy()
    return df_ready

