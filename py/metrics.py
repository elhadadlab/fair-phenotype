'''
This metrics file takes in data in a similar setup to what was provided in crohns_phenotypes_final.ipynb
and returns the aforementioned metric
'''

import numpy as np
import itertools 

def calculate_demographic_parity(list_of_dataframes, men_total, women_total):
    '''
    Calculates and returns the demographic parity for as many phenotypes as given (A, B, C, etc).
    Quoting the PAKDD lecture, the fundamental idea is that the proportion of each protected class 
    should receive the positive (and negative) outcomes at equal rates. 
    Let class 0 be men (8507), class 1 be women (8532).
    
    Input:
        list_of_dataframes - List of Pandas dataframes queried from a server [for example, for first dataframe:
            sql_query_string = 'select * from dbo.results.disease_demographics where cohort_definition_id = XYZ'
            dataframe_1 = pd.io.sql.read_sql(sql_query_string, conn)]
        men_total - Int, number of men in total in the dataframe
        women_total - Int, number of women total in the dataset
    Returns:
        diffs - List of floats of demographic parities
    '''
    
    diffs = []

    for phenotype in list_of_dataframes:
        class_0_prop_outcome = len(phenotype[phenotype.gender_concept_id == 8507]) / (men_total)
        class_1_prop_outcome = len(phenotype[phenotype.gender_concept_id == 8532]) / (women_total)
        print('Men: ' + str(class_0_prop_outcome))
        print('Women: ' + str(class_1_prop_outcome))
        print('Diff: ' + str(class_1_prop_outcome - class_0_prop_outcome))
        diffs.append(class_1_prop_outcome - class_0_prop_outcome)
        
    return diffs

def calculate_equality_of_opportunity(list_of_dataframes, cohort_definition_ids):
    '''
    Calculates and returns the equality of opportunity for as many phenotypes as given (A, B, C, etc).
    Again quoting the PAKDD lecture, equalized odds is when the predicted output of the model is independent of
    the protected attribute conditional on the data. This definition is restrictive because it’s hard to
    calculate conditional independence, so there is instead a relaxed version (also known as the Equality of
    Opportunity) which is used instead. We need the true definitions. In this case, the true cases are majority
    class for the phenotypes, or the set of patients that are in the majority of cases.
    
    Input:
        list_of_dataframes - List of Pandas dataframes queried from a server [for example, for first dataframe:
            sql_query_string = 'select * from dbo.results.disease_demographics where cohort_definition_id = XYZ'
            dataframe_1 = pd.io.sql.read_sql(sql_query_string, conn)]
        cohort_definition_ids - List of cohort definition IDs matching the list of dataframes
    Returns:
        diffs - List of floats of equality of opportunities
    '''
    
    num_phenotypes = len(list_of_dataframes)
    majority_count = np.ceil(num_phenotypes / 2)

    all_sets = {}
    inds = []
    
    for df, cohort_definition_id in zip(list_of_dataframes, cohort_definition_ids):
        all_sets[cohort_definition_id] = set(df[df.cohort_definition_id == cohort_definition_id].subject_id.unique())
        inds.append(cohort_definition_id)
    
    # Permute inds for all possible sets
    Y = set()
    for ele in itertools.combinations(inds, r = int(majority_count)):
        # print(ele)
        intermediate_set = set()
        for item in ele:
            intermediate_set = intermediate_set.intersection(all_sets[item])
        
        Y = Y.union(intermediate_set)
    
    full_df = pd.concat([df[df.person_id.isin(Y)] for df in list_of_dataframes]).drop_duplicates()

    # Let class 0 be men (8507), class 1 be women (8532)

    diffs = []

    for phenotype in list_of_dataframes:
        class_0_eqq_opp = len(phenotype[(phenotype.gender_concept_id == 8507) & 
                                        (phenotype.person_id.isin(Y))]) / len(full_df[full_df.gender_concept_id == 8507])
        class_1_eqq_opp = len(phenotype[(phenotype.gender_concept_id == 8532) & 
                                        (phenotype.person_id.isin(Y))]) / len(full_df[full_df.gender_concept_id == 8532])
        print('Men: ' + str(class_0_eqq_opp))
        print('Women: ' + str(class_1_eqq_opp))
        print('Diff: ' + str(class_1_eqq_opp - class_0_eqq_opp))
        diffs.append(class_1_eqq_opp - class_0_eqq_opp)
        
    return diffs

def calculate_predictive_rate_parity(list_of_dataframes, cohort_definition_ids):
    '''
    Calculates and returns the predictive rate parity for as many phenotypes as given (A, B, C, etc).
    Again quoting the PAKDD lecture, predictive rate parity (or sufficiency) means that the true label
    is independent of the protected attribute conditional on the predicted outputs.
    
    Input:
        list_of_dataframes - List of Pandas dataframes queried from a server [for example, for first dataframe:
            sql_query_string = 'select * from dbo.results.disease_demographics where cohort_definition_id = XYZ'
            dataframe_1 = pd.io.sql.read_sql(sql_query_string, conn)]
        cohort_definition_ids - List of cohort definition IDs matching the list of dataframes
    Returns:
        diffs - List of floats of predictive rate parities
    '''
    
    num_phenotypes = len(list_of_dataframes)
    majority_count = np.ceil(num_phenotypes / 2)

    all_sets = {}
    inds = []
    
    for df, cohort_definition_id in zip(list_of_dataframes, cohort_definition_ids):
        all_sets[cohort_definition_id] = set(df[df.cohort_definition_id == cohort_definition_id].subject_id.unique())
        inds.append(cohort_definition_id)
    
    # Permute inds for all possible sets
    Y = set()
    for ele in itertools.combinations(inds, r = int(majority_count)):
        # print(ele)
        intermediate_set = set()
        for item in ele:
            intermediate_set = intermediate_set.intersection(all_sets[item])
        
        Y = Y.union(intermediate_set)

    diffs = []

    for phenotype in list_of_dataframes:
        class_0_prp = len(phenotype[(phenotype.gender_concept_id == 8507) & 
                                    (phenotype.person_id.isin(Y))]) / len(phenotype[phenotype.gender_concept_id == 8507])
        class_1_prp = len(phenotype[(phenotype.gender_concept_id == 8532) & 
                                    (phenotype.person_id.isin(Y))]) / len(phenotype[phenotype.gender_concept_id == 8532])
        print('Men: ' + str(class_0_prp))
        print('Women: ' + str(class_1_prp))
        print('Diff: ' + str(class_1_prp - class_0_prp))
        diffs.append(class_1_prp - class_0_prp)
        
    return diffs