import numpy as np
import castle
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.algorithms import PC, GES, ICALiNGAM, GOLEM, RL, DirectLiNGAM, TTPM,Notears,NotearsNonlinear,ANMNonlinear,ICALiNGAM,PNL, DAG_GNN
from main import dfn


def structural_hamming_distance(adj_matrix1, adj_matrix2):
    # Ensure both adjacency matrices have the same shape
    assert adj_matrix1.shape == adj_matrix2.shape, "Adjacency matrices must have the same shape."

    # Convert adjacency matrices to boolean arrays
    adj_matrix1_bool = adj_matrix1.astype(bool)
    adj_matrix2_bool = adj_matrix2.astype(bool)

    # Calculate the number of edge additions and deletions
    additions_deletions = np.sum(np.logical_xor(adj_matrix1_bool, adj_matrix2_bool))

    # Calculate the number of edge reversals
    adj_matrix1_reversed = adj_matrix1_bool.T
    reversals = np.sum(np.logical_and(adj_matrix1_reversed, adj_matrix2_bool))

    # Compute the Structural Hamming Distance
    shd = additions_deletions + reversals

    return shd

# NEED TO ADD SENSETIVITY TRESHOLD
def filter_edges(graph, threshold):
    filtered_graph = graph.copy()
    edges_to_remove = [(u, v) for u, v, d in graph.edges(data=True) if abs(d['weight']) <= threshold]
    filtered_graph.remove_edges_from(edges_to_remove)
    return filtered_graph

def count_nonzero(matrices_list):
    number_of_connections=0
    for i in matrices_list:
        for j in i:
            for k in j:
                if k==1:
                    number_of_connections+=1
    return number_of_connections

dfs=dfn.drop('actions',axis=1)
print(dfs,"EPTAAAAAAAAA")
## PC algorithm 


# Split dataset into 10 different datasets for structureal 
pc_matrices=[]
for data_set in np.array_split(dfs, 10): 
    # Build the model
    pc = PC()
    pc.learn(data_set)
    
    # Create list of matices
    pc_matrices.append(pc.causal_matrix)

# PC evaluation
PC_score=0
for matrix_1 in pc_matrices:
    for matrix_2 in pc_matrices:
        PC_score+=structural_hamming_distance(matrix_1, matrix_2)
print("PC score : ",PC_score)
PC_adj_score=PC_score/count_nonzero(pc_matrices)
print("PC adjusted score : ",PC_adj_score)

## GOLEM algorithm
# Split dataset into 10 different datasets for structureal 
golem_matrices=[]
for data_set in np.array_split(dfs, 10): 
    # Build the model
    golem = GOLEM()
    golem.learn(data_set)
    
    golem_matrices.append(golem.causal_matrix)

GOLEM_score=0
for matrix_1 in golem_matrices:
    for matrix_2 in golem_matrices:
        GOLEM_score+=structural_hamming_distance(matrix_1, matrix_2)
        
print("GOLEM Score : ",GOLEM_score)

GOLEM_adj_score=GOLEM_score/count_nonzero(golem_matrices)
print("GOLEM Adjusted Score : ",GOLEM_adj_score)


# GES 

# Split dataset into 10 different datasets for structureal 
ges_matrices=[]
for data_set in np.array_split(dfs.iloc[:,:2], 10): 
    # Build the model
    ges = GES()#PC(alpha=0.001) # set significance level to 0.01
    ges.learn(data_set)
    
    ges_matrices.append(ges.causal_matrix)

GES_score=0
for matrix_1 in ges_matrices:
    for matrix_2 in ges_matrices:
        GES_score+=structural_hamming_distance(matrix_1, matrix_2)
        
print("GES Score : ",GES_score)

GES_adj_score=GES_score/count_nonzero(ges_matrices)
print("GES Adjusted Score : ",GES_adj_score)

# DirectLINGAM
# Split dataset into 10 different datasets for structureal 
directlingam_matrices=[]
for data_set in np.array_split(dfs, 10): 
    # Build the model
    directlingam =DirectLiNGAM()
    directlingam.learn(data_set)
    
    directlingam_matrices.append(directlingam.causal_matrix)

DirectLiNGAM_score=0
for matrix_1 in directlingam_matrices:
    for matrix_2 in directlingam_matrices:
        DirectLiNGAM_score+=structural_hamming_distance(matrix_1, matrix_2)
print("DirectLiNGAM Score : ",DirectLiNGAM_score)

DirectLiNGAM_adj_score=DirectLiNGAM_score/count_nonzero(directlingam_matrices)
print("DirectLiNGAM Adjusted Score : ",DirectLiNGAM_adj_score)

# NOTEARS
# Split dataset into 10 different datasets for structureal 
notears_matrices=[]
for data_set in np.array_split(dfs, 10): 
    # Build the model
    notears = Notears()#PC(alpha=0.001) # set significance level to 0.01
    notears.learn(data_set)
    
    notears_matrices.append(notears.causal_matrix)

NOTEARS_score=0
for matrix_1 in notears_matrices:
    for matrix_2 in notears_matrices:
        NOTEARS_score+=structural_hamming_distance(matrix_1, matrix_2)
print('NOTEARS Score :',NOTEARS_score)

NOTEARS_adj_score=NOTEARS_score/count_nonzero(notears_matrices)
print('NOTEARS Adjusted Score :',NOTEARS_adj_score)

# ICALiNGAM
# Split dataset into 10 different datasets for structureal 
icalingam_matrices=[]
for data_set in np.array_split(dfs.iloc[:,:2], 10): 
    # Build the model
    icalingam =ICALiNGAM()
    icalingam.learn(data_set)
    
    #create list of different causal matrices to compare
    icalingam_matrices.append(icalingam.causal_matrix)

ICALiNGAM_score=0
for matrix_1 in icalingam_matrices:
    for matrix_2 in icalingam_matrices:
        ICALiNGAM_score+=structural_hamming_distance(matrix_1, matrix_2)
print("ICALiNGAM Score : ",ICALiNGAM_score)

ICALiNGAM_adj_score=ICALiNGAM_score/count_nonzero(icalingam_matrices)
print("ICALiNGAM Adjusted Score : ",ICALiNGAM_adj_score)


# NotearsNonlinear
# Split dataset into 10 different datasets for structureal 
list_matrices=[]
for data_set in np.array_split(dfs, 10): 
    # Build the model
    model =NotearsNonlinear()
    model.learn(data_set)
    
    #create list of different causal matrices to compare
    list_matrices.append(model.causal_matrix)

sh_score=0
for matrix_1 in list_matrices:
    for matrix_2 in list_matrices:
        sh_score+=structural_hamming_distance(matrix_1, matrix_2)
print("NotearsNonlinear Score : ",sh_score)

NotearsNonlinear_adj_score=sh_score/count_nonzero(list_matrices)
print("NotearsNonlinear Adjusted Score : ",NotearsNonlinear_adj_score)

# List to store all variables
results = []

# Add each algorithm's scores to the list
results.append({"Algorithm": "PC", "Score": PC_score, "Adjusted Score": PC_adj_score})
results.append({"Algorithm": "GOLEM", "Score": GOLEM_score, "Adjusted Score": GOLEM_adj_score})
results.append({"Algorithm": "GES", "Score": GES_score, "Adjusted Score": GES_adj_score})
results.append({"Algorithm": "DirectLiNGAM", "Score": DirectLiNGAM_score, "Adjusted Score": DirectLiNGAM_adj_score})
results.append({"Algorithm": "NOTEARS", "Score": NOTEARS_score, "Adjusted Score": NOTEARS_adj_score})
results.append({"Algorithm": "ICALiNGAM", "Score": ICALiNGAM_score, "Adjusted Score": ICALiNGAM_adj_score})
results.append({"Algorithm": "NotearsNonlinear", "Score": sh_score, "Adjusted Score": NotearsNonlinear_adj_score})

# Now you can print or manipulate 'results' as needed
for result in results:
    print(result)