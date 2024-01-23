import numpy as np
import pandas as pd
import gym
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import mean_squared_error
from matplotlib import pyplot as plt
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym.envs import box2d
from keras.models import load_model
import warnings
import tensorflow.compat.v1
warnings.filterwarnings("ignore")
from collections import OrderedDict




def load_dataframe(path_to_dataset):
# Load dataframe with environment data

    if path_to_dataset=="Training_datasets/Bipedal_walker_PPO_training_data.pkl":

        # Specify the path of your CSV file
        df = pd.read_pickle(path_to_dataset)

        # Convert to single elements dataframe without lists-3 
        dfn = pd.DataFrame()


        columns_states=['Hull Angle', 'Hull Angular Velocity', 'Horizontal Speed',
            'Vertical Speed', 'Hip Joint 1 Angle', 'Hip Joint 1 Speed',
            'Knee Joint 1 Angle', 'Knee Joint 1 Speed', 'Hip Joint 2 Angle',
            'Hip Joint 2 Speed', 'Knee Joint 2 Angle', 'Knee Joint 2 Speed',
            'Lidar Range 1', 'Lidar Range 2', 'Lidar Range 3', 'Lidar Range 4',
            'Lidar Range 5', 'Lidar Range 6', 'Lidar Range 7', 'Lidar Range 8',
            'Lidar Range 9', 'Lidar Range 10', 'Leg 1 Ground Contact',
            'Leg 2 Ground Contact']

        columns_actions=['Torque on Hip Joint 1','Torque on Knee Joint 1', 'Torque on Hip Joint 2','Torque on Knee Joint 2']

        for i in range(len(df)):
            values=df['state'][i].tolist()
            dfn.loc[i,columns_states]=values

            values=df['action'][i].tolist()
            dfn.loc[i,columns_actions]=values

            dfn.loc[i,['reward', 'done']]=df.loc[i,['reward', 'done']]
        
        return dfn
    elif path_to_dataset=="Training_datasets/Cartpole_A2C_training_data.pkl":
        # Specify the path of your CSV file
        df = pd.read_pickle(path_to_dataset)

        # Convert to single elements dataframe without lists-3 
        dfn = pd.DataFrame()

        columns_states=['Position','Velocity','Angle','Angular Velocity']

        for i in range(len(df)):
            values=df['state'][i].tolist()
            dfn.loc[i,columns_states]=values
            dfn.loc[i,['actions']]=df.loc[i,['action']][0]
            dfn.loc[i,['reward', 'done']]=df.loc[i,['reward', 'done']]
        return dfn

    elif path_to_dataset=="Training_datasets/MountainCar_DQN_training_data.pkl":

        # Specify the path of your CSV file
        df = pd.read_pickle(path_to_dataset)

        # Convert to single elements dataframe without lists-3 
        dfn = pd.DataFrame()

        columns_states=['x','x_vel']

        for i in range(len(df)):
            values=df['state'][i].tolist()
            dfn.loc[i,columns_states]=values
            dfn.loc[i,['actions']]=df.loc[i,['action']][0]
            dfn.loc[i,['reward', 'done']]=df.loc[i,['reward', 'done']]
        return dfn
    elif path_to_dataset=="Training_datasets/Pendulum_training_data.pkl":

        # Specify the path of your CSV file
        df = pd.read_pickle(path_to_dataset)
        
        # Convert to single elements dataframe without lists-3 
        dfn = pd.DataFrame()

        columns_states=['x','y','ang_vel']

        for i in range(len(df)):
            values=df['state'][i].tolist()
            dfn.loc[i,columns_states]=values
            dfn.loc[i,['actions']]=df.loc[i,['action']][0]
            dfn.loc[i,['reward', 'done']]=df.loc[i,['reward', 'done']]
        return dfn
    elif path_to_dataset=="Training_datasets/Lunar_lander_DQN_training_data.pkl":


        # Specify the path of your CSV file
        df = pd.read_pickle(path_to_dataset)
        # Convert to single elements dataframe without lists-3 
        dfn = pd.DataFrame()


        columns_states=['x','y','x_vel','y_vel','angle','ang_vel','r_leg','l_leg']


        for i in range(len(df)):
            values=df['state'][i].tolist()
            dfn.loc[i,columns_states]=values
            dfn.loc[i,['actions']]=df.loc[i,['action']][0]
            dfn.loc[i,['reward', 'done']]=df.loc[i,['reward', 'done']]
        return dfn
    else:
        print("Invalid option")

if __name__ == "__main__":
    """
    Possible Input options are : 
        Bipedal_walker_PPO_training_data.pkl
        Cartpole_A2C_training_data.pkl
        MountainCar_DQN_training_data.pkl
        Pendulum_training_data.pkl

    """
    import os
    # Folder and file names
    folder_name = "Training_datasets"
    file_name = "Lunar_lander_DQN_training_data.pkl"

    # Construct the full path
    path = os.path.join(folder_name, file_name)
    dfn=load_dataframe(path)


    # Drop column done
    dfn.drop('done', axis=1, inplace=True)

    if path=="Training_datasets/Pendulum_training_data.pkl":
        
        from clustering import cluster_labels

        # Apply the function to the 'actions' column
        dfn['actions'] = cluster_labels
        
    elif path=="Training_datasets/Bipedal_walker_PPO_training_data.pkl":
        from clustering import cluster_labels

        # Apply the function to the 'actions' column
        dfn['actions'] = cluster_labels
    else:
        pass



    # # Compare perfomance of causal discovery algorithm on the given dataset
    # filename = 'comparing_algorithms.py'
    # with open(filename) as file:
    #     exec(file.read())


    # Splitting the original dataset into action-specific parts
    df_dict = {}

    # Get unique action values
    unique_actions = dfn['actions'].unique()

    # Iterate over each unique action value
    for action in unique_actions:
        # Create a new DataFrame entry in the dictionary for each action
        # The key is formatted as 'df_' followed by the action value
        df_dict[f"df_{action}"] = dfn[dfn['actions'] == action].drop('actions', axis=1)

    # Now df_dict contains a DataFrame for each unique action, with the action column dropped
        

    column_names = dfn.columns.tolist()
    column_names.remove('actions')
    column_names


    # Create an empty list to store the DataFrames
    dfs_list = []
    # Iterate over the values of the dictionary and add them to the list
    for key in df_dict:
        dfs_list.append(key)

    import numpy as np
    import networkx as nx
    import castle
    from castle.common import GraphDAG
    from castle.metrics import MetricsDAG
    from castle.algorithms import PC, GES, ICALiNGAM, GOLEM, RL, DirectLiNGAM, TTPM,Notears,NotearsNonlinear,ANMNonlinear,ICALiNGAM,PNL, DAG_GNN

    graphs={} # define a dictionary that will house all graphs 


    for dict in dfs_list:
        df_state_vars=df_dict[dict].iloc[:,:]
        nd_array = df_state_vars.to_numpy()
        
        # Build the model contrainet based
        algorithm = GOLEM() #GOLEM
        algorithm .learn(nd_array)
        adjacency_matrix = algorithm.causal_matrix # Get the learned adjacency matrix

        #  # Build the model socre based
        # dilingam = ICALiNGAM()
        # dilingam.learn(nd_array)
        # adjacency_matrix = dilingam.weight_causal_matrix   # Get the learned adjacency matrix

        # if dict=="df_0":
        #     first_matrix=adjacency_matrix


        
        # Get learned graph
        learned_graph = nx.DiGraph()
        
        # Add nodes to the graph
        nodes = column_names
        learned_graph.add_nodes_from(nodes)
        
        # Add weighted edges to the graph
        for i in range(adjacency_matrix.shape[0]):
            for j in range(adjacency_matrix.shape[1]):
                if adjacency_matrix[i, j] != 0:
                    learned_graph.add_edge(nodes[i], nodes[j], weight=adjacency_matrix[i, j])
        
        
        graphs[dict]=learned_graph



    import matplotlib.pyplot as plt
    import networkx as nx

    # Assuming graphs is your dictionary containing different graphs

    # Filter the edges based on the threshold (if necessary)
    # Assuming threshold is defined
    # For example: threshold = 0

    filtered_graphs = {}
    # Put to dictionary
    action_graphs={}

    for key, graph in graphs.items():
        # If you need to filter the graph based on some condition, do it here
        # For now, we're assuming no filtering is needed
        filtered_graphs[key] = graph
        action_graphs[key]=graph


    new_dict={}

    for key in range(len(action_graphs)):
        # Assuming G is your existing Directed Acyclic Graph (DAG)
        G = action_graphs[dfs_list[key]]

        # Node of interest
        node_of_interest = 'reward'

        # Get the set of all ancestors of the node of interest
        ancestors = nx.ancestors(G, node_of_interest)

        # Create the new DAG
        new_G = nx.DiGraph()

        # Add the nodes and edges that are ancestors of the node of interest
        for node in ancestors:
            # Add the node itself
            new_G.add_node(node)

            # Add all edges in G that end at this node and start at another ancestor
            for pred in G.predecessors(node):
                if pred in ancestors:
                    new_G.add_edge(pred, node)

        # Add the node of interest and its edges to the new graph
        new_G.add_node(node_of_interest)
        for pred in G.predecessors(node_of_interest):
            new_G.add_edge(pred, node_of_interest)


        fig, axes = plt.subplots(1, 2, figsize=(30, 15))
        new_G
        G=filtered_graphs[dfs_list[key]]

        # Draw the original graph G
        nx.draw_shell(G, ax=axes[0], with_labels=True, node_color='lightblue', node_size=16000, width=2.0, arrowsize=20)
        axes[0].set_title("Original Graph causal graph", fontsize=20)

        # Draw the new graph new_G
        nx.draw_spring(new_G, ax=axes[1], with_labels=True, node_color='lightgreen', node_size=16000, width=4.0, arrowsize=50,font_size=18)
        axes[1].set_title("Ancestral subgraph of rewards node", fontsize=20)

        # Add a title to the entire plot
        fig.suptitle(f"Causal influence graphs for action {key}", fontsize=35, fontweight='bold')
        # Save the figure
        fig.savefig(f"Causal influence graphs for action {key}", dpi=300)  # Specify the filename and resolution
        new_dict[key] = new_G



    

    dfs=dfn
    # Remove the last column and store it in a variable
    last_col = dfs.pop(dfs.columns[-1])

    # Insert the last column at the second last position
    dfs.insert(len(dfs.columns) - 1, last_col.name, last_col)
    dfs



    # Dfine action_set and obs_set
    state_set=[]
    action_set=[]
    for index, row in list(dfs.iterrows())[:]:
        

        # add instance of state instance to the list of obs_set
        state_set.append(row[:9].tolist())
        

        # add instance of action to action_set list
        action_set.append(str(int(row[-1:].tolist()[0])))



    next_state_set = state_set[1:] # Inputs:[0, 1, 2, 3, 4, 5] Outputs: [1, 2, 3, 4, 5]
    action_set = action_set[:-1] # Input:[0, 1, 2, 3, 4, 5] Outputs: [0, 1, 2, 3, 4]
    state_set = state_set[:-1]
        
    action_influence_dataset = {} # will be dictionary {action1:{'state' : [], 'next_state': []}, action2: {'state' : [], 'next_state': []}}

    for i in range(len(action_set)):
        if action_set[i] in action_influence_dataset:
            action_influence_dataset[action_set[i]]['state'].append(state_set[i])
            action_influence_dataset[action_set[i]]['next_state'].append(next_state_set[i])
        else:
            action_influence_dataset[action_set[i]] = {'state' : [], 'next_state': []}
            action_influence_dataset[action_set[i]]['state'].append(state_set[i])
            action_influence_dataset[action_set[i]]['next_state'].append(next_state_set[i])



    def generate_unique_functions(graph_dict):   #before generate_unique_functions(graph_dict)
        unique_functions = {}

        for action, graph in graph_dict.items():
            for node in graph.nodes():
                predecessors = graph.predecessors(node)
                key = (node, action)
                if key not in unique_functions:
                    unique_functions[key] = set()
                unique_functions[key].update(predecessors)
                
                #////// delete empty sets
                if unique_functions[key]==set():
                    # delete element of the array
                    del unique_functions[key]
                #//////
                
        return unique_functions

    uniqueu_functions=generate_unique_functions(new_dict)




    def name_to_index(name):
        
        # Alternatively you can use "first_nine" variable
        #name_index=first_nine.tolist()
        name_index=column_names

        index = name_index.index(name)
        return index




    """use different types of regressors"""        	
    def get_regressor(x_feature_cols, key, regressor_type):
        if regressor_type == 'lr':
            return tf.estimator.LinearRegressor(feature_columns=x_feature_cols, model_dir='./scm_models/linear_regressor/1_0'+str(key[0])+'_'+str(key[1]))
        if regressor_type == 'mlp':
            return tf.estimator.DNNRegressor(hidden_units=[64, 32, 16], feature_columns=x_feature_cols, model_dir='./scm_models/mlp/'+str(key[0])+'_'+str(key[1]))
        if regressor_type == 'dt':
            return tensorflow.compat.v1.estimator.BoostedTreesRegressor(feature_columns=x_feature_cols, n_batches_per_layer=1000, n_trees=1, model_dir='./scm_models/decision_tree/'+str(key[0])+'_'+str(key[1]))
        
    structeral_equations = {}

    for key in uniqueu_functions: # key is (node,action) i.e. ('Lidar Range 4', '0')
        if str(key[1]) in action_influence_dataset:
            x_data = []
            for x_feature in uniqueu_functions[key]: # x_feature i.e. "Knee Joint 1 Angle"
                x_data.append(np.array(action_influence_dataset[str(key[1])]['state'])[:,name_to_index(x_feature)])

            x_feature_cols = [tf.feature_column.numeric_column(str(i)) for i in range(len(x_data))]
            y_data = np.array(action_influence_dataset[str(key[1])]['next_state'])[:,name_to_index(key[0])] #- NEXT STATE
            #y_data = np.array(action_influence_dataset[str(key[1])]['state'])[:,name_to_index(key[0])]
            structeral_equations[key] = {
                                        'X': x_data,
                                        'Y': y_data,
                                        'function': get_regressor(x_feature_cols, key,'mlp')# select regressor here
                                        }
        else :
            print(key[1],action_influence_dataset)
            print('TRUE ? :',key[1] in action_influence_dataset)
            print(x_data)
            print(structeral_equations)
            import sys
            sys.exit()


    # used in train_structeral_equations() defines training parameters

    def get_input_fn(data_set, num_epochs=None, n_batch = 128, shuffle=False):
            x_data = {str(k): data_set['X'][k] for k in range(len(data_set['X']))}
            # print("X_data SHAPE : ",pd.DataFrame(x_data).shape,"\nDATA :",pd.DataFrame(x_data))
            # print("Y_data SHAPE: ",pd.Series(data_set['Y']).shape,"\nDATA :",pd.Series(data_set['Y']))
            return tensorflow.compat.v1.estimator.inputs.pandas_input_fn(       
                    x=pd.DataFrame(x_data),
                    y = pd.Series(data_set['Y']),       
                    batch_size=n_batch,          
                    num_epochs=num_epochs,       
                    shuffle=shuffle)


    # If getting errror try to delete folder with schepoints
    def train_structeral_equations():
        # defines training parameters to train arbitrar regressor
        for key in structeral_equations:
            input_f=get_input_fn(structeral_equations[key],num_epochs=None,n_batch = 128,shuffle=False)
            # print("KEY :",key)
            # print("HERE IS THE INPUT :",input_f)
            structeral_equations[key]['function'].train(input_fn=input_f,steps=1000) 
    train_structeral_equations()



    def get_prediction_input_fn(data_list):
        # make each number a list and put all of them into another list
        # convert to DataFrame
        
        def generate_x_data(feature_indexes, values):
            if isinstance(values, list):
                feature_indexes=list(range(len(values)))
                x_data = {str(feature_indexes[i]): values[i] for i in range(len(values))}
            else:  # if values is a single number
                x_data = {str(feature): values for feature in feature_indexes}
            return x_data
        
        x_data = generate_x_data(['0'],data_list)

        
        
        return tensorflow.compat.v1.estimator.inputs.pandas_input_fn(
            x=pd.DataFrame(x_data, index=[0]),shuffle=False)


    def predict_action(index):
        reward_list=[]
        for i in unique_actions:
            values = dfs.loc[index, uniqueu_functions[('reward',int(i))]].tolist()
            predictions=structeral_equations[('reward',int(i))]['function'].predict(input_fn=get_prediction_input_fn(values))
            
        
            for prediction in predictions:
                reward_list.append(prediction['predictions'][0])
        print('Reward list :',reward_list)
        return reward_list.index(max(reward_list))

    actions_list=[]
    for i in range(100):
        actions_list.append(predict_action(i))
    actions_list


    # Counter number of differenting vlaues:
    # Count differing values
    count=0
    for index,value in enumerate(actions_list):
        if value != dfs['actions'].iloc[index]:
            count+=1
    print('Faithfulnes of the resultant model: ',count)






    ### This function can be used to predict to predict action from the state input
    #def predict_action_from_state(state):
