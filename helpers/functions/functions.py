import pandas as pd


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
    else:
        print("Invalid option")