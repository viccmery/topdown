
import pandas as pd
import numpy as np
import os 
import pyarrow.feather as feather
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point
from scipy.spatial.distance import cdist
from shapely.affinity import scale
from shapely.wkt import dumps as wkt_dumps
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import cv2
from shapely import wkt
from shapely.affinity import scale
from shapely.wkt import loads as load_wkt
from itertools import combinations
from joblib import Parallel, delayed
from itertools import product
import itertools

class PseudoAnalysis:

    def __init__(self, directory):
         self.directory = directory 
         self.pseudo_files = [] # list of the files 
         self.pseudo_data = {}  # Initialize the track_data dictionary

         self.use_shorten = True 
         self.shorten_duration = None

    def pseudo(self):
        self.pseudo_files = [f for f in os.listdir(self.directory) if f.startswith('pseudo_population_') and f.endswith('.csv')]
        for file in self.pseudo_files: 
            path = os.path.join(self.directory, file)
            df = pd.read_csv(path)
            self.pseudo_data[file] = df

    def shorten(self, frame=-1):
        for file in self.pseudo_files:
            df = self.pseudo_data[file]
            df = df[df['frame'] <= frame]
            self.pseudo_data[file] = df # update the track data 

        self.use_shorten = True
        self.shorten_duration = frame  # e.g., 600

    def digging_mask(self):
        for file in self.pseudo_files:
            df = self.pseudo_data[file]
            df = self.compute_digging(df)

            if df['digging_status'].any():
                print(f"Digging detected in file: {file}") # get rid off 

            self.pseudo_data[file] = df[df['digging_status'] == False].copy()



    # def compute_digging(self, df):
    #     df = df.sort_values(['track_id', 'frame']).reset_index(drop=True)

    #     df['x'] = (
    #         df.groupby('track_id', group_keys=False)['x_body']
    #         .apply(lambda x: x.rolling(window=5, min_periods=1).mean()))
    #     df['y'] = (
    #         df.groupby('track_id', group_keys=False)['y_body']
    #         .apply(lambda y: y.rolling(window=5, min_periods=1).mean()))

    #     df['dx'] = df.groupby('track_id')['x'].diff().fillna(0)
    #     df['dy'] = df.groupby('track_id')['y'].diff().fillna(0)

    #     df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
    #     df['is_moving'] = df['distance'] > 0.1

    #     df['cumulative_displacement'] = df.groupby('track_id')['distance'].cumsum()
    #     df['cumulative_displacement_rate'] = df.groupby('track_id')['cumulative_displacement'].apply(lambda x: x.diff(10) / 10).fillna(0)

    #     df['x_std'] = df.groupby('track_id')['x'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
    #     df['y_std'] = df.groupby('track_id')['y'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
    #     df['overall_std'] = np.sqrt(df['x_std']**2 + df['y_std']**2)

    #     df['final_movement'] = (df['cumulative_displacement_rate'] > 0.1) | ((df['overall_std'] > 0.1) & (df['is_moving']))

    #     window_size = 50
    #     df['digging_status'] = (
    #         df.groupby('track_id')['final_movement']
    #         .transform(lambda x: (~x).rolling(window=window_size, center=False).apply(lambda r: r.sum() >= (window_size / 2)).fillna(0).astype(bool)))
        

    #     ### backfilling TRUE for larvae that actually end up digging 

    #     df['prev'] = (
    #             df.groupby('track_id')['digging_status']
    #             .shift(1)
    #             .fillna(False)
    #         )
    #     df['false_true'] = df['digging_status'] & ~df['prev'] # digging status = True ; prev frame digging status = False


    #     df['future_digging'] = (
    #     df.groupby('track_id')['digging_status']
    #     .rolling(window=50, min_periods=50)
    #     .sum()
    #     .shift(-49)
    #     .reset_index(level=0, drop=True)
    # )
    #     df['long_digging'] = df['false_true'] & (df['future_digging'] >= 50)

    #     # 1) Initialize backfill column
    #     df['backfill'] = False

    #     # 2) Loop per track
    #     for track_id, group in df.groupby('track_id'):
    #         idx   = group.index
    #         starts = idx[group.loc[idx, 'long_digging']]
    #         for s in starts:
    #             pre = max(idx.min(), s - 30)
    #             df.loc[pre:s-1, 'backfill'] = True  # back-fill up to the frame *before* 

    #     df['digging_status'] = df['digging_status'] | df['backfill']

    #     df.drop(columns=['backfill', 'long_digging', 'false_true', 'future_digging'], inplace=True)

    #     return df
    
    ## COMPUTE DIGGING MATCHING THE BEHAVIOURAL ANALYSIS.PY CLASS (UNSURE WHEN CHANGED)

    def compute_digging(self, df):
        df = df.sort_values(['track_id', 'frame']).reset_index(drop=True)

        # Smooth x and y
        df['x'] = (
            df.groupby('track_id', group_keys=False)['x_body']
            .apply(lambda x: x.rolling(window=5, min_periods=1).mean())
        )
        df['y'] = (
            df.groupby('track_id', group_keys=False)['y_body']
            .apply(lambda y: y.rolling(window=5, min_periods=1).mean())
        )

        # Differences
        df['dx'] = df.groupby('track_id')['x'].diff().fillna(0)
        df['dy'] = df.groupby('track_id')['y'].diff().fillna(0)

        # Distance and moving status
        df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
        df['is_moving'] = df['distance'] > 0.1

        # Cumulative and std
        df['cumulative_displacement'] = df.groupby('track_id')['distance'].cumsum()
        df['cumulative_displacement_rate'] = df.groupby('track_id')['cumulative_displacement'].apply(lambda x: x.diff(10) / 10).fillna(0)

        df['x_std'] = df.groupby('track_id')['x'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
        df['y_std'] = df.groupby('track_id')['y'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
        df['overall_std'] = np.sqrt(df['x_std']**2 + df['y_std']**2)

        df['movement_score'] = df['cumulative_displacement_rate'] * df['overall_std']

        df['final_movement'] = (df['cumulative_displacement_rate'] > 0.1) | (df['movement_score'] > 0.25)

        ## smoothed final movement
        window_size = 50
        df['digging_status'] = (
            df.groupby('track_id')['final_movement']
            .transform(lambda x: (~x).rolling(window=window_size, center=False).apply(lambda r: r.sum() >= (window_size * 0.8)).fillna(0).astype(bool))
        )

        ### backfilling TRUE for larvae that actually end up digging 

        df['prev'] = (
                df.groupby('track_id')['digging_status']
                .shift(1)
                .fillna(False)
            )
        df['false_true'] = df['digging_status'] & ~df['prev'] # digging status = True ; prev frame digging status = False


        df['future_digging'] = (
        df.groupby('track_id')['digging_status']
        .rolling(window=50, min_periods=50)
        .sum()
        .shift(-49)
        .reset_index(level=0, drop=True)
    )
        df['long_digging'] = df['false_true'] & (df['future_digging'] >= 50)

        # 1) Initialize backfill column
        df['backfill'] = False

        # 2) Loop per track
        for track_id, group in df.groupby('track_id'):
            idx   = group.index
            starts = idx[group.loc[idx, 'long_digging']]
            for s in starts:
                pre = max(idx.min(), s - 30)
                df.loc[pre:s-1, 'backfill'] = True  # back-fill up to the frame *before* 

        df['digging_status'] = df['digging_status'] | df['backfill']

        df.drop(columns=['backfill', 'long_digging', 'false_true', 'future_digging'], inplace=True)

        # df.to_csv(os.path.join(self.directory, 'test.csv'), index=False)

        return df




    def distance_from_centre(self): 

        data = []

        for file in self.pseudo_files:
            predictions = self.pseudo_data[file]
            predictions.sort_values(by='frame', ascending=True)

            centre_x, centre_y = 0, 0

            for index, row in predictions.iterrows():
                x, y = row['x_body'], row['y_body']
                distance = np.sqrt((centre_x - x)**2 + (centre_y - y)**2)
                data.append({'file': file, 'frame': row['frame'], 'track': row['track_id'], 'distance_from_centre': distance})

        df_distance_over_time = pd.DataFrame(data)

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"distance_from_centre{suffix}.csv"
        df_distance_over_time.to_csv(os.path.join(self.directory, filename), index=False)
    
    
    def euclidean_distance(self):

        data = []

        for file in self.pseudo_files:
            track_data = self.pseudo_data[file]
            track_data.sort_values(by='frame', ascending=True)

            for frame in track_data['frame'].unique():
                unique_frame =  track_data[track_data['frame'] == frame]

                # if unique_frame.empty or unique_frame[['x_body', 'y_body']].isnull().any().any(): #unecessary?
                #     print(f"Skipping frame {frame} in {pseudo_track} due to missing data.")
                #     continue
                body_coordinates = unique_frame[['x_body', 'y_body']].to_numpy()
                distance = cdist(body_coordinates, body_coordinates, 'euclidean')
                np.fill_diagonal(distance, np.nan)

                average_distance = np.nanmean(distance)
                data.append({'time': frame, 'average_distance': average_distance, 'file': file})

        df = pd.DataFrame(data)
        df = df.sort_values(by=['time', 'file'], ascending=True)

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"euclidean_distances{suffix}.csv"
        df.to_csv(os.path.join(self.directory, filename), index=False)


    def speed(self):

        data = []

        for file in self.pseudo_files:
            track_data = self.pseudo_data[file]
            for track in track_data['track_id'].unique():
                track_unique = track_data[track_data['track_id'] == track]

                for i in range(len(track_unique) - 1):

                    row = track_unique.iloc[i]
                    next_row = track_unique.iloc[i+1]

                    distance = np.sqrt((row['x_body'] - next_row['x_body'])**2 + (row['y_body'] - next_row['y_body'])**2)

                    time1 = row['frame']
                    time2 = next_row['frame']
                    time = time2 - time1

                    if time > 2:
                        continue

                    speed_value = distance / time 
                    data.append({'time': time2, 'speed': speed_value, 'file': file})
    
        speed_over_time = pd.DataFrame(data)
        speed_over_time = speed_over_time.sort_values(by=['time'], ascending=True)

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"speed_over_time{suffix}.csv"
        speed_over_time.to_csv(os.path.join(self.directory, filename), index=False)


    def acceleration(self):

        data = []

        for file in self.pseudo_files:
            track_data = self.pseudo_data[file]
            for track in track_data['track_id'].unique():
                track_unique = track_data[track_data['track_id'] == track]

                previous_speed = None
                previous_time = None

                for i in range(len(track_unique) - 1):

                    row = track_unique.iloc[i]
                    next_row = track_unique.iloc[i+1]

                    distance = np.sqrt((row['x_body'] - next_row['x_body'])**2 + (row['y_body'] - next_row['y_body'])**2)

                    time1 = row['frame']
                    time2 = next_row['frame']
                    time = time2 - time1
                    if time > 2:
                        continue

                    speed_value = distance / time 

                    if previous_speed is not None and previous_time is not None:
                        acceleration_value = (speed_value - previous_speed) / time 
                        data.append({'time': time2, 'acceleration': acceleration_value, 'file': file})

                    previous_speed = speed_value
                    previous_time = time
    
        acceleration_accross_time = pd.DataFrame(data)
        acceleration_accross_time = acceleration_accross_time.sort_values(by=['time'], ascending=True)

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"acceleration_accross_time{suffix}.csv"
        acceleration_accross_time.to_csv(os.path.join(self.directory, filename), index=False)



    def ensemble_msd(self):

        data = []

        for file in self.pseudo_files:
            track_data = self.pseudo_data[file]

            centre_x, centre_y = 0, 0

            for track_id in track_data['track_id'].unique():
                track_unique = track_data[track_data['track_id'] == track_id].sort_values(by=['frame']).reset_index(drop=True)

                for _, row in track_unique.iterrows():
                    squared_distance = (row['x_body'] - centre_x) ** 2 + (row['y_body'] - centre_y) ** 2
                    data.append({
                    'time': row['frame'], 
                    'squared_distance': squared_distance, 
                    'file': file
                })
                    
        df = pd.DataFrame(data)
        df = df.sort_values(by=['time'], ascending=True)

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"ensemble_msd{suffix}.csv"

        output_path = os.path.join(self.directory, filename)
        df.to_csv(output_path, index=False)
        

    def time_average_msd(self, taus):

        dfs = []

        for filename, dataframe in self.pseudo_data.items():
            dataframe['file'] = filename
            dfs.append(dataframe)

        df = pd.concat(dfs, ignore_index=True)

        df = df[["file", "track_id", "frame", "x_body", "y_body"]] 
 
        def msd_per_tau(df, tau): # one value per tau 
            squared_displacements = []
            grouped_data = df.groupby(['file', 'track_id'])

            # really dont get why you have to iterate in such a way ????
            for (file, track_id), unique_track in grouped_data:

                unique_track = unique_track.sort_values(by='frame').reset_index(drop=True)

                if len(unique_track) > tau:
                    initial_positions = unique_track[['x_body', 'y_body']].values[:-tau] # values up till tau as a NumPy array # positions from t to t-N-tau # represent starting points
                    tau_positions = unique_track[['x_body', 'y_body']].values[tau:] # values from tau onwards # t+tau to t-N # representing ending points 
                    disp = np.sum((tau_positions - initial_positions) ** 2, axis=1) # squared displacement for each pair
                    squared_displacements.append(disp)  

            if squared_displacements:
            # Flatten the list of arrays into a single NumPy array
                flattened_displacements = np.concatenate(squared_displacements)
            # Filter out NaN and inf values
                valid_displacements = flattened_displacements[np.isfinite(flattened_displacements)]

                if valid_displacements.size > 0:
                    mean_disp = np.mean(valid_displacements)
                    return mean_disp

        msds = []
        for tau in taus:
            msd = msd_per_tau(df, tau)
            msds.append(msd)

        tau_msd_df = pd.DataFrame({'tau': taus, 'msd': msds})
        tau_msd_df = tau_msd_df.sort_values(by='tau', ascending=True)

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"time_average_msd{suffix}.csv"
        tau_msd_df.to_csv(os.path.join(self.directory, filename), index=False)
   


    def trajectory(self):

        dfs = []

        for filename, dataframe in self.pseudo_data.items():
            dataframe['file'] = filename
            dfs.append(dataframe)

        df = pd.concat(dfs, ignore_index=True)

        grouped_data = df.groupby(['file', 'track_id'])
        
        def angle_calculator(vector_A, vector_B):

            # convert to an array for mathmatical ease 
            A = np.array(vector_A, dtype=np.float64)
            B = np.array(vector_B, dtype=np.float64)
            
            # Ensure there are no NaN values in the vectors and check for zero-length vectors
            if not np.isnan(A).any() and not np.isnan(B).any():
                # calculate magnitude of the vector
                magnitude_A = np.linalg.norm(A)
                magnitude_B = np.linalg.norm(B)
                
                # ensure magnitude =! 0
                if magnitude_A != 0 and magnitude_B != 0:
                    # Calculate the dot product
                    dot_product = np.dot(A, B)
                    
                    # cosθ
                    cos_theta = dot_product / (magnitude_A * magnitude_B)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure valid range for arccos
                    # θ in radians
                    theta_radians = np.arccos(cos_theta)
                    # θ in degrees
                    theta_degrees = np.degrees(theta_radians)
                    return theta_degrees
            return np.nan
        
        data = []

        # really dont get why you have to iterate in such a way ????
        for (file, track_id), unique_track in grouped_data:
            unique_track = unique_track.sort_values(by='frame').reset_index(drop=True)

            for i in range(len(unique_track) - 1):
                head = unique_track.iloc[i][['x_head', "y_head"]].values
                body = unique_track.iloc[i][['x_body', 'y_body']].values
                tail = unique_track.iloc[i][['x_tail', 'y_tail']].values

                HB = head - body
                BT = tail - body 

                angle = angle_calculator(HB, BT)

                frame = unique_track.iloc[i]['frame']
    
                data.append({'time': frame, 'angle': angle, 'file': file})
        
        angle_over_time = pd.DataFrame(data)
        angle_over_time = angle_over_time.sort_values(by=['time'], ascending=True)

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"angle_over_time{suffix}.csv"
        angle_over_time.to_csv(os.path.join(self.directory, filename), index=False)



    def contacts(self, proximity_threshold=1): 

        data = []
        no_contacts = []

        def process_track_pair(track_a, track_b, df, track_file, proximity_threshold=1):
            results = []

            track_a_data = df[df['track_id'] == track_a]
            track_b_data = df[df['track_id'] == track_b]

            common_frames = sorted(set(track_a_data['frame']).intersection(track_b_data['frame']))

            if not common_frames:
                return results

            # Precompute node-node distances for all common frames
            parts = ['head', 'body', 'tail']
            distance_rows = []

            for frame in common_frames:
                row_a = track_a_data[track_a_data['frame'] == frame]
                row_b = track_b_data[track_b_data['frame'] == frame]

                if row_a.empty or row_b.empty:
                    continue

                positions = {}
                for part in parts:
                    positions[f'a_{part}'] = row_a[[f'x_{part}', f'y_{part}']].to_numpy().flatten()
                    positions[f'b_{part}'] = row_b[[f'x_{part}', f'y_{part}']].to_numpy().flatten()

                distances = {
                    'head_head': np.linalg.norm(positions['a_head'] - positions['b_head']),
                    'body_body': np.linalg.norm(positions['a_body'] - positions['b_body']),
                    'tail_tail': np.linalg.norm(positions['a_tail'] - positions['b_tail']),
                    'head_tail': np.linalg.norm(positions['a_head'] - positions['b_tail']),
                    'tail_head': np.linalg.norm(positions['a_tail'] - positions['b_head']),
                    'body_head': np.linalg.norm(positions['a_body'] - positions['b_head']),
                    'head_body': np.linalg.norm(positions['a_head'] - positions['b_body']),
                    'body_tail': np.linalg.norm(positions['a_body'] - positions['b_tail']),
                    'tail_body': np.linalg.norm(positions['a_tail'] - positions['b_body']),
                }

                for interaction_type, dist in distances.items():
                    distance_rows.append({
                        'frame': frame,
                        'interaction_type': interaction_type,
                        'Distance': dist
                    })

            if not distance_rows:
                return results

            # Convert to DataFrame - per interaction
            dist_df = pd.DataFrame(distance_rows)

            # Get min distance & node-node type per frame
            # frame | interaction-type | distance
            min_df = dist_df.groupby('frame').apply(
                lambda g: g.loc[g['Distance'].idxmin()]
            ).reset_index(drop=True)

            # Now iterate through min_df and build bouts
            interaction_id_local = 0
            i = 0
            frames = min_df['frame'].values

            while i < len(min_df):
                frame = frames[i]
                dist = min_df.loc[i, 'Distance']
                interaction_type = min_df.loc[i, 'interaction_type']

                if dist < proximity_threshold:
                    current_bout = []

                    while i < len(min_df):
                        frame = frames[i]
                        dist = min_df.loc[i, 'Distance']
                        interaction_type = min_df.loc[i, 'interaction_type']

                        if dist < proximity_threshold:
                            current_bout.append((frame, dist, interaction_type))
                            i += 1
                        else:
                            break
                else:
                    i += 1
                    continue

                # Check for frame continuity
                bout_frames = [f for f, _, _ in current_bout]
                if bout_frames[-1] - bout_frames[0] + 1 == len(bout_frames):
                    interaction_id_local += 1
                    for frame, dist, interaction_type in current_bout:
                        results.append({
                            'file': track_file,
                            'interaction': interaction_id_local,
                            'frame': frame,
                            'Interaction Pair': (track_a, track_b),
                            'Distance': dist,
                            'Interaction Type': interaction_type
                        })

            return results 

        for file in self.pseudo_files:
            df = self.pseudo_data[file]
            df = df.sort_values(by='frame', ascending=True)
            df['filename'] = file

            track_ids = df['track_id'].unique()
            track_combinations = list(combinations(track_ids, 2))

            all_results = Parallel(n_jobs=-1)(
                delayed(process_track_pair)(track_a, track_b, df, file, proximity_threshold)
                for track_a, track_b in track_combinations)

            flattened_results = [item for sublist in all_results for item in sublist]
            if not flattened_results:
                print(f"No contact results for {file}")
                no_contacts.append(file)
                continue

            results_df = pd.DataFrame(flattened_results)
            results_df.set_index('frame', inplace=True, drop=False)
            data.append(results_df)

        ### for files with no interactions- create placeholders 
        for file in no_contacts:
            placeholder = pd.DataFrame([{
                'file': file,
                'interaction': np.nan,
                'frame': np.nan,
                'Interaction Pair': None,
                'Distance': np.nan,
                'Interaction Type': None,
                'Interaction Number': np.nan
            }])
            data.append(placeholder)

        interaction_data = pd.concat(data, ignore_index=True)

        # Assign global interaction IDs across files and pairs
        interaction_data['Interaction Number'] = (
            interaction_data
            .groupby(['file','Interaction Pair','interaction'])
            .ngroup() + 1  # make it start at 1
        )
        interaction_data.drop(columns=['interaction'], inplace=True)  # Drop the local ID if you don't need it

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"contacts_{proximity_threshold}mm{suffix}.csv"
        interaction_data.to_csv(os.path.join(self.directory, filename), index=False)


    # def nearest_neighbour(self):

    #         dfs = []

    #         for file in self.pseudo_files:
    #             df = self.pseudo_data[file]
    #             df = df.sort_values(by='frame', ascending=True)
    #             df['filename'] = file
            

    #             def speed(group, x, y):
    #                 dx = group[x].diff()
    #                 dy = group[y].diff()
    #                 distance = np.sqrt(dx**2 + dy**2)
    #                 dt = group['frame'].diff()
    #                 speed = distance / dt.replace(0, np.nan) # Avoid division by zero
    #                 return speed

    #             df['speed'] = df.groupby('track_id').apply(lambda group: speed(group, 'x_body', 'y_body')).reset_index(level=0, drop=True)
    #             df['acceleration'] = df.groupby('track_id')['speed'].diff() / df.groupby('track_id')['frame'].diff()
        

    #             def calculate_angle(df, v1_x, v1_y, v2_x, v2_y):
    #                 dot_product = (df[v1_x] * df[v2_x]) + (df[v1_y] * df[v2_y])
    #                 magnitude_v1 = np.hypot(df[v1_x], df[v1_y])  # Same as sqrt(x^2 + y^2
    #                 magnitude_v2 = np.hypot(df[v2_x], df[v2_y])

    #                 # Avoid division by zero
    #                 cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    #                 cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure values are in valid range for arccos
                    
    #                 return np.degrees(np.arccos(cos_theta))  # Convert radians to degrees
                
    #             df['v1_x'] = df['x_head'] - df['x_body']
    #             df['v1_y'] = df['y_head'] - df['y_body']
    #             df['v2_x'] = df['x_tail'] - df['x_body']
    #             df['v2_y'] = df['y_tail'] - df['y_body']

    #             # Apply function correctly
    #             df['angle'] = calculate_angle(df, 'v1_x', 'v1_y', 'v2_x', 'v2_y')

    #             for frame in df['frame'].unique():
    #                 unique_frame =  df[df['frame'] == frame]
    #                 if len(unique_frame) < 2:
    #                     continue

    #                 body_coordinates = unique_frame[['x_body', 'y_body']].to_numpy()
    #                 distance = cdist(body_coordinates, body_coordinates, 'euclidean')
    #                 np.fill_diagonal(distance, np.nan)

    #                 # unique_frame['body-body'] = np.nanmin(distance, axis=1)
    #                 df.loc[unique_frame.index, 'body-body'] = np.nanmin(distance, axis=1)

    #             dfs.append(df)
    #                 # df.to_csv(os.path.join(self.directory, 'df.csv'), index=False)
            
    #         data = pd.concat(dfs, ignore_index=True)

    #         if self.shorten and self.shorten_duration is not None:
    #             suffix = f"_{self.shorten_duration}"
    #         else:
    #             suffix = ""

    #         filename = f"nearest_neighbour{suffix}.csv"
    #         data.to_csv(os.path.join(self.directory, filename), index=False)


    def nearest_neighbour(self):

        dfs = []

        parts = ['head', 'body', 'tail']

        def unify_interaction_type(p1, p2):
            return '-'.join(sorted([p1, p2]))
        

        for file in self.pseudo_files:
            df = self.pseudo_data[file]
            df = df.sort_values(by='frame', ascending=True)
            df['filename'] = file

            # --------------------------------------------------
            # SPEED + ACCELERATION
            # --------------------------------------------------
            def speed(group, x, y):
                dx = group[x].diff()
                dy = group[y].diff()
                dist = np.sqrt(dx**2 + dy**2)
                dt = group['frame'].diff()
                return dist / dt.replace(0, np.nan)

            df['speed'] = (
                df.groupby('track_id')
                .apply(lambda g: speed(g, 'x_body', 'y_body'))
                .reset_index(level=0, drop=True)
            )

            df['acceleration'] = (
                df.groupby('track_id')['speed'].diff()
                / df.groupby('track_id')['frame'].diff()
            )

            # --------------------------------------------------
            # BODY ANGLE (UNCHANGED)
            # --------------------------------------------------
            df['v1_x'] = df['x_head'] - df['x_body']
            df['v1_y'] = df['y_head'] - df['y_body']
            df['v2_x'] = df['x_tail'] - df['x_body']
            df['v2_y'] = df['y_tail'] - df['y_body']

            def calculate_angle(df, v1_x, v1_y, v2_x, v2_y):
                dot = df[v1_x] * df[v2_x] + df[v1_y] * df[v2_y]
                mag1 = np.hypot(df[v1_x], df[v1_y])
                mag2 = np.hypot(df[v2_x], df[v2_y])
                cos = np.clip(dot / (mag1 * mag2), -1, 1)
                return np.degrees(np.arccos(cos))

            df['angle'] = calculate_angle(df, 'v1_x', 'v1_y', 'v2_x', 'v2_y')

            # --------------------------------------------------
            # OUTPUT COLUMNS
            # --------------------------------------------------
            df['body-body'] = np.nan

            df['other_id'] = np.nan
            df['closest_node_interaction'] = np.nan
            df['closest_node_distance'] = np.nan
            df['approach_angle'] = np.nan

            df['head_other_id'] = np.nan
            df['closest_node_to_head'] = np.nan
            df['head_distance'] = np.nan

            # --------------------------------------------------
            # PER-FRAME COMPUTATION
            # --------------------------------------------------
            for frame, frame_df in df.groupby('frame'):
                if frame_df['track_id'].nunique() < 2:
                    continue

                # ==========================
                # BODY–BODY NEAREST
                # ==========================
                body_coords = frame_df[['x_body', 'y_body']].to_numpy(float)
                D_body = cdist(body_coords, body_coords)
                np.fill_diagonal(D_body, np.nan)

                df.loc[
                    frame_df.index,
                    'body-body'
                ] = np.nanmin(D_body, axis=1)

                # ==========================
                # NODE–NODE NEAREST
                # ==========================
                node_rows = []
                for idx, row in frame_df.iterrows():
                    for part in parts:
                        node_rows.append({
                            'index': idx,
                            'track_id': row['track_id'],
                            'part': part,
                            'x': row[f'x_{part}'],
                            'y': row[f'y_{part}'],
                        })

                nodes = pd.DataFrame(node_rows)
                # coords = nodes[['x', 'y']].to_numpy(float)
                # D = cdist(coords, coords)

                # group node table by focal larva row (df index)
                for focal_idx, focal_nodes in nodes.groupby('index'):
                    focal_track = focal_nodes['track_id'].iloc[0]

                    other_nodes = nodes[nodes['track_id'] != focal_track]
                    if other_nodes.empty:
                        continue

                    A = focal_nodes[['x', 'y']].to_numpy(float)      # 3x2 (head/body/tail)
                    B = other_nodes[['x', 'y']].to_numpy(float)      # (3*(n-1))x2

                    D = cdist(A, B)

                    if np.isnan(D).all():
                        continue

                    a, b = np.unravel_index(np.nanargmin(D), D.shape)

                    focal_part = focal_nodes.iloc[a]['part']
                    nearest = other_nodes.iloc[b]

                    interaction = unify_interaction_type(focal_part, nearest['part'])

                    df.at[focal_idx, 'other_id'] = nearest['track_id']
                    df.at[focal_idx, 'closest_node_interaction'] = interaction
                    df.at[focal_idx, 'closest_node_distance'] = D[a, b]

                     # NEW: closest other node to the focal HEAD
                    focal_head = focal_nodes[focal_nodes['part'] == 'head'][['x', 'y']].to_numpy(float)
                    # if focal_head.size == 2: #one row with two values e.g. xy dont want nans 
                    if focal_head.shape[0] != 0:
        
                        Dh = cdist(focal_head, B)  # 1 x (3*(n-1))
                        if not np.isnan(Dh).all():
                            b_h = int(np.nanargmin(Dh))
                            nearest_h = other_nodes.iloc[b_h]
                            df.at[focal_idx, 'head_other_id'] = nearest_h['track_id']
                            df.at[focal_idx, 'closest_node_to_head'] = nearest_h['part']
                            df.at[focal_idx, 'head_distance'] = float(Dh[0, b_h])

                    # approach angle: body->head vs head->(nearest node)
                    v_body_head = np.array([
                        df.at[focal_idx, 'x_head'] - df.at[focal_idx, 'x_body'],
                        df.at[focal_idx, 'y_head'] - df.at[focal_idx, 'y_body']
                    ])

                    v_head_other = np.array([
                        nearest['x'] - df.at[focal_idx, 'x_head'],
                        nearest['y'] - df.at[focal_idx, 'y_head']
                    ])

                    if np.linalg.norm(v_body_head) > 0 and np.linalg.norm(v_head_other) > 0:
                        cos = np.dot(v_body_head, v_head_other) / (
                            np.linalg.norm(v_body_head) * np.linalg.norm(v_head_other)
                        )
                        df.at[focal_idx, 'approach_angle'] = np.degrees(
                            np.arccos(np.clip(cos, -1, 1))
                        )


            dfs.append(df)

        data = pd.concat(dfs, ignore_index=True)

        suffix = f"_{self.shorten_duration}" if self.shorten and self.shorten_duration else ""
        filename = f"nearest_neighbour{suffix}.csv"
        data.to_csv(os.path.join(self.directory, filename), index=False)








    # def interaction_types(self, threshold=1):
    #     def unify_interaction_type(part1, part2):
    #         return '_'.join(sorted([part1, part2]))

    #     data = []

    #     for file in self.pseudo_files:
    #         df = self.pseudo_data[file]
    #         # track_ids = df['track_id'].unique()

    #         # Set up interaction counters using unified types
    #         interaction_counts = {
    #             'head_head': 0,
    #             'tail_tail': 0,
    #             'body_body': 0,
    #             'body_head': 0,  # includes head_body
    #             'body_tail': 0,  # includes tail_body
    #             'head_tail': 0,  # includes tail_head
    #             'file': file,
    #         }

    #         for frame in df['frame'].unique():
    #             frame_data = df[df['frame'] == frame]

    #             if len(frame_data) < 2:
    #                 continue

    #             # Loop over all unordered combinations of body parts
    #             for part1, part2 in [
    #                 ('head', 'head'),
    #                 ('tail', 'tail'),
    #                 ('body', 'body'),
    #                 ('head', 'body'),
    #                 ('tail', 'body'),
    #                 ('head', 'tail'),
    #             ]:
    #                 interaction_type = unify_interaction_type(part1, part2)

    #                 positions1 = frame_data[[f'track_id', f'x_{part1}', f'y_{part1}']].to_numpy()
    #                 positions2 = frame_data[[f'track_id', f'x_{part2}', f'y_{part2}']].to_numpy()

    #                 ids1 = positions1[:, 0]
    #                 ids2 = positions2[:, 0]
    #                 coords1 = positions1[:, 1:].astype(float)
    #                 coords2 = positions2[:, 1:].astype(float)

    #                 distances = cdist(coords1, coords2)
    #                 mask = ids1[:, None] != ids2[None, :]

    #                 if positions1 is positions2:
    #                     upper_triangle = np.triu((distances < threshold) & mask, k=1)
    #                     interaction_counts[interaction_type] += np.sum(upper_triangle)
    #                 else:
    #                     # interaction_counts[interaction_type] += np.sum((distances < threshold) & mask)
    #                     for i, id1 in enumerate(ids1):
    #                         for j, id2 in enumerate(ids2):
    #                             if id1 < id2 and distances[i, j] < threshold:
    #                                 interaction_counts[interaction_type] += 1

    #         data.append(interaction_counts)

    #     interaction_df = pd.DataFrame(data)
    #     melted_df = interaction_df.melt(id_vars='file', var_name='interaction_type', value_name='count').sort_values(by='file')

    #     if self.shorten and self.shorten_duration is not None:
    #         suffix = f"_{self.shorten_duration}"
    #     else:
    #         suffix = ""

    #     filename = f"interaction_types{suffix}.csv"
    #     melted_df.to_csv(os.path.join(self.directory, filename), index=False)



    def interaction_types(self, threshold=1):
        def unify_interaction_type(part1, part2):
            return '_'.join(sorted([part1, part2]))

        data = []

        for file in self.pseudo_files:
            df = self.pseudo_data[file]
            # track_ids = df['track_id'].unique()

            df['frame_bin'] = (df['frame'] // 600) * 600  # Bins: 0, 600, 1200...

            for bin in sorted(df['frame_bin'].unique()):
                df_bin = df[df['frame_bin'] == bin]

                # Set up interaction counters using unified types
                interaction_counts = {
                    'head_head': 0,
                    'tail_tail': 0,
                    'body_body': 0,
                    'body_head': 0,  # includes head_body
                    'body_tail': 0,  # includes tail_body
                    'head_tail': 0,  # includes tail_head
                    'file': file,
                    'frame_bin': bin
                }

                # for frame in df['frame'].unique():
                #     frame_data = df[df['frame'] == frame]

                for frame in df_bin['frame'].unique():
                    frame_data = df_bin[df_bin['frame'] == frame]

                    if len(frame_data) < 2:
                        continue

                    # Loop over all unordered combinations of body parts
                    for part1, part2 in [
                        ('head', 'head'),
                        ('tail', 'tail'),
                        ('body', 'body'),
                        ('head', 'body'),
                        ('tail', 'body'),
                        ('head', 'tail'),
                    ]:
                        interaction_type = unify_interaction_type(part1, part2)

                        positions1 = frame_data[[f'track_id', f'x_{part1}', f'y_{part1}']].to_numpy()
                        positions2 = frame_data[[f'track_id', f'x_{part2}', f'y_{part2}']].to_numpy()

                        ids1 = positions1[:, 0]
                        ids2 = positions2[:, 0]
                        coords1 = positions1[:, 1:].astype(float)
                        coords2 = positions2[:, 1:].astype(float)

                        distances = cdist(coords1, coords2)
                        mask = ids1[:, None] != ids2[None, :]

                        if positions1 is positions2:
                            upper_triangle = np.triu((distances < threshold) & mask, k=1)
                            interaction_counts[interaction_type] += np.sum(upper_triangle)
                        else:
                            # interaction_counts[interaction_type] += np.sum((distances < threshold) & mask)
                            ## STOP TRACK 1-2 AND 2-1 BEING COUNTED 
                            for i, id1 in enumerate(ids1):
                                for j, id2 in enumerate(ids2):
                                    if id1 < id2 and distances[i, j] < threshold:
                                        interaction_counts[interaction_type] += 1


                data.append(interaction_counts)

        interaction_df = pd.DataFrame(data)
        melted_df = interaction_df.melt(id_vars=['file', 'frame_bin'], var_name='interaction_type', value_name='count').sort_values(by='file')

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"interaction_types{suffix}.csv"
        melted_df.to_csv(os.path.join(self.directory, filename), index=False)
    


    ### METHOD INTERACTION_TYPES_CLOSEST: COUNTS CLOSEST! PROXIMAL CONTACTS BETWEEN LARVAE (1MM THRESHOLD) 
    def interaction_types_closest(self, threshold=1):

        """
        Frame-level closest-contact detection (no bouts).
        For each larval pair per frame:
        - compute all 9 node-node distances
        - keep only the minimum distance + its node-node type
        - only log frames where min distance < threshold
        Output: one row per (file, frame, pair) contact frame
        """

        data = []
        no_contacts = []

        parts = ['head', 'body', 'tail']
        interaction_pairs = list(itertools.product(parts, parts))

        def unify_interaction_type(part1, part2):
            return '_'.join(sorted([part1, part2]))

        def process_track_pair(track_a, track_b, df, track_file):
            results = []
            track_a_data = df[df['track_id'] == track_a]
            track_b_data = df[df['track_id'] == track_b]

            common_frames = sorted(set(track_a_data['frame']).intersection(track_b_data['frame']))
            if not common_frames:
                return results

            for frame in common_frames:
                row_a = track_a_data[track_a_data['frame'] == frame]
                row_b = track_b_data[track_b_data['frame'] == frame]
                if row_a.empty or row_b.empty:
                    continue

                # build coords
                coords_a = {p: row_a[[f'x_{p}', f'y_{p}']].to_numpy().flatten() for p in parts}
                coords_b = {p: row_b[[f'x_{p}', f'y_{p}']].to_numpy().flatten() for p in parts}

                # compute all 9 distances, keep minimum
                min_dist = float('inf')
            #   min_type = None
                min_part_a = None
                min_part_b = None
                for part1, part2 in interaction_pairs:
                    dist = np.linalg.norm(coords_a[part1] - coords_b[part2])
                    if dist < min_dist:
                        min_dist = dist
                        min_part_a = part1
                        min_part_b = part2
                        # min_type = unify_interaction_type(part1, part2)

                if min_dist < threshold:
                    results.append({
                        'file': track_file,
                        'frame': frame,
                        'Interaction Pair': tuple(sorted((track_a, track_b))),
                        'track_0': track_a,
                        'track_1': track_b,
                        'track_0_node': min_part_a,
                        'track_1_node': min_part_b,
                        'Distance': min_dist,
                        'Closest Interaction Type': unify_interaction_type(min_part_a, min_part_b)
                    })

            return results
        
        for file in self.pseudo_files:
            df = self.pseudo_data[file]
            df = df.sort_values(by='frame', ascending=True)
            df['filename'] = file
            track_file = file

            track_ids = sorted(df['track_id'].unique()) # 0 always first
            track_combinations = list(combinations(track_ids, 2))

            all_results = Parallel(n_jobs=-1)(
                delayed(process_track_pair)(track_a, track_b, df, track_file)
                for track_a, track_b in track_combinations
            )

            flattened_results = [item for sublist in all_results for item in sublist]
            if not flattened_results:
                print(f"No closest-contact frames for {track_file}")
                no_contacts.append(track_file)
                continue

            data.append(pd.DataFrame(flattened_results))

        # placeholders for files with none
        for file in no_contacts:
            data.append(pd.DataFrame([{
                'file': file,
                'frame': np.nan,
                'Interaction Pair': None,
                'Distance': np.nan,
                'Closest Interaction Type': None
            }]))

        closest_df = pd.concat(data, ignore_index=True)

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"closest_contacts_{threshold}mm{suffix}.csv"
        closest_df.to_csv(os.path.join(self.directory, filename), index=False)

        return closest_df










    def total_digging(self, total_larvae=None, cleaned=False):

        data = []

        for file in self.pseudo_files:
            df = self.pseudo_data[file]
            df = self.compute_digging(df)  # apply dynamic method

            if cleaned:
                df['count'] = df.groupby('frame')['track_id'].transform('nunique')
            else:
                df['count'] = total_larvae

            summary = df.groupby('frame').agg(
                number_digging=('digging_status', 'sum'),
                count=('count', 'first')  # same for all rows in group
            ).reset_index()

            summary['moving'] = summary['count'] - summary['number_digging']

            summary['normalised_digging'] = (summary['number_digging'] / summary['count']) * 100
            summary['file'] = file
            data.append(summary)


        result = pd.concat(data, ignore_index=True)
        result = result.sort_values(by='frame', ascending=True)

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"number_digging{suffix}.csv"
        result.to_csv(os.path.join(self.directory, filename), index=False)
    

    def digging_behaviour(self):

        single_larvae = []
        two_larvae = []

        for file in self.pseudo_files:
            df = self.pseudo_data[file]
            df = self.compute_digging(df)

            for frame, group in df.groupby('frame'):
                digging = group[group['digging_status']]
                not_digging = group[~group['digging_status']]

                # Exactly 1 digger: distance to all others
                if len(digging) == 1 and not not_digging.empty:

                    digger_coords = digging[['x_body', 'y_body']].values
                    others_coords = not_digging[['x_body', 'y_body']].values
                    distances = cdist(digger_coords, others_coords)[0]

                    for target_id, dist in zip(not_digging['track_id'], distances):
                        single_larvae.append({
                            'frame': frame,
                            'file': file,
                            'digger_id': digging['track_id'].values[0],
                            'target_id': target_id,
                            'distance': dist
                        })
                # Exactly 2 diggers: mutual distance
                elif len(digging) == 2:
                    coords = digging[['x_body', 'y_body']].values
                    dist = np.linalg.norm(coords[0] - coords[1])
                    ids = digging['track_id'].values
                    two_larvae.append({
                        'frame': frame,
                        'file': file,
                        'digger_id_1': ids[0],
                        'digger_id_2': ids[1],
                        'distance': dist
                    })

        df_single = pd.DataFrame(single_larvae)    
        df_two = pd.DataFrame(two_larvae)

        # Ensure the DataFrames at least have the correct columns even if empty
        if df_single.empty:
            df_single = pd.DataFrame(columns=['frame', 'file', 'digger_id', 'target_id', 'distance'])
        if df_two.empty:
            df_two = pd.DataFrame(columns=['frame', 'file', 'digger_id_1', 'digger_id_2', 'distance'])

        df_single.to_csv(os.path.join(self.directory, 'digging_distances_single.csv'), index=False)
        df_two.to_csv(os.path.join(self.directory, 'digging_distances_pair.csv'), index=False)

    

    def individual_approach_responses(self, threshold=10):

        parts = ['head', 'body', 'tail']

        def min_node_distance(row_a, row_b):
            best_d = np.inf
            best_pair = None

            for p in parts:
                x2 = row_b.get(f"x_{p}", np.nan)
                y2 = row_b.get(f"y_{p}", np.nan)
                if pd.isna(x2) or pd.isna(y2):
                    continue

                d = np.hypot(row_a['x_head'] - x2, row_a['y_head'] - y2)
                if d < best_d:
                    best_d = d
                    best_pair = p

            return best_d, best_pair

        def min_approach_angle(row_a, row_b):
            # heading vector: body -> head (focal larva)
            hx, hy = row_a['x_head'], row_a['y_head']
            bx, by = row_a['x_body'], row_a['y_body']
            v_heading = np.array([hx - bx, hy - by])

            if np.linalg.norm(v_heading) == 0:
                return np.nan, None

            best_angle = np.inf
            best_node = None

            for p in parts:
                tx = row_b.get(f"x_{p}", np.nan)
                ty = row_b.get(f"y_{p}", np.nan)
                if pd.isna(tx) or pd.isna(ty):
                    continue

                v_target = np.array([tx - hx, ty - hy])
                if np.linalg.norm(v_target) == 0:
                    continue

                cosang = np.dot(v_heading, v_target) / (
                    np.linalg.norm(v_heading) * np.linalg.norm(v_target)
                )
                cosang = np.clip(cosang, -1, 1)
                angle = np.degrees(np.arccos(cosang))

                if angle < best_angle:
                    best_angle = angle
                    best_node = p

            return best_angle, best_node
        
        def approach_angle_to_node(row_a, row_b, node):
            # heading vector: body -> head (focal larva)
            hx, hy = row_a['x_head'], row_a['y_head']
            bx, by = row_a['x_body'], row_a['y_body']
            v_heading = np.array([hx - bx, hy - by])

            if np.linalg.norm(v_heading) == 0:
                return np.nan

            tx = row_b.get(f"x_{node}", np.nan)
            ty = row_b.get(f"y_{node}", np.nan)
            if pd.isna(tx) or pd.isna(ty):
                return np.nan

            v_target = np.array([tx - hx, ty - hy])
            if np.linalg.norm(v_target) == 0:
                return np.nan

            cosang = np.dot(v_heading, v_target) / (
                np.linalg.norm(v_heading) * np.linalg.norm(v_target)
            )
            cosang = np.clip(cosang, -1, 1)
            return float(np.degrees(np.arccos(cosang)))
        
        def speed(row_now, row_next):
            return np.hypot(
                row_next['x_body'] - row_now['x_body'],
                row_next['y_body'] - row_now['y_body']
            )


    
        all_events = []


        for file in self.pseudo_files:
            df = self.pseudo_data[file]
            df = df.sort_values(by='frame', ascending=True)
            df['filename'] = file

            frame_groups = dict(tuple(df.groupby('frame')))
            frames = sorted(frame_groups.keys())

            per_frame = {}
            for fr, g in frame_groups.items():
                per_frame[fr] = {
                    tid: row._asdict()
                    for tid, row in zip(g['track_id'], g.itertuples(index=False))
                }

            track_ids = sorted(df['track_id'].unique())

            for focal_id in track_ids:
                for stim_id in track_ids:
                    if stim_id == focal_id:
                        continue

                    in_encounter = False
                    t_start = None
                    d_start = None
                    start_node = None
                    touch = False

                    for frame in frames:
                        rows = per_frame.get(frame, {})
                        if focal_id not in rows or stim_id not in rows:
                            in_encounter = False
                            continue

                        focal = rows[focal_id]
                        stim  = rows[stim_id]

                        d_min, stim_node = min_node_distance(focal, stim)
                        if stim_node is None:
                            continue

                        angle = approach_angle_to_node(focal, stim, stim_node)
                        if not np.isfinite(angle):
                            continue

                        # START (THIS IS THE ONLY NEW GATE)
                        if not in_encounter and d_min < threshold and angle < 35:
                            in_encounter = True
                            t_start = frame
                            d_start = d_min
                            start_node = stim_node
                            touch = False

                            focal_dist_centre = np.hypot(
                                focal['x_body'] - 0,
                                focal['y_body'] - 0
                            )

                        # DURING
                        elif in_encounter and d_min < 1.0:
                            touch = True

                        # END
                        elif in_encounter and d_min >= threshold:
                            all_events.append({
                                'filename': file,
                                'focal_id': focal_id,
                                'stim_id': stim_id,
                                'start_frame': t_start,
                                'start_min_dist': float(d_start),
                                'first_node': start_node,
                                'touch': touch,
                                'dist_from_centre': float(focal_dist_centre)
                            })
                            in_encounter = False

        df_out = pd.DataFrame(all_events)
        df_out.to_csv(
            os.path.join(self.directory, f'individual_approach_responses_{threshold}.csv'),
            index=False)
        
    


    def individual_approach_responses_consistent_approach_angle(self, threshold=10): 

        parts = ['head', 'body', 'tail']

        def min_node_distance(row_a, row_b):
            best_d = np.inf
            best_pair = None

            for p in parts:
                x2 = row_b.get(f"x_{p}", np.nan)
                y2 = row_b.get(f"y_{p}", np.nan)
                if pd.isna(x2) or pd.isna(y2):
                    continue

                d = np.hypot(row_a['x_head'] - x2, row_a['y_head'] - y2)
                if d < best_d:
                    best_d = d
                    best_pair = p

            return best_d, best_pair

        
        def approach_angle_to_node(row_a, row_b, node):
            # heading vector: body -> head (focal larva)
            hx, hy = row_a['x_head'], row_a['y_head']
            bx, by = row_a['x_body'], row_a['y_body']
            v_heading = np.array([hx - bx, hy - by])

            if np.linalg.norm(v_heading) == 0:
                return np.nan

            tx = row_b.get(f"x_{node}", np.nan)
            ty = row_b.get(f"y_{node}", np.nan)
            if pd.isna(tx) or pd.isna(ty):
                return np.nan

            v_target = np.array([tx - hx, ty - hy])
            if np.linalg.norm(v_target) == 0:
                return np.nan

            cosang = np.dot(v_heading, v_target) / (
                np.linalg.norm(v_heading) * np.linalg.norm(v_target)
            )
            cosang = np.clip(cosang, -1, 1)
            return float(np.degrees(np.arccos(cosang)))
        
        def speed(row_now, row_next):
            return np.hypot(
                row_next['x_body'] - row_now['x_body'],
                row_next['y_body'] - row_now['y_body']
            )   
        

        all_frames = []
        event_id = 0

        for file in self.pseudo_files:
            df = self.pseudo_data[file]
            df = df.sort_values(by='frame', ascending=True)
            df['filename'] = file

            frame_groups = dict(tuple(df.groupby('frame')))
            frames = sorted(frame_groups.keys())

            per_frame = {}
            for fr, g in frame_groups.items():
                per_frame[fr] = {
                    tid: row._asdict()
                    for tid, row in zip(g['track_id'], g.itertuples(index=False))
                }

            track_ids = sorted(df['track_id'].unique())

            for focal_id in track_ids:
                for stim_id in track_ids:

                    if stim_id == focal_id:
                        continue

                    in_encounter = False
                    encounter_rows = []
                    t_start = None

                    for i, frame in enumerate(frames):

                        rows = per_frame.get(frame, {})

                        if focal_id not in rows or stim_id not in rows:
                            # end encounter safely
                            if in_encounter:
                                all_frames.extend(encounter_rows)
                                event_id += 1
                                in_encounter = False
                            continue

                        focal = rows[focal_id]
                        stim  = rows[stim_id]

                        # d_min, stim_node = min_node_distance(focal, stim)

                        # if stim_node is None:
                        #     continue

                        # angle = approach_angle_to_node(focal, stim, stim_node)
                        # if not np.isfinite(angle):
                        #     continue

                        # focal head -> closest stim node
                        d_focal_to_stim, stim_node = min_node_distance(focal, stim)
                        if stim_node is None:
                            continue

                        angle_focal = approach_angle_to_node(focal, stim, stim_node)
                        if not np.isfinite(angle_focal):
                            continue

                        # stim head -> closest focal node  (same logic, swapped)
                        d_stim_to_focal, focal_node = min_node_distance(stim, focal)
                        if focal_node is None:
                            continue

                        angle_stim = approach_angle_to_node(stim, focal, focal_node)
                        if not np.isfinite(angle_stim):
                            continue

                        # pick one distance to use consistently (I’d keep focal→stim distance)
                        d_min = d_focal_to_stim


                        # ---- START CONDITION (same as your original) ----
                        if not in_encounter and d_min < threshold and angle_focal < 35:
                            in_encounter = True
                            t_start = frame
                            encounter_rows = []

                        # ---- DURING ENCOUNTER ----
                        if in_encounter and d_min < threshold:

                            focal_speed = np.nan
                            stim_speed = np.nan

                            if i < len(frames) - 1:
                                next_frame = frames[i + 1]
                                rows_next = per_frame.get(next_frame, {})

                                if focal_id in rows_next:
                                    focal_speed = speed(focal, rows_next[focal_id])

                                if stim_id in rows_next:
                                    stim_speed = speed(stim, rows_next[stim_id])

                            encounter_rows.append({
                                'event_id': event_id,
                                'filename': file,
                                'frame': frame,
                                'rel_time': frame - t_start,
                                'focal_id': focal_id,
                                'stim_id': stim_id,
                                'focal_angle': angle_focal,
                                'stim_angle': angle_stim,
                                'distance': d_min,
                                'focal_speed': focal_speed,
                                'stim_speed': stim_speed
                            })

                        # ---- END CONDITION ----
                        elif in_encounter and d_min >= threshold:
                            all_frames.extend(encounter_rows)
                            event_id += 1
                            in_encounter = False

        df = pd.DataFrame(all_frames)
        df = df.sort_values(['event_id', 'frame'])

        touch_frames = (
                df[df['distance'] < 1.5]
                .groupby('event_id')['frame']
                .min()
            )
        
        min_dist_frames = (
                df.loc[df.groupby('event_id')['distance'].idxmin()]
                .set_index('event_id')['frame']
            )

        decision_frame = min_dist_frames.copy()
        # overwrite with touch frame where it exists
        decision_frame.update(touch_frames)

        df['decision_frame'] = df['event_id'].map(decision_frame)
        df_approach = df[df['frame'] <= df['decision_frame']].copy()

        bins = np.arange(0, 181, 30)
        df_approach['angle_bin'] = pd.cut(
            df_approach['stim_angle'],
            bins=bins,
            right=False
        )

        consistent_events = (
            df_approach
            .groupby('event_id')['angle_bin']
            .nunique() == 1
        )
        keep_ids = consistent_events[consistent_events].index
        df_consistent = df_approach[df_approach['event_id'].isin(keep_ids)].copy()

        df_consistent = df_consistent[
            df_consistent.groupby('event_id')['event_id'].transform('size') > 2
        ].copy()


        df_consistent.to_csv(
            os.path.join(self.directory, f'individual_approach_responses_consistent_angle_{threshold}.csv'),
            index=False
        )   











        

            







######################################################################################################################################################


def perform_analysis(directory):
    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        return

    analysis = PseudoAnalysis(directory)
    analysis.pseudo()

    # Optional preprocessing
    # analysis.shorten(frame=600)
    analysis.digging_mask()

    # analysis.distance_from_centre()
    # analysis.euclidean_distance()
    # analysis.speed()
    # analysis.acceleration()
    # analysis.ensemble_msd()
    # analysis.time_average_msd(list(range(1, 101, 1)))
    # analysis.trajectory()
    # analysis.contacts(proximity_threshold=5)
    # analysis.nearest_neighbour()
    # analysis.interaction_types()
    # analysis.interaction_types_closest()
    # analysis.individual_approach_responses()

    # analysis.total_digging(cleaned=True)
    # analysis.digging_behaviour()

    # analysis.individual_approach_responses(2)
    # analysis.individual_approach_responses(3)
    # analysis.individual_approach_responses(4)
    # analysis.individual_approach_responses(5)
    # analysis.individual_approach_responses(6)
    # analysis.individual_approach_responses(7)
    # analysis.individual_approach_responses(8)
    # analysis.individual_approach_responses(9)
    # analysis.individual_approach_responses(10)

    analysis.individual_approach_responses_consistent_approach_angle(10)


    print(f"Analysis complete for {directory}")



perform_analysis('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/group-housed')
perform_analysis('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/pseudo-n10/socially-isolated')
# perform_analysis('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/group-housed')
# perform_analysis('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/pseudo-n2/socially-isolated')







