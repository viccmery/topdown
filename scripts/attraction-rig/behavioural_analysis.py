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
import random
from itertools import combinations
from joblib import Parallel, delayed
import re
from itertools import product
from collections import defaultdict, Counter
import itertools
from scipy.ndimage import label, find_objects
from scipy.spatial.distance import pdist


class HoleAnalysis:

    def __init__(self, directory):

        self.directory = directory 
        self.coordinate_files = []
        self.track_files = [] # list of the files 
        self.hole_boundaries = []
        self.matching_pairs = []
        self.track_data = {}  # Initialize the track_data dictionary # actually has the data so we dont have to keep reloading 
        
        self.perimeter()
        self.coordinates() # used by hole boundary 
        self.hole_boundary()
        self.tracks()
        self.match_files()
        self.conversion()

        self.use_shorten = True 
        self.shorten_duration = None

        # self.digging = None


    # METHOD COORDINATES: IDENTIFIES AND STORES THE HOLE COORDINATE FILES

    def coordinates(self):
        # 2024-05-20_16-08-22_td1_hole.csv
        self.coordinate_files = [f for f in os.listdir(self.directory) if f.endswith('hole.csv')]
        print(f"Coordinate files: {self.coordinate_files}")

    # METHOD TRACKS: IDENTIES AND STORES THE SLEAP TRACK FILES; TRACK DATA IS SUBSEQUENTLY READ  

    def tracks(self):
        # 2024-04-30_14-31-44_td5.000_2024-04-30_14-31-44_td5.analysis.csv
        self.track_files = [f for f in os.listdir(self.directory) if f.endswith('tracks.feather')]
    
        for track_file in self.track_files: 
            track_path = os.path.join(self.directory, track_file)
            df = pd.read_feather(track_path)
            # NEED DIAMATER CONVERSION FFS 
            # # cant access the perimeter right now here 
            # diameter = self.diameter()
            # print(diameter)

            # pixels_to_mm = ['x_tail', 'y_tail', 'x_body', 'y_body', 'x_head', 'y_head']
            # df[pixels_to_mm] = df[pixels_to_mm] * (90 / diameter)
            # print(df.head())
            self.track_data[track_file] = df
    
   # METHOD SHORTEN: OPTIONAL METHOD TO SHORTEN THE TRACK FILES TO INCLUDE UP TO A CERTAIN FRAME  
    
    def shorten(self, frame=-1):

        for track_file in self.track_files:

            df = self.track_data[track_file]
            df = df[df['frame'] <= frame]
            self.track_data[track_file] = df # update the track data 

            # # create path 
            # shortened_path = os.path.join(self.directory, track_file.replace('.feather', f'_shortened_{frame}.feather'))
            # # save 
            # df.reset_index(drop=True, inplace=True)  # Feather requires a default integer index
            # df.to_feather(shortened_path)  # Save the DataFrame without 'index=False'
            # print(f"Shortened file saved: {shortened_path}")
        self.use_shorten = True
        self.shorten_duration = frame  # e.g., 600

        
    ### METHOD DIGGING_MASK: FILTERS FOR NON-DIGGING LARVAE

    def digging_mask(self):

        for track_file in self.track_files:
            df = self.track_data[track_file]
            df = self.compute_digging(df)
            # df.to_csv(os.path.join(self.directory, 'digging.csv'), index=False) # get rid 
            self.track_data[track_file] = df[df['digging_status'] == False].copy()
    

    ### METHOD HOLE_MASK: FILTERS FOR NON-HOLE LARVAE

    def hole_mask(self):

        for track_file in self.track_files:

            df = self.track_data[track_file]
            mask = (~df['within_hole']) & (~df['digging_outside_hole'])  # exclude both
            df = df[mask].copy()  # update df with filtered version
            self.track_data[track_file] = df  # save it back
            # df.to_csv(os.path.join(self.directory, 'test.csv'), index=False)

            

    # METHOD POST_PROCESSING: 1) FILTERS TRACK'S AVERAGE INSTANCE SCORE < 0.9 

    def post_processing(self):
        
        for track_file in self.track_files:
            df = self.track_data[track_file]
            # group by tracks, calculate mean per tracks, if True >= 0.9 include in df 
            df = df[df.groupby('track_id')['instance_score'].transform('mean') >= 0.9]

            self.track_data[track_file] = df  # Update the in-memory version
    

    # METHOD PERIMETER: IDENTIFY XY CENTRE POINTS AND PERIMETER OF THE PETRI DISH

    def perimeter(self):
        
        # function to process the video 1) identify centre coordinates and the perimeter
        def process_video(video_path):
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            # Check if the perimeter file already exists
            wkt_file_path = os.path.join(self.directory, f"{video_name}_perimeter.wkt")
            if os.path.exists(wkt_file_path):
                return

            def detect_largest_circle(frame):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_blurred = cv2.medianBlur(gray, 5)
        
                circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.0, minDist=100,
                                       param1=500, param2=50, minRadius=400, maxRadius=600)
                if circles is not None:
                    largest_circle = max(circles[0, :], key=lambda c: c[2])  # No rounding for accuracy
                    return largest_circle  # x, y, r (center coordinates and radius)
                return None

            def circle_to_polygon(x, y, radius, num_points=100):
                angles = np.linspace(0, 2 * np.pi, num_points)
                points = [(x + radius * np.cos(angle), y + radius * np.sin(angle)) for angle in angles]
                return Polygon(points)
            
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 10) # frame 10 
            ret, frame = cap.read()
            
            if ret:
                circle = detect_largest_circle(frame)
                if circle is not None:
                    x, y, r = circle
                    petri_dish_boundary = circle_to_polygon(x, y, r)

                    save_dir = self.directory
                    wkt_file_path = os.path.join(save_dir, f"{video_name}_perimeter.wkt")
                    with open(wkt_file_path, 'w') as f:
                        f.write(petri_dish_boundary.wkt)
                
                    # Draw the circle on the frame
                    cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 2)

                    # Updated PNG-saving logic
                    frame_with_boundary_path = os.path.join(save_dir, f"{video_name}_perimeter.png")
                    cv2.imwrite(frame_with_boundary_path, frame)
            
                else:
                    print(f"No Perimeter detected for {video_name} .")
            else:
                print(f"Failed to extract the 10th frame from the video.")

            cap.release()
            return None
        
        # Iterate through video files in the directory
        video_files = [f for f in os.listdir(self.directory) if f.endswith('.mp4')]
        for file in video_files:
            video_path = os.path.join(self.directory, file)
            process_video(video_path)
    
    def merged_dataframes(self):
        
        dfs = []
        for track_file in self.track_files:
            df = self.track_data[track_file]
            df['file'] = track_file
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)
        output = os.path.join(self.directory, 'merged.track.feather')
        df.to_feather(output)


    # METHOD HOLE_BOUNDARY: CREATES A POLYGON AROUND THE HOLE BOUNDARY WITH SCALAR OPTION
     # 1. CONVEX HULL: CONVEX SHAPE THAT ENCLOSES A SET OF POINTS (CONTINIOUS BOUNDARY)
     # 2. VERTICES: CORNER POINTS OF THE CONVEX SHAPE
     # 3. POLYGON: GEOMETRIC SHAPE FORMED BY CONNECTING THESE VERTICES

    def hole_boundary(self, scale_factor=1.0):  

        self.hole_boundaries = []

        for coordinates in self.coordinate_files:

            file_path = os.path.join(self.directory, coordinates)

            df = pd.read_csv(file_path, header=None, names=['x', 'y'])

            # Convert coordinates from pixels to millimeters using the same conversion factor
            # conversion_factor = 90 / self.diameter
            # df[['x', 'y']] = df[['x', 'y']] * conversion_factor

            points = df[['x', 'y']].values # values creates numpy array

            hull = ConvexHull(points)  

            # struggle with this understanding - ask callum 
            # this retrieves the points of the shapes 
            # defines the boundary points
            hull_points = points[hull.vertices]

            # create the polygon
            polygon = Polygon(hull_points)

            # scaled_polygon = scale(polygon, xfact=scale_factor, yfact=scale_factor, origin='center') # polygon scaled uniform relative to center

            self.hole_boundaries.append(polygon) #change from scale to poly

            wkt_string = wkt_dumps(polygon) # Convert the scaled polygon to WKT format

            # Save the WKT string to a file with the same name as the original but with a .wkt extension
            # hole_boundary = coordinates.replace('.csv', '.wkt')
            hole_boundary = os.path.join(self.directory, coordinates.replace('.csv', '.wkt'))

            with open(hole_boundary, 'w') as f:
                f.write(wkt_string)
        
        print(f"Hole boundaries: {self.hole_boundaries}")
    
    
    # METHOD MATCH_FILES: MATCHES THE TRACK FILES WITH THEIR COORDINATE FILES (BY EXTENTION THE HOLE POLYGON)

    def match_files(self):
        # Initialize a list for all matching pairs
        self.matching_pairs = []

        # Gather all video and perimeter files
        video_files = [f for f in os.listdir(self.directory) if f.endswith('.mp4')]
        perimeter_files = [f for f in os.listdir(self.directory) if f.endswith('_perimeter.wkt')]

        # Iterate over all track files
        for track_file in self.track_files:
            # Extract the common prefix from the track file
            track_prefix = '_'.join(track_file.split('_')[:3]).replace('.tracks.feather', '')
            matched_data = {
                'track_file': track_file,
                'hole_boundary': None,
                'video_file': None,
                'perimeter_file': None}

            # Match with coordinate files (hole boundaries)
            for i, coordinates_file in enumerate(self.coordinate_files):
                hole_prefix = '_'.join(coordinates_file.split('_')[:3]).rsplit('.', 1)[0]
                if hole_prefix == track_prefix:
                    # print(f"Match found: {track_file} with {coordinates_file}")
                    matched_data['hole_boundary'] = self.hole_boundaries[i]  # Assign the associated hole boundary polygon

            # Match with video files
            for video_file in video_files:
                video_prefix = '_'.join(video_file.split('_')[:3]).rsplit('.', 1)[0]
                if video_prefix == track_prefix:
                    matched_data['video_file'] = video_file

            # Match with perimeter files
            for perimeter_file in perimeter_files:
                perimeter_prefix = '_'.join(perimeter_file.split('_')[:3]).rsplit('.', 1)[0]
                if perimeter_prefix == track_prefix:
                    matched_data['perimeter_file'] = perimeter_file
                    # print(f"Match found: {track_file} with {perimeter_file}")

                    # Read the perimeter file and parse it into a Polygon object
                    perimeter_path = os.path.join(self.directory, perimeter_file)
                    with open(perimeter_path, 'r') as f:
                        perimeter_wkt = f.read()

                    polygon = wkt.loads(perimeter_wkt)

                    matched_data['perimeter_polygon'] = polygon           
                    
            # Append the matched data to the matching_pairs list
            self.matching_pairs.append(matched_data)
    
    # METHOD CONVERSION:CONVERTS EACH FILE FROM PIXELS INTO MM

    def conversion(self):

        for match in self.matching_pairs:
            
            perimeter_polygon = match.get('perimeter_polygon')
            
            if perimeter_polygon:
                # Calculate the diameter of the perimeter 
                minx, miny, maxx, maxy = perimeter_polygon.bounds
                diameter = maxx - minx  # This assumes the perimeter is a circle and uses its width as the diameter.

                conversion_factor = 90 / diameter # 90mm 

                # IF PERIMETER DETECTED BADLY 
                threshold = 0.09 #
                if conversion_factor > threshold:
                    print(f"Conversion factor {conversion_factor:.3f} is above threshold for {match['track_file']}. Using default conversion factor:")
                    conversion_factor = 90 / 1032  # Use the old conversion factor
              

                # scaled_perimeter_polygon = scale(perimeter_polygon, xfact=conversion_factor, yfact=conversion_factor,  origin=(0, 0))
                perimeter_coordinates = np.array(perimeter_polygon.exterior.coords)
                perimeter_coordinates *= conversion_factor
                scaled_perimeter_polygon = Polygon(perimeter_coordinates)

                match['perimeter_polygon'] = scaled_perimeter_polygon  # Update the scaled polygon.

                # Apply conversion to hole boundaries.
                hole_boundary = match.get('hole_boundary')
                if hole_boundary:
                    # Scale the hole boundary using the conversion factor.

                    coordinates = np.array(hole_boundary.exterior.coords)

                    coordinates *= conversion_factor
                    scaled_polygon = Polygon(coordinates)
                    print(scaled_polygon)

                    match['hole_boundary'] = scaled_polygon

                    # scaled_hole_boundary = scale(hole_boundary, xfact=conversion_factor, yfact=conversion_factor, origin='center')
                    # match['hole_boundary'] = scaled_hole_boundary  # Update the scaled polygon.
                    # print(scaled_hole_boundary)

                track_file = match['track_file']
                track_data = self.track_data[track_file]

                pixel_columns = ['x_tail', 'y_tail', 'x_body', 'y_body', 'x_head', 'y_head']
                track_data[pixel_columns] = track_data[pixel_columns] * conversion_factor
                self.track_data[track_file] = track_data  # Update the track data.
                print(f"Conversion applied for {track_file} with conversion factor: {conversion_factor:.3f}")
            
            else:
                print(f"no perimeter detected for {match['track_file']}")
  
                conversion_factor = 90 / 1032 # the one i used to use 
                hole_boundary = match.get('hole_boundary')
                if hole_boundary:
                    coordinates = np.array(hole_boundary.exterior.coords)
                    coordinates *= conversion_factor
                    scaled_polygon = Polygon(coordinates)
                    print(scaled_polygon)
                    match['hole_boundary'] = scaled_polygon
                
                track_file = match['track_file']
                track_data = self.track_data[track_file]

                pixel_columns = ['x_tail', 'y_tail', 'x_body', 'y_body', 'x_head', 'y_head']
                track_data[pixel_columns] = track_data[pixel_columns] * conversion_factor
                self.track_data[track_file] = track_data  # Update the track data.
                print(f"Conversion applied for {track_file} with conversion factor: {conversion_factor:.3f}")


    # METHOD HOLE_CENTROID: REPLACE THE HOLE BOUNDARY WITH A HOLE CENTROID COORDINATE 

    def hole_centroid(self):

        updated_matching_pairs = [] # update matching pairs with funderals 

        for track_file, hole_boundary in self.matching_pairs:

            centroid = hole_boundary.centroid  # Calculate centroid of the polygon

            updated_matching_pairs.append((track_file, (centroid.x, centroid.y))) # centroid is a tuple
        
        self.matching_pairs = updated_matching_pairs
        print(f"Matching pairs with centroids: {self.matching_pairs}")
        return self.matching_pairs
    
    # METHOD DISTANCE_FROM_CENTRE: CALCULATES DISTANCES FROM CENTRE COORDINATES 

    def distance_from_centre(self): 

        data = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            perimeter = match.get('perimeter_polygon')
            
            if perimeter is None:
                print(f"No perimeter polygon available for track file: {track_file}")
                continue

            centre_x, centre_y = perimeter.centroid.x, perimeter.centroid.y

            predictions = self.track_data[track_file]

            for index, row in predictions.iterrows():
                x, y = row['x_body'], row['y_body']
                distance = np.sqrt((centre_x - x)**2 + (centre_y - y)**2)

                data.append({'file': track_file, 'frame': row['frame'], 'track': row['track_id'], 'distance_from_centre': distance})

        df_distance_over_time = pd.DataFrame(data)

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"distance_from_centre{suffix}.csv"
    
        df_distance_over_time.to_csv(os.path.join(self.directory, filename), index=False)
        print(f'Distance over time saved: {df_distance_over_time}')

        return df_distance_over_time

    # METHOD EUCLIDEAN_DISTANCE: CALCULATES THE AVERAGE DISTANCE BETWEEN LARVAE ACCROSS FRAMES

    def euclidean_distance(self):

        data = []

        for track_file in self.track_files:
            track_data = self.track_data[track_file]


            for frame in track_data['frame'].unique():

                unique_frame =  track_data[track_data['frame'] == frame]

                # cdist function requires two 2-dimensional array-like objects as inputs
                # create an array of the coordinates for that specific frame
                    
                body_coordinates = unique_frame[['x_body', 'y_body']].to_numpy()

                # The cdist function computes the distance between every pair of points in the two arrays passed to it.

                distance = cdist(body_coordinates, body_coordinates, 'euclidean')

                np.fill_diagonal(distance, np.nan)

                average_distance = np.nanmean(distance)

                data.append({'time': frame, 'average_distance': average_distance, 'file': track_file})

        df = pd.DataFrame(data)
        df = df.sort_values(by=['time', 'file'], ascending=True)

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"euclidean_distances{suffix}.csv"
        df.to_csv(os.path.join(self.directory, filename), index=False)

        return df
    

    # METHOD EUCLIDEAN DISTANCE VARIANCE: TO CALCULATE THE VARIANCE IN THE PLATEU OF THE EUCLIDEAN_DISTANCE DATA

    def euclidean_distance_variance(self, first_frame, last_frame):

        euclidean_df = self.euclidean_distance() # call the euclidean distance
        print('euclidean distance imported')

        euclidean_df = euclidean_df[(euclidean_df['time'] >= first_frame) & (euclidean_df['time'] <= last_frame)]

        distance_variance = euclidean_df.groupby('file')['average_distance'].var()
        # print(distance_variance)

        distance_variance_df = distance_variance.reset_index()
        distance_variance_df.columns = ['file', 'variance']

        distance_variance_df.to_csv(os.path.join(self.directory, 'average_distance_variance.csv'), index=False)
        return distance_variance
    


        ### IDENTIFY MISSING LARVAE AND ASSIGN THEM INSIDE HOLE- OBVS IF THEY DIG THEIR OWN HOLE THIS ISNT GREAT 

    
    # METHOD PROBABILITY_DENSITY: PROBABILITY DENSITY FOR INPUTTED 1D ARRAY
    
    @staticmethod
    def probability_density(df, ax=None, color=None, label=None, linestyle=None):

        data = df.iloc[:, 0].values #iloc indexed based selection - : all rows - 0 first column

        # Replace infinite values with NaN and drop them
        data = pd.Series(data).replace([np.inf, -np.inf], np.nan).dropna().values

        kde = gaussian_kde(data) #kernel estimate density function for the kde data 

        value_range = np.linspace(data.min(), data.max(), 100)
         # this generates a range of values over which the KDE will be evaluated.
        # np.linspace(start, stop, num)

        density = kde(value_range)
   
        # evaluates the KDE at each of the points in the range provided 

        if ax is None:
            ax = plt.gca()
        
        ax = sns.lineplot(x=value_range, y=density, ax=ax, color=color, label=label, linestyle=linestyle)

        return ax # ax = probability_density() to modify graph when called
    
    # METHOD SPEED: CALCULATES SPEED: 1) SPEED VALUES 2) SPEED OVER TIME 

    def speed(self):

        data = []

        for track_file in self.track_files:
            track_data = self.track_data[track_file]

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

                    data.append({'time': time2, 'speed': speed_value, 'file': track_file})
    
        speed_over_time = pd.DataFrame(data)
        speed_over_time = speed_over_time.sort_values(by=['time'], ascending=True)

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"speed_over_time{suffix}.csv"

        speed_over_time.to_csv(os.path.join(self.directory, filename), index=False)

        return speed_over_time
    


    # METHOD ACCELERATION: 

    def acceleration(self):

        data = []

        for track_file in self.track_files:
            track_data = self.track_data[track_file]

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
                        data.append({'time': time2, 'acceleration': acceleration_value, 'file': track_file})

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
        return acceleration_accross_time
        
    
    # METHOD ENSEMBLE_MSD: CALCULATES SQUARED DISTANCE FOR EVERY POSITION FROM THE CENTROID COORDINATES
    
    def ensemble_msd(self):

        data = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            perimeter = match.get('perimeter_polygon')

            # Ensure the perimeter polygon is available
            if perimeter is None:
                print(f"No perimeter polygon available for track file: {track_file}")
                continue

            # Calculate the centroid of the perimeter polygon
            centre_x, centre_y = perimeter.centroid.x, perimeter.centroid.y

            track_data = self.track_data[track_file]

            for track_id in track_data['track_id'].unique():
                track_unique = track_data[track_data['track_id'] == track_id].sort_values(by=['frame']).reset_index(drop=True)

                for _, row in track_unique.iterrows():
                    squared_distance = (row['x_body'] - centre_x) ** 2 + (row['y_body'] - centre_y) ** 2
                    data.append({
                    'time': row['frame'], 
                    'squared_distance': squared_distance, 
                    'file': track_file
                })
                    
        # Create a DataFrame from the MSD data
        df = pd.DataFrame(data)
        df = df.sort_values(by=['time'], ascending=True)

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"ensemble_msd{suffix}.csv"

        # Save the DataFrame as a CSV file
        output_path = os.path.join(self.directory, filename)
        df.to_csv(output_path, index=False)
        print(f"Ensemble MSD saved to {output_path}")
        return df 


    # # METHOD ENSEMBLE_MSD: CALCULATES SQUARED DISTANCE FOR EVERY POSITION FROM FIRST TRACK APPEARANCE
     
    # def ensemble_msd(self): 

    #     # frame distance and file 

    #     data = []

    #     for match in self.matching_pairs:

    #         track_file = match['track_file']
    #         perimeter = match.get('perimeter_polygon')
            
    #         if perimeter is None:
    #             print(f"No perimeter polygon available for track file: {track_file}")
    #             continue

    #         centre_x, centre_y = perimeter.centroid.x, perimeter.centroid.y

        
    #     for track_file in self.track_files:
    #         track_data = self.track_data[track_file]

    #         # calculate average x,y for first frame 
    #         # (needs to change such that it time 0 for each unique track compared back to)

    #         for track in track_data['track_id'].unique():
    #             track_unique = track_data[track_data['track_id'] == track].sort_values(by=['frame'])

    #             x0 = track_unique.iloc[0]['x_body']
    #             y0 = track_unique.iloc[0]['y_body']

    #             for i in range(len(track_unique)):

    #                 squared_distance = (track_unique.iloc[i]['x_body'] - x0)**2 + (track_unique.iloc[i]['y_body'] - y0)**2
    #                 # print(squared_distance)

    #                 frame = track_unique.iloc[i]['frame']

    #                 data.append({'time': frame, 'squared distance': squared_distance, 'file': track_file})
        
    #     df = pd.DataFrame(data)
    #     df = df.sort_values(by=['time'], ascending=True)

    #     df.to_csv(os.path.join(self.directory, 'ensemble_msd.csv'), index=False)

    #     return df 



    # METHOD TIME_AVERAGE_MSD: 
      # taus given in list format e.g. list(range(1, 101, 1))

    def time_average_msd(self, taus):

        dfs = []

        # Iterate over track_data dictionary {'filename': dataframe}
        for filename, dataframe in self.track_data.items():
            # Add a new column to the dataframe with the filename
            dataframe['file'] = filename
            dfs.append(dataframe)

        # Concatenate the dataframes 
        df = pd.concat(dfs, ignore_index=True)

        df = df[["file", "track_id", "frame", "x_body", "y_body"]] # chose specific parts of the dataframe
 
        # one value per tau 
        def msd_per_tau(df, tau):

            squared_displacements = []

            grouped_data = df.groupby(['file', 'track_id'])

            # really dont get why you have to iterate in such a way ????
            for (file, track_id), unique_track in grouped_data:

                unique_track = unique_track.sort_values(by='frame').reset_index(drop=True)

                if len(unique_track) > tau:

                    initial_positions = unique_track[['x_body', 'y_body']].values[:-tau] # values up till tau as a NumPy array # positions from t to t-N-tau # represent starting points
                    tau_positions = unique_track[['x_body', 'y_body']].values[tau:] # values from tau onwards # t+tau to t-N # representing ending points 
                    disp = np.sum((tau_positions - initial_positions) ** 2, axis=1) # squared displacement for each pair
                    # # print(disp) 
                    # print(f"disp for tau={tau}: {disp}")
                    # print(type(disp))

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
   
        return tau_msd_df #.dropna()
    
    # METHOD TRAJECTORY: CALCULATES TRAJECTORY ANGLES: 1) TRAJECTORY ANGLE VALUES 2) TRAJECTORY ANGLE OVER TIME 
      # ANGLE INBETWEEN 2 VECTORS: TAIL-BODY AND BODY-HEAD 

    def trajectory(self):

        dfs = []
        # Iterate over track_data dictionary {'filename': dataframe}
        for filename, dataframe in self.track_data.items():
            # Add a new column to the dataframe with the filename
            dataframe['file'] = filename
            dfs.append(dataframe)

        # Concatenate the dataframes 
        df = pd.concat(dfs, ignore_index=True)

        grouped_data = df.groupby(['file', 'track_id'])
        
        # definition to calculate angle 
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
        
        angles = []
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
                # filename = track_unique.iloc[i]['file']

                angles.append(angle)
                data.append({'time': frame, 'angle': angle, 'file': file})
        

        angle_over_time = pd.DataFrame(data)
        angle_over_time = angle_over_time.sort_values(by=['time'], ascending=True)

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"angle_over_time{suffix}.csv"

        angle_over_time.to_csv(os.path.join(self.directory, filename), index=False)

        return angle_over_time  

    
    # METHOD MOVEMENT_DIRECTION: CALCULATES THE DIRECTION OF MOVEMENT BASED ON BODY NODES OVER TIME 
    def movement_direction(self):

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

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file]
            df = df.sort_values(['track_id', 'frame'])

            for track_id, group in df.groupby('track_id'):
                group = group.sort_values(by='frame')

                body_positions = group[['x_body', 'y_body']].to_numpy(dtype=float)

                vectors = body_positions[1:] - body_positions[:-1] # foo[:-1] (slice) give me everything up to, but not including, the last item
                # makes two lists which are then subtracted to give vector between consecutive frames

                angles = [angle_calculator(vectors[i], vectors[i+1]) for i in range(len(vectors)-1)]

                # angle_frames = group['frame'].to_numpy()[2:]
                angle_frames = group['frame'].to_numpy()[1:-1]


                data.append(pd.DataFrame({
                    'file': track_file,
                    'track_id': track_id,
                    'frame': angle_frames,
                    'movement_angle': angles
                }))
        
        angle_df = pd.concat(data, ignore_index=True)
        angle_df.to_csv(os.path.join(self.directory, "movement_direction.csv"), index=False)
        return angle_df
    






 



    
    # METHOD NUMBER_DIGGING: THIS METHOD DETECTS HOW MANY LARVAE ARE DIGGING (IN ABSENCE OF MAN-MADE HOLE)

    # def number_digging(self, total_larvae):

    #     dataframe_list = [] 

    #     for match in self.matching_pairs:
    #         track_file = match['track_file']
    #         df = self.track_data[track_file]

    #         perimeter = match.get('perimeter_polygon')

    #         df = df.sort_values(by=['track_id', 'frame'])

    #         # DISTANCE MOVED 

    #         # Smooth the positions with a rolling window to reduce noise
    #         df['x'] = df['x_body'].rolling(window=5, min_periods=1).mean()
    #         df['y'] = df['y_body'].rolling(window=5, min_periods=1).mean()

    #         # Calculate the difference between consecutive rows for body coordinates
    #         df['dx'] = df.groupby('track_id')['x'].diff().fillna(0)
    #         df['dy'] = df.groupby('track_id')['y'].diff().fillna(0)

    #         # Calculate the Euclidean distance 
    #         df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)

    #         # Create a boolean mask where x,y movement is greater than 0.1 MM 
    #         df['is_moving'] = df['distance'] > 0.1

    #         # CUMALTIVE DISTANCE 

    #         df['cumulative_displacement'] = df.groupby('track_id')['distance'].cumsum()

    #         df['cumulative_displacement_rate'] = df.groupby('track_id', group_keys=False)['cumulative_displacement', ].apply(lambda x: x.diff(5) / 5).fillna(0) # unsure what groupkeys is but it asked me to put it in cause kept getting lengthy like use this for future
            
    #         # STANDARD DEVIATION OF BODY X, Y COORDINATES 

    #         df['x_std'] = df.groupby('track_id')['x'].transform(lambda x: x.rolling(window=5, min_periods=1).std())
    #         df['y_std'] = df.groupby('track_id')['y'].transform(lambda x: x.rolling(window=5, min_periods=1).std())
    #         df['overall_std'] = np.sqrt(df['x_std']**2 + df['y_std']**2)

    #         # FINAL MOVEMENT - THEY ARE BOTH QUITE GOOD TBF 

    #         # df['final_movement'] = (df['is_moving']) | ((df['overall_std'] > 0.09) & (df['cumulative_displacement_rate'] > 0.1))
    #         df['final_movement'] = (df['cumulative_displacement_rate'] > 0.05) | ((df['overall_std'] > 0.09) & (df['is_moving']))

    #         # SMOOTH ROLLING WINDOW FOR FINAL MOVEMENT 

    #         # Apply a rolling window with majority voting to smooth out the 'final_movement' column
    #         window_size = 20 # Adjust the window size as needed
    #         df['smoothed_final_movement'] = (df['final_movement']
    #                                          .rolling(window=window_size, center=True) # centre rolling window
    #                                          .apply(lambda x: x.sum() >= (window_size / 2)) # Majority 
    #                                          .fillna(0) # start and end fill with 0 = False
    #                                          .astype(bool)) # all returned True/False


    #         # df.to_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/test-number-digging/withrollingwindow.csv')

    #         df['count'] = total_larvae


    #         if perimeter: # ACOUNTS FOR NO PERIMETER FILES 
    #             # IF PERIMETER FILES BAD WANT TO IGNORE- ONLY DETECT GOOD ONES 
    #             minx, miny, maxx, maxy = perimeter.bounds
    #             diameter = maxx - minx  
        
    #             if diameter > 89:
    #                 print(track_file)
    #                 df = self.detect_larvae_leaving(df, perimeter, total_larvae)


    #         # Now count the moving frames per frame_idx
    #         moving_counts = df.groupby('frame')['smoothed_final_movement'].sum().reset_index()
    #         # Rename the column for clarity
    #         moving_counts.columns = ['frame', 'moving_count']

    #         full_frame_counts = df[['frame', 'count']].drop_duplicates().merge(moving_counts, on='frame', how='left')
    #         full_frame_counts['moving_count'] = full_frame_counts['moving_count'].fillna(0).astype(int)

    #         full_frame_counts.loc[full_frame_counts['moving_count'] > total_larvae, 'moving_count'] = total_larvae

    #         full_frame_counts = full_frame_counts.sort_values(by='frame', ascending=True)

    #         full_frame_counts['number_digging'] = full_frame_counts['count'] - full_frame_counts['moving_count']

    #         full_frame_counts.loc[full_frame_counts['number_digging'] < 0, 'number_digging'] = full_frame_counts['count']

    #         full_frame_counts['file'] = track_file
    #         print(track_file)
    #         full_frame_counts['normalised_digging'] = (full_frame_counts['number_digging'] / total_larvae) * 100


    #         dataframe_list.append(full_frame_counts)
        
    #     number_digging = pd.concat(dataframe_list, ignore_index=True)
    #     number_digging = number_digging.sort_values(by=['frame'], ascending=True)
    #     number_digging.to_csv(os.path.join(self.directory, 'number_digging.csv'), index=False)

    #     return number_digging
    


    # METHOD COMPUTE_DIGGING: THIS METHOD DETECTS IF LARVAE ARE DIGGING (IN ABSENCE OF MAN-MADE HOLE)

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


    # METHOD TOTAL_DIGGING: THIS METHOD DETECTS HOW MANY LARVAE ARE DIGGING 

    def total_digging(self, total_larvae=None, cleaned=False):

        data = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file]
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
            summary['file'] = track_file
            data.append(summary)


        result = pd.concat(data, ignore_index=True)
        result = result.sort_values(by='frame', ascending=True)

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"number_digging{suffix}.csv"

        result.to_csv(os.path.join(self.directory, filename), index=False)
        return result
    

    ### METHOD DIGGING_BEHAVIOUR:

    def digging_behaviour(self):

        single_larvae = []
        two_larvae = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file]
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
                            'file': track_file,
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
                        'file': track_file,
                        'digger_id_1': ids[0],
                        'digger_id_2': ids[1],
                        'distance': dist
                    })

        df_single = pd.DataFrame(single_larvae)    
        df_two = pd.DataFrame(two_larvae)

        df_single.to_csv(os.path.join(self.directory, 'digging_distances_single.csv'), index=False)
        df_two.to_csv(os.path.join(self.directory, 'digging_distances_pair.csv'), index=False)





    ### METHOD
    def detect_larvae_leaving(self, df, perimeter, total_larvae):
        
        df['outside_perimeter'] = df.apply(lambda row: not perimeter.contains(Point(row['x_body'], row['y_body'])),axis=1)

        df = df.sort_values('frame').reset_index(drop=True)
        df['frame'] = df['frame'].astype(int)
        df['track_id'] = df['track_id'].astype(int)

        df['track_count'] = df.groupby('frame')['track_id'].transform('nunique')

        def update_larvae_count(df):
            # Iterate over each row that is marked as outside the perimeter
            for index, row in df[df['outside_perimeter']].iterrows():

                end_frame = row['frame'] + 40
                subsequent_data = df[(df['track_id'] == row['track_id']) & (df['frame'] > row['frame']) & (df['frame'] <= end_frame)]
          
                ## CREATING DF TO ACCESS 1 ROW PER FRAME FOR EASE
                # Drop duplicates based specifically on the 'frame' column
                track_data = df[['frame', 'track_count']].drop_duplicates(subset='frame').reset_index(drop=True)
    
                track_data['rolling_track_count'] = track_data['track_count'].transform(lambda x: x.rolling(window=10).mean())
                track_data.to_csv('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-leaving-perimeter/2025-01-20-n10-agarose/df.csv',  index=False)

                frame = row['frame']
                

                after_frame = track_data.loc[track_data['frame'] == frame, 'rolling_track_count']
                before_frame = track_data.loc[track_data['frame'] == frame -1, 'rolling_track_count']

                if not after_frame.empty and not before_frame.empty:
                    
                    before_count = before_frame.iloc[0]
                    after_count = after_frame.iloc[0]
                else:
                    continue

                if subsequent_data.empty  and (after_count < before_count):
        
                    if after_count >= (total_larvae - 0.2):
                        continue
              
                    print(f'{before_count} and {after_count}')
                    print(f"Larva with track ID {row['track_id']} left the perimeter at frame {row['frame']}.")
                    df.loc[df['frame'] >= row['frame'], 'count'] -= 1
                        # If there is subsequent data, assume the larva could potentially return
                else:
                    continue  # This continues to the next larva without adjusting the count
        

        update_larvae_count(df)

        full_frame_range = range(0, 3600)  # From 0 to 3600
        existing_frames = set(df['frame'].unique())
        missing_frames = sorted(set(full_frame_range) - existing_frames)
        missing_data = [{'frame': frame, 'count': 0} for frame in missing_frames]
        df_missing = pd.DataFrame(missing_data)
        # Append missing data to the original DataFram
        df = pd.concat([df, df_missing], ignore_index=True)
        # Sort the DataFrame by frame to maintain chronological order
        df.sort_values(by='frame', inplace=True)
        # Optional: Reset index for cleanliness
        df.reset_index(drop=True, inplace=True)

        # df_path = '/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-leaving-perimeter/2025-01-20-n10-agarose/df.csv'
        # df.to_csv(df_path, index=False)
        return df
    
    ### METHOD QUALITY_CONTROL: QUALITY CONTROL TO ASSESS PREDICTION AND TRACK QUALITY

    def quality_control(self):

        data = []   

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file]

            perimeter = match.get('perimeter_polygon')

            ### CREATE FOLDER WITH FILENAME 
            file_name = track_file.replace(".tracks.feather", "")
            folder_path = os.path.join(self.directory, file_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)  


            ### COUNT UNIQUE TRACKS PER FRAME
            all_frames = pd.Series(index=range(0, 3601))

            track_counts = df.groupby('frame')['track_id'].nunique()
            track_counts = all_frames.combine_first(track_counts).fillna(0)
            
            ### COUNT UNIQUE POST PROCESSED TRACKS PER FRAME 
            df_tracks = df[df.groupby('track_id')['instance_score'].transform('mean') >= 0.9]
            track_counts_score = df_tracks.groupby('frame')['track_id'].nunique()
            track_counts_score = all_frames.combine_first(track_counts_score).fillna(0)

            plt.figure(figsize=(10, 6))
            sns.lineplot(data=track_counts, label='track number', alpha=0.5)
            sns.lineplot(data=track_counts_score, label='post-procesed track number')
            plt.title(f"Number of Tracks per Frame")
            plt.xlabel("Frame")
            plt.ylabel("Number of Track IDs")
            plt.tight_layout()
            plot_path = os.path.join(folder_path, "number_of_tracks.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()  

            ### FIRST AND LAST FRAME OFF EVERY TRACK 
            track_first_last_df = df.groupby('track_id').agg(first_frame=('frame', 'min'), last_frame=('frame', 'max')).reset_index()
            csv_path = os.path.join(folder_path, "track_first_last_frames.csv")
            track_first_last_df.to_csv(csv_path ,index=False)

            ### CREATE PLOTS FOR BODY X,Y COORDINATES OF EACH TRACK TRAJECTORY 
            
            for track_id, track_data in df.groupby('track_id'):
                plt.figure(figsize=(8, 6))
                plt.plot(track_data['x_body'], track_data['y_body']) 
                plt.title(f"Track {track_id}: Body Coordinates")
                plt.xlabel("X Body")
                plt.ylabel("Y Body")
                plt.xlim(0,122)
                plt.ylim(0,122)
                track_plot_path = os.path.join(folder_path, f"track_{track_id}.png")
                plt.tight_layout()
                plt.savefig(track_plot_path, dpi=300, bbox_inches='tight')
                plt.close()

            
            ### IDENTIFY TRACK JUMPS
            df['dx'] = df.groupby('track_id')['x_body'].diff().fillna(0) 
            df['dy'] = df.groupby('track_id')['y_body'].diff().fillna(0) 
            df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2) 

            df['track_jumps'] = df.groupby('track_id')['distance'].transform(lambda x: x > 2.5) # jumped if greater than 2.5mm 

            track_jumps = df[df['track_jumps']].copy()
            track_jump_path = os.path.join(folder_path, 'potential_track_jumps.csv')
            track_jumps.to_csv(track_jump_path, index=False)

            ### PERIMETER FILE
            perimeter_detected = "Yes" if perimeter is not None else "No"
            perimeter_size = "Correct"

            if perimeter: 
                minx, miny, maxx, maxy = perimeter.bounds
                diameter = maxx - minx  
        
                if diameter < 89:
                    perimeter_size = "Small"
            
            ### DETECT LARVAE OUTSIDE THE PERIMETER
            if diameter > 89:
                df['outside_perimeter'] = df.apply(lambda row: not perimeter.contains(Point(row['x_body'], row['y_body'])),axis=1)

                outside_perimeter = df[df['outside_perimeter']].copy()
                path = os.path.join(folder_path, 'outside_perimeter.csv')
                outside_perimeter.to_csv(path, index=False)

                outside_perimeter_number = outside_perimeter.shape[0]
            
            ### META DATA FOR DIRECTORY
            total_tracks = df['track_id'].nunique()
            track_jump_number = track_jumps.shape[0]
            track_lengths = track_first_last_df['last_frame'] - track_first_last_df['first_frame'] #df created above
            average_track_length = track_lengths.mean()

            data.append({'file':file_name, 'total tracks': total_tracks, 'average track length': average_track_length, 'track jumps': track_jump_number, 'perimeter detected': perimeter_detected, 'perimeter size': perimeter_size, 'outside perimeter': outside_perimeter_number})
    

        summary_df = pd.DataFrame(data)
        summary_path = os.path.join(self.directory, "summary.csv")
        summary_df.to_csv(summary_path, index=False)

    # METHOD PSEUDO_POPULATION_MODEL:

    def pseudo_population_model(self, number_of_iterations, number_of_animals):

        ### GENERATE LIST OF NORMALISED TRACK FILES 

        data = []   

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file]
    
            perimeter = match.get('perimeter_polygon')

            if perimeter:
                centroid = perimeter.centroid
                centroid_x = centroid.x
                centroid_y = centroid.y
 
                ## for every body coordinate need to minus this centroid 
                body_coordinates = ['x_tail', 'y_tail', 'x_body', 'y_body', 'x_head', 'y_head']
                for coord in body_coordinates:
                    if 'x' in coord:
                        df[coord] = df[coord] - centroid_x
                    elif 'y' in coord:
                        df[coord] = df[coord] - centroid_y

                df['filename'] = track_file
                    
                data.append(df)
       
            else:
                continue 
        
        for iteration in range(number_of_iterations):
            selected_files = random.sample(data, number_of_animals)

            renamed_files = []
            for i, df in enumerate(selected_files):
                df_new = df.copy()             # make a safe copy
                df_new['track_id'] = i         # assign unique ID
                renamed_files.append(df_new)

            concatenated_df = pd.concat(renamed_files, ignore_index=True)
            concatenated_df = concatenated_df.sort_values(by='frame', ascending=True)

            filepath = os.path.join(self.directory, f'pseudo_population_{iteration+1}.csv')
            concatenated_df.to_csv(filepath, index=False)


    ### METHOD INTERACTION_TYPES: COUNT DIFFERENT TYPES OF PROXIMAL INTERACTIONS BETWEEN LARVAE (1MM THRESHOLD) (ALL CONTACTS PER FRAME)
    def interaction_types(self, threshold=1):
        def unify_interaction_type(part1, part2):
            return '_'.join(sorted([part1, part2]))

        data = []

        for track_file in self.track_files:
            df = self.track_data[track_file]
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
                    'file': track_file,
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

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file].sort_values(by='frame')

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

    


    ### METHOD INTERACTION_TYPE_BOUT: WITHIN <1MM NODE-NODE BOUTS IDENTIFIES THE INITIAL AND PREDOMINANT NODE-NODE CONTACT

    # def interaction_type_bout(self):

    #     threshold=1.0  # distance threshold for interaction
    #     continue_threshold = 1.5 # allow some leeway for continuing bouts 

    #     def unify_interaction_type(part1, part2):
    #         return '_'.join(sorted([part1, part2]))
        

    #     def get_closest_part_pair(coords, id1, id2): 
    #         min_dist = float('inf')
    #         closest_type = None
    #         for part1, part2 in interaction_pairs:
    #             coord1 = coords[part1].get(id1)
    #             coord2 = coords[part2].get(id2)
    #             if coord1 is None or coord2 is None:
    #                 continue
    #             dist = np.linalg.norm(coord1 - coord2)
    #             if dist < min_dist:
    #                 min_dist = dist
    #                 closest_type = unify_interaction_type(part1, part2)
    #         return closest_type

    #     body_parts = ['head', 'body', 'tail']
    #     interaction_pairs = list(itertools.product(body_parts, body_parts))

    #     unified_types = [
    #             'head_head', 'tail_tail', 'body_body',
    #             'body_head', 'body_tail', 'head_tail'
    #         ]

    #     bouts = []
        
    #     for track_file in self.track_files:
    #         df = self.track_data[track_file].copy()
    #         df.sort_values(by='frame', inplace=True)

    #         # Keep track of active bouts per unique (track1, track2) pair
    #         active_bouts = {}  # key: (id1, id2), value: dict with bout info
    #         bout_counter = 0

    #         for frame in df['frame'].unique():
    #             frame_data = df[df['frame'] == frame]
    #             track_ids = frame_data['track_id'].unique()

    #             # Build coordinate lookups for each part
    #             coords = {
    #                 part: {
    #                     row['track_id']: np.array([row[f'x_{part}'], row[f'y_{part}']])
    #                     for _, row in frame_data.iterrows()
    #                 }
    #                 for part in body_parts
    #             }

    #             interacting_pairs = {}  # key: (id1, id2), value: list of interaction_types this frame

    #             for id1, id2 in itertools.combinations(track_ids, 2): # loops through all pairs of tracks 
    #                 interactions = []

    #                 for part1, part2 in interaction_pairs: # loops through all pairs of body parts
    #                     coord1 = coords[part1].get(id1)
    #                     coord2 = coords[part2].get(id2)
    #                     if coord1 is None or coord2 is None:
    #                         continue

    #                     dist = np.linalg.norm(coord1 - coord2)
    #                     if dist < threshold:
    #                         interaction_type = unify_interaction_type(part1, part2)
    #                         interactions.append(interaction_type)

    #                 if interactions:
    #                     pair_key = tuple(sorted((id1, id2)))
    #                     interacting_pairs[pair_key] = interactions    


    #                 ### from here new
                
    #             current_pairs = set(interacting_pairs.keys())

    #             # Handle previously active pairs
    #             for pair in list(active_bouts.keys()):
    #                 if pair not in current_pairs:
    #                     # No direct interaction this frame
    #                     active_bouts[pair]['gap_count'] += 1
    #                     if active_bouts[pair]['gap_count'] <= max_gap:
    #                         id1, id2 = pair
    #                         fallback = get_closest_part_pair(coords, id1, id2)
    #                         if fallback:
    #                             active_bouts[pair]['interactions'].append(fallback)
    #                     else:
    #                         # Too many missed frames → end bout
    #                         bout = active_bouts.pop(pair)
    #                         start, end = bout['start_frame'], bout['end_frame']
    #                         interactions = bout['interactions']
    #                         if interactions:
    #                             type_counts = Counter(interactions)
    #                             bout_data = {
    #                                 'file': track_file,
    #                                 'bout_id': bout['bout_id'],
    #                                 'track_1': pair[0],
    #                                 'track_2': pair[1],
    #                                 'start_frame': start,
    #                                 'end_frame': end,
    #                                 'duration': end - start + 1,
    #                                 'initial_type': interactions[0],
    #                                 'predominant_type': Counter(interactions).most_common(1)[0][0]}
    #                             for t in unified_types:
    #                                 bout_data[f'{t}'] = type_counts.get(t, 0)
    #                             bouts.append(bout_data)

    #             # Update or start new bouts
    #             for pair, interactions in interacting_pairs.items():
    #                 if pair in active_bouts:
    #                     active_bouts[pair]['end_frame'] = frame
    #                     active_bouts[pair]['interactions'].extend(interactions)
    #                     active_bouts[pair]['gap_count'] = 0
    #                 else:
    #                     active_bouts[pair] = {
    #                         'bout_id': bout_counter,
    #                         'start_frame': frame,
    #                         'end_frame': frame,
    #                         'interactions': interactions.copy(),
    #                         'gap_count': 0
    #                     }
    #                     bout_counter += 1

    #         # Finalize remaining bouts
    #         for pair, bout in active_bouts.items():
    #             interactions = bout['interactions']
    #             if interactions:
    #                 type_counts = Counter(interactions)
    #                 bout_data = {
    #                     'file': track_file,
    #                     'bout_id': bout['bout_id'],
    #                     'track_1': pair[0],
    #                     'track_2': pair[1],
    #                     'start_frame': bout['start_frame'],
    #                     'end_frame': bout['end_frame'],
    #                     'duration': bout['end_frame'] - bout['start_frame'] + 1,
    #                     'initial_type': interactions[0],
    #                     'predominant_type': Counter(interactions).most_common(1)[0][0]
    #                 }
    #                 for t in unified_types:
    #                     bout_data[f'{t}'] = type_counts.get(t, 0)

    #                 bouts.append(bout_data)

    #     bout_df = pd.DataFrame(bouts).sort_values(by=['file', 'bout_id'])
    #     bout_df.to_csv(os.path.join(self.directory, "interaction_type_bout.csv"), index=False)
    #     return bout_df


    def interaction_type_bout(self):

        threshold = 1.0           # must hit this to START a bout
        continue_threshold = 1.5  # once started, can CONTINUE while min_dist < this

        def unify_interaction_type(part1, part2):
            return '_'.join(sorted([part1, part2]))

        body_parts = ['head', 'body', 'tail']
        interaction_pairs = list(itertools.product(body_parts, body_parts))

        unified_types = [
            'head_head', 'tail_tail', 'body_body',
            'body_head', 'body_tail', 'head_tail'
        ]

        bouts = []

        for track_file in self.track_files:
            df = self.track_data[track_file].copy()
            df.sort_values(by='frame', inplace=True)

            active_bouts = {}  # key: (id1, id2) -> bout dict
            bout_counter = 0

            for frame in df['frame'].unique():
                frame_data = df[df['frame'] == frame]
                track_ids = frame_data['track_id'].unique()

                # Build coordinate lookups for each part
                coords = {
                    part: {
                        row['track_id']: np.array([row[f'x_{part}'], row[f'y_{part}']])
                        for _, row in frame_data.iterrows()
                    }
                    for part in body_parts
                }

                # pairs with any <1mm contacts this frame (used to START bouts + log real interactions)
                interacting_pairs = {}  # pair_key -> list of interaction types (<1mm)

                # pairs with min distance <1.5mm this frame (used to CONTINUE bouts)
                close_pairs = {}        # pair_key -> closest_type (min-distance type)

                for id1, id2 in itertools.combinations(track_ids, 2):

                    interactions = []
                    min_dist = float('inf')
                    closest_type = None

                    for part1, part2 in interaction_pairs:
                        coord1 = coords[part1].get(id1)
                        coord2 = coords[part2].get(id2)
                        if coord1 is None or coord2 is None:
                            continue

                        dist = np.linalg.norm(coord1 - coord2)

                        # track minimum distance + its type
                        if dist < min_dist:
                            min_dist = dist
                            closest_type = unify_interaction_type(part1, part2)

                        # record all true contact types (<1mm)
                        if dist < threshold:
                            interactions.append(unify_interaction_type(part1, part2))

                    pair_key = tuple(sorted((id1, id2)))

                    # continuation condition: within 1.5mm
                    if closest_type is not None and min_dist < continue_threshold:
                        close_pairs[pair_key] = closest_type

                    # start/true-contact condition: any <1mm
                    if interactions:
                        interacting_pairs[pair_key] = interactions

                current_close = set(close_pairs.keys())

                # 1) END bouts that are no longer within 1.5mm
                for pair in list(active_bouts.keys()):
                    if pair not in current_close:
                        bout = active_bouts.pop(pair)
                        interactions_all = bout['interactions']
                        if interactions_all:
                            type_counts = Counter(interactions_all)
                            bout_data = {
                                'file': track_file,
                                'bout_id': bout['bout_id'],
                                'track_1': pair[0],
                                'track_2': pair[1],
                                'start_frame': bout['start_frame'],
                                'end_frame': bout['end_frame'],
                                'duration': bout['end_frame'] - bout['start_frame'],
                                'initial_type': interactions_all[0],
                                'predominant_type': Counter(interactions_all).most_common(1)[0][0],
                            }
                            for t in unified_types:
                                bout_data[t] = type_counts.get(t, 0)
                            bouts.append(bout_data)

                # 2) UPDATE existing bouts that are still within 1.5mm
                for pair in list(active_bouts.keys()):
                    # (pair must be in close_pairs here)
                    active_bouts[pair]['end_frame'] = frame

                    if pair in interacting_pairs:
                        # real interactions (<1mm)
                        active_bouts[pair]['interactions'].extend(interacting_pairs[pair])
                    else:
                        # between 1.0 and 1.5mm: filler closest type
                        active_bouts[pair]['interactions'].append(close_pairs[pair])


                # 3) START new bouts ONLY if they hit <1mm this frame
                for pair, interactions in interacting_pairs.items():
                    if pair in active_bouts:
                        continue
                    active_bouts[pair] = {
                        'bout_id': bout_counter,
                        'start_frame': frame,
                        'end_frame': frame,
                        'interactions': interactions.copy(),
                    }
                    bout_counter += 1

            # Finalize remaining bouts at end of file
            for pair, bout in active_bouts.items():
                interactions_all = bout['interactions']
                if interactions_all:
                    type_counts = Counter(interactions_all)
                    bout_data = {
                        'file': track_file,
                        'bout_id': bout['bout_id'],
                        'track_1': pair[0],
                        'track_2': pair[1],
                        'start_frame': bout['start_frame'],
                        'end_frame': bout['end_frame'],
                        'duration': bout['end_frame'] - bout['start_frame'] + 1,
                        'initial_type': interactions_all[0],
                        'predominant_type': Counter(interactions_all).most_common(1)[0][0],
                    }
                    for t in unified_types:
                        bout_data[t] = type_counts.get(t, 0)
                    bouts.append(bout_data)

        bout_df = pd.DataFrame(bouts).sort_values(by=['file', 'bout_id'])
        bout_df.to_csv(os.path.join(self.directory, "interaction_type_bout.csv"), index=False)
        return bout_df





        #         # Update active bouts
        #         ended_pairs = set(active_bouts.keys()) - set(interacting_pairs.keys()) # this frame interacting pairs - old frames interacting pairs- if not present, the bout ended

        #         # Process ended bouts
        #         for pair in ended_pairs:
        #             bout = active_bouts.pop(pair)
        #             start, end = bout['start_frame'], bout['end_frame']
        #             duration = end - start + 1
        #             interactions = bout['interactions']

        #             initial_type = interactions[0]
        #             predominant_type = Counter(interactions).most_common(1)[0][0]

        #             bouts.append({
        #                 'file': track_file,
        #                 'bout_id': bout['bout_id'],
        #                 'track_1': pair[0],
        #                 'track_2': pair[1],
        #                 'start_frame': start,
        #                 'end_frame': end,
        #                 'duration': duration,
        #                 'initial_type': initial_type,
        #                 'predominant_type': predominant_type,
        #             })

        #         # Update or start new bouts
        #         for pair, interactions in interacting_pairs.items():
        #             if pair in active_bouts:
        #                 active_bouts[pair]['end_frame'] = frame
        #                 active_bouts[pair]['interactions'].extend(interactions)
        #             else:
        #                 active_bouts[pair] = {
        #                     'bout_id': bout_counter,
        #                     'start_frame': frame,
        #                     'end_frame': frame,
        #                     'interactions': interactions.copy(),
        #                 }
        #                 bout_counter += 1

        #     # Handle bouts that were still active at the end
        #     for pair, bout in active_bouts.items():
        #         start, end = bout['start_frame'], bout['end_frame']
        #         duration = end - start + 1
        #         interactions = bout['interactions']

        #         initial_type = interactions[0]
        #         predominant_type = Counter(interactions).most_common(1)[0][0]

        #         bouts.append({
        #             'file': track_file,
        #             'bout_id': bout['bout_id'],
        #             'track_1': pair[0],
        #             'track_2': pair[1],
        #             'start_frame': start,
        #             'end_frame': end,
        #             'duration': duration,
        #             'initial_type': initial_type,
        #             'predominant_type': predominant_type,
        #         })

        # # Output dataframe
        # bout_df = pd.DataFrame(bouts).sort_values(by=['file', 'bout_id'])

        # # Optional saving
        # filename = "interaction_type_bout.csv"
        # bout_df.to_csv(os.path.join(self.directory, filename), index=False)

        # return bout_df
    


    # METHOD INTERACTION_BOUT_DYNAMICS: 

    def interaction_bout_dynamics(self): ### method above must be run already 

        bout_df = pd.read_csv(os.path.join(self.directory, "interaction_type_bout.csv"))

        # Melt into long form: one row per larva per bout
        larva_rows = []
        for _, row in bout_df.iterrows():
            for role in ['track_1', 'track_2']:
                larva_id = row[role]
                partner_id = row['track_2'] if role == 'track_1' else row['track_1'] # want both prospectives 
                larva_rows.append({
                    'file': row['file'],
                    'larva_id': larva_id,
                    'partner_id': partner_id,
                    'start_frame': row['start_frame'],
                    'end_frame': row['end_frame'],
                    'duration': row['duration'],
                    'initial_type': row['initial_type'],
                    'predominant_type': row['predominant_type']
                    # 'original_bout_id': row['bout_id']
                })

        df = pd.DataFrame(larva_rows)
        df.sort_values(by=['file', 'larva_id', 'start_frame'], inplace=True)

        df['bout_number'] = df.groupby(['file', 'larva_id']).cumcount() + 1 #bout id per larva 
        df['time_since_last_bout'] = df.groupby(['file', 'larva_id'])['start_frame'].diff().fillna(pd.NA) # time since last bout

        # Previous partner
        df['prev_partner'] = df.groupby(['file', 'larva_id'])['partner_id'].shift()
        df['same_partner'] = df['partner_id'] == df['prev_partner']
        # df['same_partner'] = df['same_partner'].map({True: 'yes', False: 'no', pd.NA: pd.NA})

        # Save bout-level breakdown (optional, comment out if not wanted)
        out_file = os.path.join(self.directory, "inter_bout_dynamics.csv")
        df.to_csv(out_file, index=False)


    ### METHOD CONTACTS: IDENTIFY INTERACTION FREQUENCY AND DURATION

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


        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file]
            df = df.sort_values(by='frame', ascending=True)
            df['filename'] = track_file

            track_ids = df['track_id'].unique()
            track_combinations = list(combinations(track_ids, 2))

            all_results = Parallel(n_jobs=-1)(
                delayed(process_track_pair)(track_a, track_b, df, track_file, proximity_threshold)
                for track_a, track_b in track_combinations
            )

            flattened_results = [item for sublist in all_results for item in sublist]
            if not flattened_results:
                print(f"No contact results for {track_file}")
                no_contacts.append(track_file)
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





        





        # METHOD INITIAL_HOLE_FORMATION: TIME AT WHICH THE FIRST LARVAE BEGINS DIGGING
        # EXTRACTED FROM THE ABOVE !

#### old

    ### METHOD CORRELATIONS: QUANTIFY RELATIONSHIPS BETWEEN NEAREST NEIGHOUR DISTANCE AND SPEED ETC 
    # def nearest_neighbour(self):

    #     dfs = []
        
    #     for match in self.matching_pairs:
    #         track_file = match['track_file']
    #         df = self.track_data[track_file]

    #         df = df.sort_values(by='frame', ascending=True)
    #         df['filename'] = track_file
            
    #         # df here with speed, acceleration, angles and distance to nearest larva

    #         def speed(group, x, y):
    #             dx = group[x].diff()
    #             dy = group[y].diff()
    #             distance = np.sqrt(dx**2 + dy**2)
    #             dt = group['frame'].diff()
    #             speed = distance / dt.replace(0, np.nan) # Avoid division by zero
    #             return speed

    #         df['speed'] = df.groupby('track_id').apply(lambda group: speed(group, 'x_body', 'y_body')).reset_index(level=0, drop=True)
    #         df['acceleration'] = df.groupby('track_id')['speed'].diff() / df.groupby('track_id')['frame'].diff()
       

    #         def calculate_angle(df, v1_x, v1_y, v2_x, v2_y):
    #             dot_product = (df[v1_x] * df[v2_x]) + (df[v1_y] * df[v2_y])
    #             magnitude_v1 = np.hypot(df[v1_x], df[v1_y])  # Same as sqrt(x^2 + y^2
    #             magnitude_v2 = np.hypot(df[v2_x], df[v2_y])

    #             # Avoid division by zero
    #             cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    #             cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure values are in valid range for arccos
                
    #             return np.degrees(np.arccos(cos_theta))  # Convert radians to degrees
            
    #         df['v1_x'] = df['x_head'] - df['x_body']
    #         df['v1_y'] = df['y_head'] - df['y_body']
    #         df['v2_x'] = df['x_tail'] - df['x_body']
    #         df['v2_y'] = df['y_tail'] - df['y_body']

    #         # Apply function correctly
    #         df['angle'] = calculate_angle(df, 'v1_x', 'v1_y', 'v2_x', 'v2_y')


    #         node_list = ['head', 'body', 'tail']

    #         for frame in df['frame'].unique():
    #             frame_data = df[df['frame'] == frame]
    #             if len(frame_data) < 2:
    #                 continue

    #             n = len(frame_data)
    #             track_ids = frame_data['track_id'].to_numpy()
    #             index = frame_data.index.to_numpy()
                
    #             ### these are blank 
    #             min_dists = np.full(n, np.inf)
    #             best_pairs = np.empty(n, dtype=object)
    #             contact_ids = np.full(n, np.nan)

    #             for part1, part2 in product(node_list, repeat=2):
    #                 coords1 = frame_data[[f'x_{part1}', f'y_{part1}']].to_numpy()
    #                 coords2 = frame_data[[f'x_{part2}', f'y_{part2}']].to_numpy()

    #                 dist_matrix = cdist(coords1, coords2)
    #                 np.fill_diagonal(dist_matrix, np.inf)

    #                 min_idx = np.argmin(dist_matrix, axis=1)
    #                 min_val = np.min(dist_matrix, axis=1)

    #                 # Update only where this pairing is the best so far
    #                 update_mask = min_val < min_dists # this is boolean masking 
    #                 min_dists[update_mask] = min_val[update_mask]
    #                 num_updates = np.sum(update_mask)
    #                 best_pairs[update_mask] = [f"{part1}-{part2}"] * num_updates
    #                 contact_ids[update_mask] = track_ids[min_idx[update_mask]]

    #             # Save to main df
    #             df.loc[index, 'node_distance'] = min_dists
    #             df.loc[index, 'node-node'] = best_pairs
    #             df.loc[index, 'contact_track'] = contact_ids


    #         dfs.append(df)
    #             # df.to_csv(os.path.join(self.directory, 'df.csv'), index=False)
        
    #     data = pd.concat(dfs, ignore_index=True)

    #     if self.shorten and self.shorten_duration is not None:
    #         suffix = f"_{self.shorten_duration}"
    #     else:
    #         suffix = ""

    #     filename = f"nearest_neighbour{suffix}.csv"
    #     data.to_csv(os.path.join(self.directory, filename), index=False)




    # def nearest_neighbour(self):

    #     dfs = []

        
    #     for match in self.matching_pairs:
    #         track_file = match['track_file']
    #         df = self.track_data[track_file]

    #         df = df.sort_values(by='frame', ascending=True)
    #         df['filename'] = track_file
            
    #         # df here with speed, acceleration, angles and distance to nearest larva

    #         def speed(group, x, y):
    #             dx = group[x].diff()
    #             dy = group[y].diff()
    #             distance = np.sqrt(dx**2 + dy**2)
    #             dt = group['frame'].diff()
    #             speed = distance / dt.replace(0, np.nan) # Avoid division by zero
    #             return speed

    #         df['speed'] = df.groupby('track_id').apply(lambda group: speed(group, 'x_body', 'y_body')).reset_index(level=0, drop=True)
    #         df['acceleration'] = df.groupby('track_id')['speed'].diff() / df.groupby('track_id')['frame'].diff()
       

    #         def calculate_angle(df, v1_x, v1_y, v2_x, v2_y):
    #             dot_product = (df[v1_x] * df[v2_x]) + (df[v1_y] * df[v2_y])
    #             magnitude_v1 = np.hypot(df[v1_x], df[v1_y])  # Same as sqrt(x^2 + y^2
    #             magnitude_v2 = np.hypot(df[v2_x], df[v2_y])

    #             # Avoid division by zero
    #             cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    #             cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure values are in valid range for arccos
                
    #             return np.degrees(np.arccos(cos_theta))  # Convert radians to degrees
            
    #         df['v1_x'] = df['x_head'] - df['x_body']
    #         df['v1_y'] = df['y_head'] - df['y_body']
    #         df['v2_x'] = df['x_tail'] - df['x_body']
    #         df['v2_y'] = df['y_tail'] - df['y_body']

    #         # Apply function correctly
    #         df['angle'] = calculate_angle(df, 'v1_x', 'v1_y', 'v2_x', 'v2_y')

    #         df['body-body'] = np.nan 


    #         for frame in df['frame'].unique():
    #             unique_frame =  df[df['frame'] == frame]
    #             if len(unique_frame) < 2:
    #                 continue

    #             body_coordinates = unique_frame[['x_body', 'y_body']].to_numpy()
    #             distance = cdist(body_coordinates, body_coordinates, 'euclidean')
    #             np.fill_diagonal(distance, np.nan)

    #             # unique_frame['body-body'] = np.nanmin(distance, axis=1)
    #             df.loc[unique_frame.index, 'body-body'] = np.nanmin(distance, axis=1)


    #         dfs.append(df)
    #             # df.to_csv(os.path.join(self.directory, 'df.csv'), index=False)
        
    #     data = pd.concat(dfs, ignore_index=True)

    #     if self.shorten and self.shorten_duration is not None:
    #         suffix = f"_{self.shorten_duration}"
    #     else:
    #         suffix = ""

    #     filename = f"nearest_neighbour{suffix}.csv"
    #     data.to_csv(os.path.join(self.directory, filename), index=False)





    def nearest_neighbour(self):

        dfs = []

        parts = ['head', 'body', 'tail']

        def unify_interaction_type(p1, p2):
            return '-'.join(sorted([p1, p2]))

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file]

            df = df.sort_values(by='frame', ascending=True)
            df['filename'] = track_file

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
                # coords = nodes[['x', 'y']].to_numpy(float) ##
                # D = cdist(coords, coords) ##

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


    #### DEF potential_interactions


    def potential_interactions(self, threshold=10.0):

        parts = ['head', 'body', 'tail']

        def unify_interaction_type(p1, p2):
            return '-'.join(sorted([p1, p2]))

        def min_node_distance(row_a, row_b):

            best_d = np.inf
            best_pair = None

            for p1 in parts:
                x1 = row_a.get(f"x_{p1}", np.nan)
                y1 = row_a.get(f"y_{p1}", np.nan)
                if pd.isna(x1) or pd.isna(y1):
                    continue

                for p2 in parts:
                    x2 = row_b.get(f"x_{p2}", np.nan)
                    y2 = row_b.get(f"y_{p2}", np.nan)
                    if pd.isna(x2) or pd.isna(y2):
                        continue

                    d = np.hypot(x1 - x2, y1 - y2)
                    if d < best_d:
                        best_d = d
                        best_pair = unify_interaction_type(p1, p2)

            return best_d, best_pair


        all_events = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file].copy()

            df = df.sort_values('frame')
            df['filename'] = track_file

            frame_groups = dict(tuple(df.groupby('frame')))
            frames = sorted(frame_groups.keys())

            # per-frame dict: {frame: {track_id: row}}
            per_frame = {}
            for frame_key, frame_df in frame_groups.items():
                per_frame[frame_key] = {
                    track_id: row
                    for track_id, row in zip(frame_df['track_id'], frame_df.itertuples(index=False))
                }

            track_ids = sorted(df['track_id'].unique())

 
            ### ITERATE THROUGH TRACK PAIRS
            for i in range(len(track_ids)):
                for j in range(i + 1, len(track_ids)):
                    id1, id2 = track_ids[i], track_ids[j]

                    in_encounter = False
                    d_start = None
                    t_start = None
                    start_pair = None
                    
                    ### ITERATE THROUGH FRAMES
                    for frame in frames:
                        rows = per_frame.get(frame, {})
                        if id1 not in rows or id2 not in rows:
                            in_encounter = False
                            continue

                        d_min, part_pair = min_node_distance(rows[id1]._asdict(),
                                                            rows[id2]._asdict())

                        if not in_encounter and d_min < threshold:
                                # trigger encounter
                                in_encounter = True
                                t_start = frame
                                d_start = d_min
                                start_pair = part_pair

                                touch = False

                        elif in_encounter and d_min < 1.0:
                            touch = True

                        # 3. END encounter
                        elif in_encounter and d_min > threshold:
                            all_events.append({
                                'filename': track_file,
                                'track1': id1,
                                'track2': id2,
                                'start_frame': t_start,
                                'start_min_dist': float(d_start),
                                'first_nodes': start_pair,
                                'touch': touch
                            })
                            in_encounter = False


        potential_interactions_df = pd.DataFrame(all_events)
        potential_interactions_df.to_csv(os.path.join(self.directory, f"potential_interactions_{threshold}.csv"), index=False)   







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

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file].copy()
            df = df.sort_values('frame')
            df['filename'] = track_file

            perimeter = match.get('perimeter_polygon')
            centre_x, centre_y = perimeter.centroid.x, perimeter.centroid.y

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
                                focal['x_body'] - centre_x,
                                focal['y_body'] - centre_y
                            )


                        # DURING
                        elif in_encounter and d_min < 1.0:
                            touch = True

                        # END
                        elif in_encounter and d_min >= threshold:
                            all_events.append({
                                'filename': track_file,
                                'focal_id': focal_id,
                                'stim_id': stim_id,
                                'start_frame': t_start,
                                'start_min_dist': float(d_start),
                                'first_node': start_node,
                                'touch': touch,
                                'dist_from_centre': float(focal_dist_centre)
                            })
                            in_encounter = False


            # for frame in frames[:-1]:  # need next frame
            #     if frame + 1 not in per_frame:
            #         continue

            #     rows_now = per_frame[frame]
            #     rows_next = per_frame[frame + 1]

            #     for focal_id in rows_now:
            #         if focal_id not in rows_next:
            #             continue

            #         focal = rows_now[focal_id]
            #         focal_next = rows_next[focal_id]

            #         ## LARVAE WITH SMALLEST NODE DISTANCE FROM HEAD AND THEN ONE NODE OF THAT LARVAE MUST BE < 35 DEGREES TO BE INC

            #         # --- Step 1: find the nearest larva (by closest-node distance), regardless of angle ---
            #         nearest = None

            #         for stim_id in rows_now:
            #             if stim_id == focal_id:
            #                 continue
            #             if stim_id not in rows_next:
            #                 continue

            #             stim = rows_now[stim_id]

            #             d_now, stim_node = min_node_distance(focal, stim)
            #             if d_now > 20:
            #                 continue

            #             if d_now < 1.0:
            #                 continue

            #             if (nearest is None) or (d_now < nearest['d_now']):
            #                 nearest = {
            #                     'stim_id': stim_id,
            #                     'd_now': d_now,
            #                     'stim_node': stim_node
            #                 }

            #         # if no neighbour in range, skip
            #         if nearest is None:
            #             continue

            #         stim_id = nearest['stim_id']
            #         stim = rows_now[stim_id]
            #         stim_next = rows_next[stim_id]

            #         d_now = nearest['d_now']
            #         stim_node = nearest['stim_node']

            #         # --- Step 2: visibility gate based on ANY node angle ---
            #         angle, angle_node = min_approach_angle(focal, stim)
            #         if angle is None or np.isnan(angle) or angle > 35:
            #             continue  # nearest neighbour is not in front, skip this frame

                    # ----- SPEED DEVIATION GATE -----

                    # # collect baseline speeds
                    # baseline_speeds = []
                    # for k in (-4, -3, -2, -1):
                    #     fr_k = frame + k
                    #     fr_k_next = fr_k + 1
                    #     if fr_k not in per_frame or fr_k_next not in per_frame:
                    #         baseline_speeds = None
                    #         break
                    #     if focal_id not in per_frame[fr_k] or focal_id not in per_frame[fr_k_next]:
                    #         baseline_speeds = None
                    #         break

                    #     baseline_speeds.append(
                    #         speed(per_frame[fr_k][focal_id], per_frame[fr_k_next][focal_id])
                    #     )

                    # # collect response speeds
                    # response_speeds = []
                    # for k in (1, 2, 3, 4):
                    #     fr_k = frame + k
                    #     fr_k_next = fr_k + 1
                    #     if fr_k not in per_frame or fr_k_next not in per_frame:
                    #         response_speeds = None
                    #         break
                    #     if focal_id not in per_frame[fr_k] or focal_id not in per_frame[fr_k_next]:
                    #         response_speeds = None
                    #         break

                    #     response_speeds.append(
                    #         speed(per_frame[fr_k][focal_id], per_frame[fr_k_next][focal_id])
                    #     )

                    # # if windows incomplete, treat as no decision
                    # decision = 0
                    # if baseline_speeds is not None and response_speeds is not None:
                    #     v_base = np.mean(baseline_speeds)
                    #     v_resp = np.mean(response_speeds)
                    #     v_sd   = np.std(baseline_speeds)

                    #     if v_sd > 0 and abs(v_resp - v_base) >  v_sd:
                    #         decision = 1

                    # if decision == 0:
                    #     all_events.append({
                    #         'filename': track_file,
                    #         'frame': frame,
                    #         'focal_id': focal_id,
                    #         'stim_id': stim_id,
                    #         'distance': d_now,
                    #         'closest_node': stim_node,
                    #         'approach_angle': angle,
                    #         'angle_node': angle_node,
                    #         'decision': 0,
                    #         'outcome': None
                    #     })
                    #     continue



                    # # --- Step 3: outcome to the SAME larva next frame ---
                    # d_next, _ = min_node_distance(focal_next, stim_next)

                    # if d_next < d_now:
                    #     outcome = 'approach'
                    # elif d_next > d_now:
                    #     outcome = 'avoid'
                    # else:
                    #     outcome = 'neutral'

                    # all_events.append({
                    #     'filename': track_file,
                    #     'frame': frame,
                    #     'focal_id': focal_id,
                    #     'stim_id': stim_id,
                    #     'distance': d_now,
                    #     'closest_node': stim_node,
                    #     'approach_angle': angle,
                    #     'angle_node': angle_node,
                    #     'distance_moved': d_next - d_now,
                    #     'outcome': outcome
                    # })

                    # --- Step 3: outcome over next 3 frames (mean distance) and approach angle ---
                    # future_ds = []
                    # future_as = []
                    # for k in (1, 2, 3, 4): # next x frames
                    #     fr_k = frame + k
                    #     if fr_k not in per_frame:
                    #         future_ds = None
                    #         break
                    #     if focal_id not in per_frame[fr_k] or stim_id not in per_frame[fr_k]:
                    #         future_ds = None
                    #         break

                    #     focal_k = per_frame[fr_k][focal_id]
                    #     stim_k = per_frame[fr_k][stim_id]

                    #     d_k, d_node = min_node_distance(focal_k, stim_k)
                    #     if not np.isfinite(d_k):
                    #         future_ds = None
                    #         break

                    #     future_ds.append(d_k)
                        
                        ### CLOSEST NODE APPROACH ANGLE
                        # a_k, _ = min_approach_angle(focal_k, stim_k)
                        # if a_k is None or not np.isfinite(a_k):
                        #     future_ds = None
                        #     break
                        # future_as.append(a_k)
                        
                        ## APPROACH ANGLE OF CLOSEST NODE 
                    #     a_k = approach_angle_to_node(focal_k, stim_k, d_node)
                    #     if np.isnan(a_k):
                    #         future_ds = None
                    #         break
                    #     future_as.append(a_k)

                    # if future_ds is None:
                    #     continue

                    # d_future = float(np.mean(future_ds))
                    # d_future_min = float(np.min(future_ds))
                    # a_future = float(np.mean(future_as))

                    # if (d_future < 1.0) or (d_future_min < 1.0):
                    #     outcome = 'contact'
                    # elif (d_future < d_now) and (a_future <= 10):
                    #     outcome = 'strong approach'
                    # elif (d_future < d_now) and (a_future <= 35):
                    #     outcome = 'approach'
                    # elif (d_future > d_now) and (a_future <= 35):
                    #     outcome = 'facing but still/moving away'
                    # elif (d_future > d_now) and (a_future >= 90):
                    #     outcome = 'strong avoid'
                    # elif (d_future > d_now) and (a_future >= 50):
                    #     outcome = 'avoid'
                    # else:
                    #     outcome = 'neutral'

                    # if (d_future < 1.0) or (d_future_min < 1.0):
                    #     outcome = 'approach'
                    # elif (d_future < d_now) and (a_future <= angle): #35
                    #     outcome = 'approach'
                    # elif (d_future > d_now) and (a_future >= angle): #90
                    #     outcome = 'avoid'
                    # else:
                    #     outcome = 'other'



                    # all_events.append({
                    #     'filename': track_file,
                    #     'frame': frame,
                    #     'focal_id': focal_id,
                    #     'stim_id': stim_id,
                    #     'distance': d_now,
                    #     'closest_node': stim_node,
                    #     'approach_angle': angle,
                    #     'angle_node': angle_node,
                    #     'distance_future_mean': d_future,
                    #     'approach_angle_future_mean': a_future,
                    #     'distance_moved': d_future - d_now,
                    #     'angle_change': a_future - angle,
                    #     'decision': 1,
                    #     'outcome': outcome
                    # })


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

        for match in self.matching_pairs:

            track_file = match['track_file']
            df = self.track_data[track_file].copy()
            df = df.sort_values('frame')

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
                                'filename': track_file,
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











        

            








        


    ### METHOD INTERACTIONS: IDENTIFY AND QUANTIFY INTERACTIONS FOR UMAP ANALYSIS

    def interactions(self):

        #### IDENTIFY INTERACTIONS

        dfs = []

        proximity_threshold = 5  # 5mm
        min_interaction_frames = 5
        frame_buffer = 20  # Extend interaction by this many frames before and after

        def process_track_pair(track_a, track_b, df, track_file):
            results = []
            track_a_data = df[df['track_id'] == track_a]
            track_b_data = df[df['track_id'] == track_b]

            common_frames = sorted(set(track_a_data['frame']).intersection(track_b_data['frame']))
            interaction_id_local = 0
            i = 0

            while i < len(common_frames):
                current_interaction = []

                # Try to detect an interaction sequence
                while i < len(common_frames):
                    frame = common_frames[i]

                    point_a = track_a_data[track_a_data['frame'] == frame][['x_body', 'y_body']].to_numpy(dtype=float)
                    point_b = track_b_data[track_b_data['frame'] == frame][['x_body', 'y_body']].to_numpy(dtype=float)

                    dist = float(np.linalg.norm(point_a - point_b))
                    
                    if dist < proximity_threshold:
                        current_interaction.append(frame)
                        i += 1
                    elif current_interaction:
                        break
                    else:
                        i += 1

                # Only process if interaction is long enough
                if len(current_interaction) >= min_interaction_frames:
                    # Add buffer before and after
                    start_idx = max(0, common_frames.index(current_interaction[0]) - frame_buffer)
                    end_idx = min(len(common_frames), common_frames.index(current_interaction[-1]) + frame_buffer + 1)
                    interaction_frames = common_frames[start_idx:end_idx]

                    max_allowed_gap = 2  # or 1 if you want stricter
                    if np.any(np.diff(interaction_frames) > max_allowed_gap): ## check there arent jumps in frames if one larvae left or was disclided in digging idk 
                        i = end_idx
                        continue


                    interaction_id_local += 1

    
                    for frame in interaction_frames:
                        point_a = track_a_data[track_a_data['frame'] == frame][['x_body', 'y_body']].to_numpy(dtype=float)
                        point_b = track_b_data[track_b_data['frame'] == frame][['x_body', 'y_body']].to_numpy(dtype=float)

                        a_tail = track_a_data[track_a_data['frame'] == frame][['x_tail', 'y_tail']].to_numpy(dtype=float)
                        b_tail = track_b_data[track_b_data['frame'] == frame][['x_tail', 'y_tail']].to_numpy(dtype=float)

                        a_head = track_a_data[track_a_data['frame'] == frame][['x_head', 'y_head']].to_numpy(dtype=float)
                        b_head = track_b_data[track_b_data['frame'] == frame][['x_head', 'y_head']].to_numpy(dtype=float)

                        dist = float(np.linalg.norm(point_a - point_b))

                        results.append({
                            'Frame': frame,
                            'Local Interaction ID': interaction_id_local,
                            'file': track_file,
                            'Interaction Pair': (track_a, track_b),
                            'Distance': dist,
                            'Track_1 x_tail': a_tail[0, 0],
                            'Track_1 y_tail': a_tail[0, 1],
                            'Track_1 x_body': point_a[0, 0],
                            'Track_1 y_body': point_a[0, 1],
                            'Track_1 x_head': a_head[0, 0],
                            'Track_1 y_head': a_head[0, 1],
                            'Track_2 x_tail': b_tail[0, 0],
                            'Track_2 y_tail': b_tail[0, 1],
                            'Track_2 x_body': point_b[0, 0],
                            'Track_2 y_body': point_b[0, 1],
                            'Track_2 x_head': b_head[0, 0],
                            'Track_2 y_head': b_head[0, 1]
                        })

                    # Skip ahead to the frame after the current interaction to avoid overlap
                    i = end_idx
                else:
                    i += 1

            return results


        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file]
            df = df.sort_values(by='frame', ascending=True)
            df['filename'] = track_file

            track_ids = df['track_id'].unique()
            track_combinations = list(combinations(track_ids, 2))

            all_results = Parallel(n_jobs=-1)(
                delayed(process_track_pair)(track_a, track_b, df, track_file)
                for track_a, track_b in track_combinations
            )

            # Flatten results
            flattened_results = [item for sublist in all_results for item in sublist]

            results_df = pd.DataFrame(flattened_results)
            results_df.set_index('Frame', inplace=True, drop=False)
            dfs.append(results_df)

        # Combine all files
        interaction_data = pd.concat(dfs, ignore_index=True)

        # Assign global interaction IDs across files and pairs
        interaction_data['Interaction Number'] = (
            interaction_data
            .groupby(['file', 'Interaction Pair', 'Local Interaction ID'])
            .ngroup() + 1  # make it start at 1
        )

        interaction_data.drop(columns=['Local Interaction ID'], inplace=True)      # Drop the local ID if you don't need it

        interaction_data = interaction_data[['Frame', 'Interaction Number',*[col for col in interaction_data.columns if col not in ['Frame', 'Interaction Number']]]]
        
        
        #### DISTANCES BETWEEN ALL BODY PART COMBINATIONS 

        def euclidean_distance(df, x1, y1, x2, y2):
            return np.sqrt((df[x1] - df[x2])**2 + (df[y1] - df[y2])**2)

        interaction_data['t1_tail-tail_t2'] = euclidean_distance(interaction_data, 'Track_1 x_tail', 'Track_1 y_tail', 'Track_2 x_tail', 'Track_2 y_tail')
        interaction_data['t1_tail-body_t2'] = euclidean_distance(interaction_data, 'Track_1 x_tail', 'Track_1 y_tail', 'Track_2 x_body', 'Track_2 y_body')
        interaction_data['t1_tail-head_t2'] = euclidean_distance(interaction_data, 'Track_1 x_tail', 'Track_1 y_tail', 'Track_2 x_head', 'Track_2 y_head')

        interaction_data['t1_body-tail_t2'] = euclidean_distance(interaction_data,'Track_1 x_body', 'Track_1 y_body', 'Track_2 x_tail', 'Track_2 y_tail')
        interaction_data['t1_body-body_t2'] = euclidean_distance(interaction_data, 'Track_1 x_body', 'Track_1 y_body', 'Track_2 x_body', 'Track_2 y_body')
        interaction_data['t1_body-head_t2'] = euclidean_distance(interaction_data, 'Track_1 x_body', 'Track_1 y_body', 'Track_2 x_head', 'Track_2 y_head')

        interaction_data['t1_head-tail_t2'] = euclidean_distance(interaction_data, 'Track_1 x_head', 'Track_1 y_head', 'Track_2 x_tail', 'Track_2 y_tail')
        interaction_data['t1_head-body_t2'] = euclidean_distance(interaction_data, 'Track_1 x_head', 'Track_1 y_head', 'Track_2 x_body', 'Track_2 y_body')
        interaction_data['t1_head-head_t2'] = euclidean_distance(interaction_data, 'Track_1 x_head', 'Track_1 y_head', 'Track_2 x_head', 'Track_2 y_head')
        

        #### QUANTIFY SPEED
        def speed(group, x, y):
            dx = group[x].diff()
            dy = group[y].diff()
            
            distance = np.sqrt(dx**2 + dy**2)
            dt = group['Frame'].diff()

            speed = distance / dt.replace(0, np.nan)
            return speed

        interaction_data['track1_speed'] = interaction_data.groupby('Interaction Number').apply(lambda group: speed(group, 'Track_1 x_body', 'Track_1 y_body')).reset_index(level=0, drop=True)
        interaction_data['track2_speed'] = interaction_data.groupby('Interaction Number').apply(lambda group: speed(group, 'Track_2 x_body', 'Track_2 y_body')).reset_index(level=0, drop=True)

        #### QUANTIFY ACCELERATION
        interaction_data['track1_acceleration'] = interaction_data.groupby('Interaction Number')['track1_speed'].diff() / interaction_data.groupby('Interaction Number')['Frame'].diff()
        interaction_data['track2_acceleration'] = interaction_data.groupby('Interaction Number')['track2_speed'].diff() / interaction_data.groupby('Interaction Number')['Frame'].diff()

        #### TAIL-BODY-HEAD LENGTH

        interaction_data['track1_length'] = (
            np.sqrt((interaction_data['Track_1 x_body'] - interaction_data['Track_1 x_tail'])**2 + 
                    (interaction_data['Track_1 y_body'] - interaction_data['Track_1 y_tail'])**2) 
            +
            np.sqrt((interaction_data['Track_1 x_head'] - interaction_data['Track_1 x_body'])**2 + 
                    (interaction_data['Track_1 y_head'] - interaction_data['Track_1 y_body'])**2)
        )


        interaction_data['track2_length'] = (
            np.sqrt((interaction_data['Track_2 x_body'] - interaction_data['Track_2 x_tail'])**2 + 
                    (interaction_data['Track_2 y_body'] - interaction_data['Track_2 y_tail'])**2) 
            +
            np.sqrt((interaction_data['Track_2 x_head'] - interaction_data['Track_2 x_body'])**2 + 
                    (interaction_data['Track_2 y_head'] - interaction_data['Track_2 y_body'])**2)
        )

        #### ANGLE BETWEEN TAIL-BODY AND BODY-HEAD PARTS

        # Tail-Body Vector for Track 1
        interaction_data['track1 TB_x'] =  interaction_data['Track_1 x_tail'] - interaction_data['Track_1 x_body'] 
        interaction_data['track1 TB_y'] =  interaction_data['Track_1 y_tail'] - interaction_data['Track_1 y_body'] 
        # Body-Head Vector for Track 1
        interaction_data['track1 BH_x'] = interaction_data['Track_1 x_head'] - interaction_data['Track_1 x_body']
        interaction_data['track1 BH_y'] = interaction_data['Track_1 y_head'] - interaction_data['Track_1 y_body']
        # Tail-Body Vector for Track 2
        interaction_data['track2 TB_x'] = interaction_data['Track_2 x_tail'] - interaction_data['Track_2 x_body'] 
        interaction_data['track2 TB_y'] = interaction_data['Track_2 y_tail'] - interaction_data['Track_2 y_body'] 
        # Body-Head Vector for Track 2
        interaction_data['track2 BH_x'] = interaction_data['Track_2 x_head'] - interaction_data['Track_2 x_body']
        interaction_data['track2 BH_y'] = interaction_data['Track_2 y_head'] - interaction_data['Track_2 y_body']


        def calculate_angle(interaction_data, v1_x, v1_y, v2_x, v2_y):
            dot_product = (interaction_data[v1_x] * interaction_data[v2_x]) + (interaction_data[v1_y] * interaction_data[v2_y])

            magnitude_v1 = np.hypot(interaction_data[v1_x], interaction_data[v1_y])  # Same as sqrt(x^2 + y^2)
            magnitude_v2 = np.hypot(interaction_data[v2_x], interaction_data[v2_y])
            
            # Avoid division by zero
            cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure values are in valid range for arccos
            
            return np.degrees(np.arccos(cos_theta))  # Convert radians to degrees


        # Calculate angles for each track
        interaction_data['track1_angle'] = calculate_angle(interaction_data,'track1 TB_x', 'track1 TB_y', 'track1 BH_x', 'track1 BH_y')
        interaction_data['track2_angle'] = calculate_angle(interaction_data, 'track2 TB_x', 'track2 TB_y', 'track2 BH_x', 'track2 BH_y')


        #### == DEFINE CLOSEST BODY PARTS BETWEEN TRACKS == ####

        # Define distance columns
        distance_columns = [
            't1_tail-tail_t2', 't1_tail-body_t2', 't1_tail-head_t2',
            't1_body-tail_t2', 't1_body-body_t2', 't1_body-head_t2',
            't1_head-tail_t2', 't1_head-body_t2', 't1_head-head_t2'
        ]

        interaction_data["min_distance"] = interaction_data[distance_columns].min(axis=1) # identifies smallest numerical value
        interaction_data["interaction_type"] = interaction_data[distance_columns].idxmin(axis=1) # returns column name holding smallest value
        interaction_data["interaction_type"] = interaction_data["interaction_type"].str.extract(r"t1_(.*-.*)_t2")
        
        #### == ANGLE OF APPROACH BETWEEN INTERACTION PARTNERS == ####
            

        # Mapping from interaction_type (e.g., 'body-head') to coordinate columns
        # part_mapping = {
        #     'tail-tail': ('x_tail', 'y_tail'),
        #     'tail-body': ('x_body', 'y_body'),
        #     'tail-head': ('x_head', 'y_head'),
        #     'body-tail': ('x_tail', 'y_tail'),
        #     'body-body': ('x_body', 'y_body'),
        #     'body-head': ('x_head', 'y_head'),
        #     'head-tail': ('x_tail', 'y_tail'),
        #     'head-body': ('x_body', 'y_body'),
        #     'head-head': ('x_head', 'y_head'),
        # }

        # Function to compute angle between two vectors
        def angle_between_vectors(x1, y1, x2, y2):
            dot = x1 * x2 + y1 * y2
            mag1 = np.hypot(x1, y1)
            mag2 = np.hypot(x2, y2)
            cos_theta = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
            return np.degrees(np.arccos(cos_theta))

        # def get_track1_approach_angle(row):
        #     try:
        #         x_part, y_part = part_mapping.get(row['interaction_type'], (None, None))
        #         if x_part is None:
        #             return np.nan

        #         # Make both vectors start at the head
        #         hx = row['Track_1 x_body'] - row['Track_1 x_head']
        #         hy = row['Track_1 y_body'] - row['Track_1 y_head']

        #         ax = row[f'Track_2 {x_part}'] - row['Track_1 x_head']
        #         ay = row[f'Track_2 {y_part}'] - row['Track_1 y_head']

        #         return angle_between_vectors(hx, hy, ax, ay)
        #     except Exception as e:
        #         print(f"❌ Track 1 error at row {row.name}: {e}")
        #         return np.nan


        # def get_track2_approach_angle(row):
        #     try:
        #         x_part, y_part = part_mapping.get(row['interaction_type'], (None, None))
        #         if x_part is None:
        #             return np.nan

        #         # heading = head - body
        #         hx = row['Track_2 x_body'] - row['Track_2 x_head']
        #         hy = row['Track_2 y_body'] - row['Track_2 y_head']

        #         # approach = other part - head
        #         ax = row[f'Track_1 {x_part}'] - row['Track_2 x_head']
        #         ay = row[f'Track_1 {y_part}'] - row['Track_2 y_head']

        #         return angle_between_vectors(hx, hy, ax, ay)
        #     except Exception as e:
        #         print(f"❌ Track 2 error at row {row.name}: {e}")
        #         return np.nan

        def get_track1_approach_angle(row):
            try:
                part1, part2 = row["interaction_type"].split("-")

                # Track 1 heading: body → head
                # hx = row['Track_1 x_body'] - row['Track_1 x_head']
                # hy = row['Track_1 y_body'] - row['Track_1 y_head']

                hx = row['Track_1 x_head'] - row['Track_1 x_body']
                hy = row['Track_1 y_head'] - row['Track_1 y_body']

                # Approach vector: Track_2 part2 - Track_1 head
                ax = row[f'Track_2 x_{part2}'] - row['Track_1 x_head']
                ay = row[f'Track_2 y_{part2}'] - row['Track_1 y_head']

                return angle_between_vectors(hx, hy, ax, ay)

            except Exception as e:
                print(f"❌ Track 1 error at row {row.name}: {e}")
                return np.nan
            
        
        def get_track2_approach_angle(row):
            try:
                part1, part2 = row["interaction_type"].split("-")

                # Track 2 heading: body → head
                # hx = row['Track_2 x_body'] - row['Track_2 x_head']
                # hy = row['Track_2 y_body'] - row['Track_2 y_head']
                hx = row['Track_2 x_head'] - row['Track_2 x_body']
                hy = row['Track_2 y_head'] - row['Track_2 y_body']

                # Approach vector: Track_1 part1 - Track_2 head
                ax = row[f'Track_1 x_{part1}'] - row['Track_2 x_head']
                ay = row[f'Track_1 y_{part1}'] - row['Track_2 y_head']

                return angle_between_vectors(hx, hy, ax, ay)

            except Exception as e:
                print(f"❌ Track 2 error at row {row.name}: {e}")
                return np.nan


        # Apply to DataFrame
        interaction_data['track1_approach_angle'] = interaction_data.apply(get_track1_approach_angle, axis=1)
        interaction_data['track2_approach_angle'] = interaction_data.apply(get_track2_approach_angle, axis=1)



         #### == IDENTIFY CLOSEST POINT OF INTERACTION AND NORMALISE FRAMES == ####

        # min_distance_frames = interaction_data.groupby("Interaction Number")["min_distance"].idxmin()


        # min_distance_frames = interaction_data.groupby("Interaction Number")["min_distance"].idxmin()

        ### NORMALISE FRAMES TO FRAME OF CLOSEST NODE-NODE CONTACT E.G. HEAD-TAIL 

        def normalize_frames(group):
            # min_frame = group.loc[min_distance_frames[group.name], "Frame"]  # Get the min distance frame
            # group["Normalized Frame"] = group["Frame"] - min_frame  # Normalize all frames in the group
            # return group
        
            min_distance_position = group["min_distance"].values.argmin()
            min_frame = group.iloc[min_distance_position]["Frame"]
            
            group["Normalized Frame"] = group["Frame"] - min_frame
            return group

        interaction_data = interaction_data.groupby("Interaction Number", group_keys=False).apply(normalize_frames)
 


        #### == NORMALISE TRACK COORDINATES TO MIDPOINT OF BODY COORDINATES AT THE CLOSEST DISTANCE == ####

        # Normalize coordinates based on closest pair at frame 0

        distance_columns = [
            't1_tail-tail_t2', 't1_tail-body_t2', 't1_tail-head_t2',
            't1_body-tail_t2', 't1_body-body_t2', 't1_body-head_t2',
            't1_head-tail_t2', 't1_head-body_t2', 't1_head-head_t2'
        ]

        coordinate_columns = [
            "Track_1 x_body", "Track_1 y_body", "Track_2 x_body", "Track_2 y_body",
            "Track_1 x_tail", "Track_1 y_tail", "Track_2 x_tail", "Track_2 y_tail",
            "Track_1 x_head", "Track_1 y_head", "Track_2 x_head", "Track_2 y_head"
        ]

        for interaction in interaction_data["Interaction Number"].unique():
            interaction_subset = interaction_data[interaction_data["Interaction Number"] == interaction]
            min_frame_row = interaction_subset[interaction_subset["Normalized Frame"] == 0]

            if min_frame_row.empty:
                continue

            row = min_frame_row.iloc[0]  # take first in case there are multiple

            closest_part = row[distance_columns].astype(float).idxmin()


            match = re.match(r't1_(\w+)-(\w+)_t2', closest_part)
            if not match:
                continue  # skip if pattern doesn't match (safety check)

            part1, part2 = match.groups()  # e.g., 'tail', 'head'

            part1_x = f"Track_1 x_{part1}"
            part1_y = f"Track_1 y_{part1}"
            part2_x = f"Track_2 x_{part2}"
            part2_y = f"Track_2 y_{part2}"

            # mid_x = (row[part1_x] + row[part2_x]) / 2
            # mid_y = (row[part1_y] + row[part2_y]) / 2

            # # Subtract midpoint from all coordinate columns for this interaction
            # for col in coordinate_columns:
            #     if "x_" in col:
            #         interaction_data.loc[interaction_data["Interaction Number"] == interaction, col] -= mid_x
            #     elif "y_" in col:
            #         interaction_data.loc[interaction_data["Interaction Number"] == interaction, col] -= mid_y

            mid_x = (row[part1_x] + row[part2_x]) / 2
            mid_y = (row[part1_y] + row[part2_y]) / 2

            # Create masks for update
            mask = interaction_data["Interaction Number"] == interaction

            # Save midpoints for future un-normalization
            interaction_data.loc[mask, "Normalization mid_x"] = mid_x
            interaction_data.loc[mask, "Normalization mid_y"] = mid_y

            # Subtract midpoint from all coordinate columns
            for col in coordinate_columns:
                if "x_" in col:
                    interaction_data.loc[mask, col] -= mid_x
                elif "y_" in col:
                    interaction_data.loc[mask, col] -= mid_y

        desired_order = ['file', "Frame", "Interaction Number", "Normalized Frame"]
        interaction_data = interaction_data[desired_order + [col for col in interaction_data.columns if col not in desired_order]]
        interaction_data = interaction_data.sort_values(by=["Interaction Number", "Normalized Frame"])


        print("Number of interaction rows:", interaction_data.shape[0])
        print("Interaction DataFrame head:\n", interaction_data.head())

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"interactions{suffix}.csv"


        interaction_data.to_csv(os.path.join(self.directory, filename), index=False)


##############################################  ---- GH+SI ----  ##############################################


    def GH_SI_interaction_types_closest(self, threshold=1):

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

        # NEW: track_id -> SI/GH label
        def track_social_group(track_id: int) -> str:
            # 0-4 socially isolated, 5-9 group-housed
            return "SI" if int(track_id) <= 4 else "GH"

        # NEW: order-invariant pair label (always SI-GH not GH-SI)
        def social_experience_label(track_a: int, track_b: int) -> str:
            a = track_social_group(track_a)
            b = track_social_group(track_b)
            return "-".join(sorted([a, b]))  # ensures GH-SI becomes SI-GH

        def process_track_pair(track_a, track_b, df, track_file):
            results = []
            track_a_data = df[df['track_id'] == track_a]
            track_b_data = df[df['track_id'] == track_b]

            common_frames = sorted(set(track_a_data['frame']).intersection(track_b_data['frame']))
            if not common_frames:
                return results

             # NEW: compute once per pair (same for all frames)
            pair_social_experience = social_experience_label(track_a, track_b)

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
                        'social_experience': pair_social_experience,  # NEW COLUMN
                        'track_0': track_a,
                        'track_1': track_b,
                        'track_0_node': min_part_a,
                        'track_1_node': min_part_b,
                        'Distance': min_dist,
                        'Closest Interaction Type': unify_interaction_type(min_part_a, min_part_b)
                    })

            return results

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file].sort_values(by='frame')

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
                'social_experience': None,  # NEW placeholder field
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








############################################  ---- HEAD-HEAD ----  ############################################   

    def head_head_interaction_type(self, proximity_threshold=1):
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

                if min_dist < proximity_threshold:
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

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file].sort_values(by='frame')

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

        filename = f"closest_contacts_{proximity_threshold}mm{suffix}.csv"
        closest_df.to_csv(os.path.join(self.directory, filename), index=False)

        return closest_df
    


    ## FOR RASTA PLOTTING
    def head_head_interaction_type_over_time(self, proximity_threshold=1):

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
                min_type = None
                all_dists = []  # (dist, part1, part2, unified_type)
                for part1, part2 in interaction_pairs:
                    dist = np.linalg.norm(coords_a[part1] - coords_b[part2])
                    unified = unify_interaction_type(part1, part2)
                    all_dists.append((dist, part1, part2, unified))

                    if dist < min_dist:
                        min_dist = dist
                        min_type = unified
                
                touching = min_dist < proximity_threshold


                # ---- choose "relevant" interaction type with HH / HT priority ----
                relevant_type = None
                if touching:
                    best_special = None  # (dist, unified_type)

                    for dist, p1, p2, unified in all_dists:
                        if unified not in ('head_head', 'head_tail'):
                            continue
                        if dist < proximity_threshold:
                            if best_special is None or dist < best_special[0]:
                                best_special = (dist, unified)

                    if best_special is not None:
                        # prefer head_head / head_tail if present under threshold
                        relevant_type = best_special[1]
                    else:
                        # fallback: just use the global minimum type
                        relevant_type = min_type


                results.append({
                'file': track_file,
                'frame': frame,
                'Interaction Pair': tuple(sorted((track_a, track_b))),
                'touching': touching,
                'Distance': min_dist if touching else np.nan,
                'Closest Interaction Type': min_type if touching else None,
                'Relevant Interaction Type': relevant_type if touching else None,})

            return results


        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file].sort_values(by='frame')

            track_ids = df['track_id'].unique()
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
                'touching': False,
                'Distance': np.nan,
                'Closest Interaction Type': None,
                'Relevant Interaction Type': None,
            }]))


        closest_df = pd.concat(data, ignore_index=True)

        if self.shorten and self.shorten_duration is not None:
            suffix = f"_{self.shorten_duration}"
        else:
            suffix = ""

        filename = f"closest_contacts_{proximity_threshold}mm{suffix}_overtime.csv"
        closest_df.to_csv(os.path.join(self.directory, filename), index=False)

        return closest_df
    




    def head_head_approach_angle(self):

        def angle_calculator(vector_A, vector_B):
            # Same helper as in movement_direction
            A = np.array(vector_A, dtype=np.float64)
            B = np.array(vector_B, dtype=np.float64)

            if not np.isnan(A).any() and not np.isnan(B).any():
                mag_A = np.linalg.norm(A)
                mag_B = np.linalg.norm(B)

                if mag_A != 0 and mag_B != 0:
                    dot_product = np.dot(A, B)
                    cos_theta = dot_product / (mag_A * mag_B)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    theta_radians = np.arccos(cos_theta)
                    theta_degrees = np.degrees(theta_radians)
                    return theta_degrees

            return np.nan
        

        body_parts = ['head', 'body', 'tail']
        dfs = []
        
        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file]
            df['file'] = track_file     # ← create column first
            df = df.sort_values(by=['file', 'frame'])

            # BODY-HEAD VECTOR
            df['body_head_dx'] = df['x_head'] - df['x_body']
            df['body_head_dy'] = df['y_head'] - df['y_body']
            
            # INITIALISE COLUMNS
            df['body_body_distance'] = np.nan
            df['other_id'] = np.nan
            df['closest_other_node'] = np.nan
            df['head_other_dx'] = np.nan
            df['head_other_dy'] = np.nan
            df['approach_angle'] = np.nan
            df['closest_other_node_distance'] = np.nan


            for frame, frame_group in df.groupby('frame'):
                if frame_group['track_id'].nunique() != 2:
                    continue

                rows = frame_group.sort_values('track_id')
                row1 = rows.iloc[0]
                row2 = rows.iloc[1]

                idx1 = row1.name
                idx2 = row2.name

                body1 = np.array([row1['x_body'], row1['y_body']], float)
                body2 = np.array([row2['x_body'], row2['y_body']], float)
                body_body_dist = np.linalg.norm(body2 - body1)

                df.at[idx1, 'body_body_distance'] = body_body_dist
                df.at[idx2, 'body_body_distance'] = body_body_dist

                # build node arrays for each larva: [head, body, tail]
                nodes1 = np.array([
                    [row1['x_head'], row1['y_head']],
                    [row1['x_body'], row1['y_body']],
                    [row1['x_tail'], row1['y_tail']],
                ], dtype=float)

                nodes2 = np.array([
                    [row2['x_head'], row2['y_head']],
                    [row2['x_body'], row2['y_body']],
                    [row2['x_tail'], row2['y_tail']],
                ], dtype=float)

                 # head positions
                head1 = nodes1[0]  # larva 1 head
                head2 = nodes2[0]  # larva 2 head

                # --- focal = larva 1, other = larva 2 ---
                diffs_1 = nodes2 - head1           # head1 -> each node of larva 2
                dists_1 = np.linalg.norm(diffs_1, axis=1)

                if not np.isnan(dists_1).all():
                    idx_min_1 = np.nanargmin(dists_1)  # 0=head, 1=body, 2=tail
                    df.at[idx1, 'other_id'] = row2['track_id']
                    df.at[idx1, 'closest_other_node'] = body_parts[idx_min_1]

                    closest_vec_1 = diffs_1[idx_min_1]          # head1 -> closest node on larva 2
                    v_body_head_1 = np.array(
                        [row1['body_head_dx'], row1['body_head_dy']], dtype=float
                    )
                    dist1 = np.linalg.norm(closest_vec_1)
                    df.at[idx1, 'closest_other_node_distance'] = dist1

                    if not (np.isnan(v_body_head_1).any() or np.linalg.norm(v_body_head_1) == 0):
                        angle_1 = angle_calculator(v_body_head_1, closest_vec_1)
                        df.at[idx1, 'head_other_dx'] = closest_vec_1[0]
                        df.at[idx1, 'head_other_dy'] = closest_vec_1[1]
                        df.at[idx1, 'approach_angle'] = angle_1


                # --- focal = larva 2, other = larva 1 ---
                diffs_2 = nodes1 - head2           # head2 -> each node of larva 1
                dists_2 = np.linalg.norm(diffs_2, axis=1)

                if not np.isnan(dists_2).all():
                    idx_min_2 = np.nanargmin(dists_2)
                    df.at[idx2, 'other_id'] = row1['track_id']
                    df.at[idx2, 'closest_other_node'] = body_parts[idx_min_2]

                    closest_vec_2 = diffs_2[idx_min_2]          # head2 -> closest node on larva 1
                    v_body_head_2 = np.array(
                        [row2['body_head_dx'], row2['body_head_dy']], dtype=float
                    )
                    dist2 = np.linalg.norm(closest_vec_2)
                    df.at[idx2, 'closest_other_node_distance'] = dist2

                    if not (np.isnan(v_body_head_2).any() or np.linalg.norm(v_body_head_2) == 0):
                        angle_2 = angle_calculator(v_body_head_2, closest_vec_2)
                        df.at[idx2, 'head_other_dx'] = closest_vec_2[0]
                        df.at[idx2, 'head_other_dy'] = closest_vec_2[1]
                        df.at[idx2, 'approach_angle'] = angle_2
            
            dfs.append(df)
        
        result_df = pd.concat(dfs, ignore_index=True)
        result_df.to_csv(os.path.join(self.directory, "head_head_approach_angles.csv"), index=False)

    


    def head_head_first_contact_kinematics(self, proximity_threshold=1, window=30):

        data = []

        parts = ['head', 'body', 'tail']
        interaction_pairs = list(itertools.product(parts, parts))

        def heading_angle(body, head, tail):
            """
            Heading defined by tail->body and body->head vectors.
            Returns angle in degrees, 180 = forward.
            """
            v1 = np.array(body) - np.array(tail)   # tail -> body
            v2 = np.array(head) - np.array(body)   # body -> head

            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                return np.nan

            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_theta = np.clip(cos_theta, -1, 1)
            return np.degrees(np.arccos(cos_theta))
        
        def compute_speed(row_prev, row_curr):
            dx = row_curr['x_body'] - row_prev['x_body']
            dy = row_curr['y_body'] - row_prev['y_body']
            return np.sqrt(dx*dx + dy*dy)


        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file].sort_values('frame')

            track_ids = sorted(df['track_id'].unique())
            if len(track_ids) != 2:
                continue

            track_a, track_b = track_ids

            df_a = df[df['track_id'] == track_a]
            df_b = df[df['track_id'] == track_b]

            common_frames = sorted(set(df_a['frame']).intersection(df_b['frame']))
            if not common_frames:
                continue

            hh_frame = None

            # ---- find FIRST head–head contact ----
            for frame in common_frames:
                row_a = df_a[df_a['frame'] == frame]
                row_b = df_b[df_b['frame'] == frame]

                if row_a.empty or row_b.empty:
                    continue

                coords_a = {p: row_a[[f'x_{p}', f'y_{p}']].to_numpy().flatten() for p in parts}
                coords_b = {p: row_b[[f'x_{p}', f'y_{p}']].to_numpy().flatten() for p in parts}

                min_dist = np.inf
                min_pair = None

                for p1, p2 in interaction_pairs:
                    dist = np.linalg.norm(coords_a[p1] - coords_b[p2])
                    if dist < min_dist:
                        min_dist = dist
                        min_pair = (p1, p2)

                if min_dist < proximity_threshold and set(min_pair) == {'head'}:
                    hh_frame = frame
                    break

            if hh_frame is None:
                continue  # no head–head contact in this file

            start = hh_frame
            end = hh_frame + window
            interaction_min_dist = np.inf


            # ---- extract kinematics after first contact ----
            for frame in range(start, end + 1):
                frame_rows = df[df['frame'] == frame]
                if frame_rows['track_id'].nunique() != 2:
                    continue

                # ---- compute min node-node distance for THIS frame ----
                rows = frame_rows.sort_values('track_id')
                row_a = rows.iloc[0]
                row_b = rows.iloc[1]

                coords_a = {p: np.array([row_a[f'x_{p}'], row_a[f'y_{p}']]) for p in parts}
                coords_b = {p: np.array([row_b[f'x_{p}'], row_b[f'y_{p}']]) for p in parts}

                frame_min_dist = np.inf
                for p1, p2 in interaction_pairs:
                    dist = np.linalg.norm(coords_a[p1] - coords_b[p2])
                    if dist < frame_min_dist:
                        frame_min_dist = dist

               


                for _, row in frame_rows.iterrows():
                    body = (row['x_body'], row['y_body'])
                    head = (row['x_head'], row['y_head'])
                    tail = (row['x_tail'], row['y_tail'])

                    angle = heading_angle(body, head, tail)

                    prev = df[(df['track_id'] == row['track_id']) & (df['frame'] == frame - 1)]
                    if not prev.empty:
                        speed = compute_speed(prev.iloc[0], row)
                    else:
                        speed = np.nan


                    data.append({
                        'file': track_file,
                        'frame': frame,
                        'rel_frame': frame - hh_frame,
                        'track_id': row['track_id'],
                        'speed': speed,
                        'heading_angle': angle,
                        'min_distance': frame_min_dist 
                    })

        result_df = pd.DataFrame(data)
        result_df.to_csv(
            os.path.join(self.directory, 'head_head_first_contact_kinematics.csv'),
            index=False
        )

        return result_df



    def head_head_contacts_kinematics_over_time(self, proximity_threshold=1, window=30):

        data = []

        parts = ['head', 'body', 'tail']
        interaction_pairs = list(itertools.product(parts, parts))

        def heading_angle(body, head, tail):
            v1 = np.array(body) - np.array(tail)
            v2 = np.array(head) - np.array(body)

            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                return np.nan

            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_theta = np.clip(cos_theta, -1, 1)
            return np.degrees(np.arccos(cos_theta))

        def compute_speed(row_prev, row_curr):
            dx = row_curr['x_body'] - row_prev['x_body']
            dy = row_curr['y_body'] - row_prev['y_body']
            return np.sqrt(dx*dx + dy*dy)

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file].sort_values('frame')

            track_ids = sorted(df['track_id'].unique())
            if len(track_ids) != 2:
                continue

            df_a = df[df['track_id'] == track_ids[0]]
            df_b = df[df['track_id'] == track_ids[1]]

            common_frames = sorted(set(df_a['frame']).intersection(df_b['frame']))
            if not common_frames:
                continue

            interaction_number = 0
            next_allowed_frame = -np.inf

            # ---- scan frames sequentially ----
            for frame in common_frames:

                if frame < next_allowed_frame:
                    continue

                row_a = df_a[df_a['frame'] == frame]
                row_b = df_b[df_b['frame'] == frame]
                if row_a.empty or row_b.empty:
                    continue

                coords_a = {p: row_a[[f'x_{p}', f'y_{p}']].to_numpy().flatten() for p in parts}
                coords_b = {p: row_b[[f'x_{p}', f'y_{p}']].to_numpy().flatten() for p in parts}

                min_dist = np.inf
                min_pair = None
                for p1, p2 in interaction_pairs:
                    dist = np.linalg.norm(coords_a[p1] - coords_b[p2])
                    if dist < min_dist:
                        min_dist = dist
                        min_pair = (p1, p2)

                # ---- detect new head–head interaction ----
                if min_dist < proximity_threshold and set(min_pair) == {'head'}:
                    interaction_number += 1
                    interaction_frame = frame
                    next_allowed_frame = frame + window + 1


                    # pull initial body xy DIRECTLY
                    row_a0 = df_a[df_a["frame"] == interaction_frame].iloc[0]
                    row_b0 = df_b[df_b["frame"] == interaction_frame].iloc[0]

                    x0_a, y0_a = row_a0["x_body"], row_a0["y_body"]
                    x0_b, y0_b = row_b0["x_body"], row_b0["y_body"]

                    start_pos = {
                        track_ids[0]: (x0_a, y0_a),
                        track_ids[1]: (x0_b, y0_b),
                    }
     
                    # ---- extract kinematics window ----
                    for f in range(interaction_frame, interaction_frame + window + 1):
                        frame_rows = df[df['frame'] == f]
                        if frame_rows['track_id'].nunique() != 2:
                            continue

                        rows = frame_rows.sort_values('track_id')
                        row_a = rows.iloc[0]
                        row_b = rows.iloc[1]

                        coords_a = {p: np.array([row_a[f'x_{p}'], row_a[f'y_{p}']]) for p in parts}
                        coords_b = {p: np.array([row_b[f'x_{p}'], row_b[f'y_{p}']]) for p in parts}


                        frame_min_dist = np.inf
                        for p1, p2 in interaction_pairs:
                            dist = np.linalg.norm(coords_a[p1] - coords_b[p2])
                            if dist < frame_min_dist:
                                frame_min_dist = dist

                        for _, row in frame_rows.iterrows():
                            body = (row['x_body'], row['y_body'])
                            head = (row['x_head'], row['y_head'])
                            tail = (row['x_tail'], row['y_tail'])

                            angle = heading_angle(body, head, tail)

                            prev = df[
                                (df['track_id'] == row['track_id']) &
                                (df['frame'] == f - 1)
                            ]

                            speed = (
                                compute_speed(prev.iloc[0], row)
                                if not prev.empty else np.nan
                            )

                            x0, y0 = start_pos[row['track_id']]
                            dist_from_start = np.sqrt(
                                (row['x_body'] - x0)**2 +
                                (row['y_body'] - y0)**2
                            )

                            data.append({
                                'file': track_file,
                                'interaction_number': interaction_number,
                                'frame': f,
                                'rel_frame': f - interaction_frame,
                                'track_id': row['track_id'],
                                'speed': speed,
                                'heading_angle': angle,
                                'min_distance': frame_min_dist,
                                'dist_from_start': dist_from_start,
                                # coordinates for trajectory plotting
                                'x_head': row['x_head'],
                                'y_head': row['y_head'],
                                'x_body': row['x_body'],
                                'y_body': row['y_body'],
                                'x_tail': row['x_tail'],
                                'y_tail': row['y_tail'],
                            })

        result_df = pd.DataFrame(data)
        result_df.to_csv(
            os.path.join(self.directory, 'head_head_contacts_kinematics_over_time.csv'),
            index=False
        )

        return result_df












    

    
    








############################################  ---- HOLES ----  ############################################        

    # METHOD COMPUTE_HOLE: DETECTS WHETHER LARVAE IS WITHIN HOLE 

    def compute_hole(self):

        for match in self.matching_pairs:
            track_file = match['track_file']
            hole_boundary = match['hole_boundary']
            
            if hole_boundary is None:
                print(f"No hole boundary for track file: {track_file}")
                continue

            df = self.track_data[track_file]
            df = df.sort_values(['track_id', 'frame']).reset_index(drop=True)
            buffered_boundary = hole_boundary.buffer(1.5)  
            
            # 1. in hole?
            df['in_hole'] = df.apply(
                lambda row: buffered_boundary.contains(Point(row['x_body'], row['y_body'])) or 
                            buffered_boundary.touches(Point(row['x_body'], row['y_body'])),
                axis=1)
            
            # 2. digging near hole
            df['within_10mm'] = df.apply(lambda row: buffered_boundary.exterior.distance(Point(row['x_body'], row['y_body'])) <= 10, axis=1)

            # df['displacement'] = df.groupby('track_id').apply(lambda group: np.sqrt((group['x_body'].diff() ** 2) + (group['y_body'].diff() ** 2))).reset_index(drop=True)

            df['displacement'] = (
                    np.hypot(
                        df.groupby('track_id')['x_body'].diff(),
                        df.groupby('track_id')['y_body'].diff()
                    ).fillna(np.nan))

            df['cumulative_displacement'] = df.groupby('track_id')['displacement'].cumsum()

            df['cumulative_displacement_rate'] = df.groupby('track_id',  group_keys=False)['cumulative_displacement'].apply(lambda x: x.diff(10) / 10).fillna(0)

            df['frame_diff'] = df.groupby('track_id')['frame'].diff() #SPEED FOR LATER CALCULATIONS (same as displacement but just to be sure)
            df['speed'] = df['displacement'] / df['frame_diff']
            df['speed'] = df['speed'].fillna(0)

            df['x_std'] = df.groupby('track_id')['x_body'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
            df['y_std'] = df.groupby('track_id')['y_body'].transform(lambda x: x.rolling(window=10, min_periods=1).std())
            df['overall_std'] = np.sqrt(df['x_std']**2 + df['y_std']**2)

            # df['digging_near_hole'] = df['within_10mm'] & ((df['cumulative_displacement_rate'] < 0.4) | (df['overall_std'] < 0.8))

            df['digging_score'] = df['cumulative_displacement_rate'] * df['overall_std']
            df['digging_near_hole'] = (df['within_10mm']) & (df['digging_score'] < 0.25)


            # 3. combine
            df['hole'] = df['in_hole'] | df['digging_near_hole']

            # 3. threshold for hole rolling window
            df['within_hole'] = (
                df.groupby('track_id')['hole']
                .transform(lambda x: x.rolling(window=30, min_periods=1).sum() >= 30)
                .astype(bool))
            
            # 4. backfils true if larvae IS within hole 
            def expand_on_transitions(series, buffer=30): # transition from False to True
                result = series.copy()
                mask = series.astype(bool).values
                transitions = (mask[1:] & ~mask[:-1])  # detect False → True

                for i in np.where(transitions)[0] + 1:  # shift by 1 because diff is one step ahead
                    start = max(0, i - buffer + 1)
                    result.iloc[start:i + 1] = True

                return result

            df['within_hole'] = df.groupby('track_id')['within_hole'].transform(expand_on_transitions)

            def fill_short_false_gaps(series, max_gap):
                arr = series.values.astype(bool)
                inverse = ~arr  # find False runs
                # Label contiguous False regions
                labeled, num_features = label(inverse)

                # Flip short False runs to True
                for i in range(1, num_features + 1):
                    indices = np.where(labeled == i)[0]
                    if len(indices) <= max_gap:
                        arr[indices] = True

                return pd.Series(arr, index=series.index)

            # Apply it per track
            df['within_hole'] = df.groupby('track_id')['within_hole'].transform(
                lambda s: fill_short_false_gaps(s, max_gap=10)) # below 10 frames for returns filled out
            
            # Apply same logic to larvae outside 10mm zone
            df['digging_outside_hole'] = (
                (df['within_10mm'] == False) &
                (df['digging_score'] < 0.2) &
                (df['frame'] > 80)) # at the start they are slow
            

            def enforce_minimum_duration(series, min_duration=20):
                arr = series.values.astype(bool)
                output = np.zeros_like(arr, dtype=bool)

                start = 0
                while start < len(arr):
                    if arr[start]:
                        end = start
                        while end < len(arr) and arr[end]:
                            end += 1
                        if (end - start) >= min_duration:
                            output[start:end] = True
                        start = end
                    else:
                        start += 1
                return pd.Series(output, index=series.index)
            

            # Apply filtering: only keep runs of >= 20 frames
            df['digging_outside_hole'] = df.groupby('track_id')['digging_outside_hole'].transform(
                lambda s: enforce_minimum_duration(s, min_duration=20))


            self.track_data[track_file] = df
            # df.to_csv(os.path.join(self.directory, 'test.csv'), index=False)
    


    # METHOD HOLE_COUNTER: COUNTS NUMBER OF LARVAE IN HOLE 

    def hole_counter(self):

        data = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            
            df = self.track_data[track_file]


            for frame in df['frame'].unique():
                frame_df = df[df['frame'] == frame]

                inside_count = frame_df['within_hole'].sum()
                outside_count = 10 - inside_count

                data.append({'file': track_file, 'time': frame, 'inside_count': inside_count, 'outside':outside_count })
        
        hole_count = pd.DataFrame(data)
        hole_count = hole_count.sort_values(by=['time'], ascending=True)
        hole_count.to_csv(os.path.join(self.directory, "hole_count.csv"), index=False)

        return hole_count
    

    # METHOD 

    def hole_frame_counts(self):

        results = []

        for track_file in self.track_files:
            df = self.track_data[track_file]
            file_name = os.path.basename(track_file)

            frame_counts = (
                df.groupby(['track_id', 'within_hole'])
                .size()
                .unstack(fill_value=0)
                .rename(columns={True: 'frames_in_hole', False: 'frames_outside_hole'})
                .reset_index()
            )

            frame_counts['file'] = file_name
            results.append(frame_counts)

        summary = pd.concat(results, ignore_index=True)
        summary = summary[['file', 'track_id', 'frames_in_hole', 'frames_outside_hole']]

        summary['percent_in_hole'] = (summary['frames_in_hole'] / (summary['frames_in_hole'] + summary['frames_outside_hole']))
        summary['percent_in_hole'] *= 100
        
        out_file = os.path.join(self.directory, "hole_frame_counts.csv")
        summary.to_csv(out_file, index=False)
        
        return summary



    
    # METHOD TIME_TO_ENTER: CALCULATES TIME TO ENTER HOLE 

    def time_to_enter(self):

        data = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            hole_boundary = match['hole_boundary']
            
            if hole_boundary is None:
                print(f"No hole boundary for track file: {track_file}")
                continue

            df = self.track_data[track_file]
      
            for track in df['track_id'].unique():
                unique_track = df[df['track_id'] == track]
                unique_track = unique_track.sort_values(by=['frame'], ascending=True)

                # entered = False

                # for row in unique_track.itertuples():
    
                #     if row.within_hole:
                #         data.append({'file': track_file, 'track': track, 'time': row.frame})
                #         entered = True
                #         break
                # if not entered:
                #     data.append({'file': track_file, 'track': track, 'time': 3600})

                entered = False
                passes = 0
                in_vicinity = False  # track whether we're currently inside the 10mm zone

                for row in unique_track.itertuples():
                    if row.within_hole:
                        data.append({
                            'file': track_file,
                            'track': track,
                            'time': row.frame,
                            'passes_before_entry': passes
                        })
                        entered = True
                        break

                    elif row.within_10mm and not row.within_hole:
                        if not in_vicinity:
                            passes += 1
                            in_vicinity = True  # mark entry into the vicinity
                    else:
                        in_vicinity = False  # exited the vicinity
                    
                if not entered:
                        data.append({
                            'file': track_file,
                            'track': track,
                            'time': 3600,
                            'passes_before_entry': passes
                        })

        hole_entry_time = pd.DataFrame(data)
        hole_entry_time = hole_entry_time.sort_values(by=['file'], ascending=True)
        hole_entry_time.to_csv(os.path.join(self.directory, 'hole_entry_time.csv'), index=False)

        return hole_entry_time


    # DEF RETURNS: CALCULATES THE NUMBER OF LARVAE WHICH RETURN TO THE HOLE AND THE TIME TAKEN 

    def returns(self):

        data = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file]

            for track in df['track_id'].unique():
                track_df = df[df['track_id'] == track].sort_values(by='frame').reset_index(drop=True)
                states = track_df['within_hole'].astype(bool).values
                frames = track_df['frame'].values

                i = 1
                while i < len(states):
                    # Detect True → False transition (exit)
                    if states[i - 1] and not states[i]:
                        exit_frame = frames[i]

                        # Now search for the next False → True (re-entry)
                        j = i + 1
                        while j < len(states):
                            if not states[j - 1] and states[j]:
                                return_frame = frames[j]
                                return_time = return_frame - exit_frame
    
                                return_distance = track_df.loc[
                                        (track_df['frame'] > exit_frame) & (track_df['frame'] <= return_frame),
                                        'displacement'].sum() #displacement was calculated in the compute_hole 

                                data.append({
                                    'file': track_file,
                                    'track': track,
                                    'exit frame': exit_frame,
                                    'return frame': return_frame,
                                    'return_time': return_time,
                                    'distance_covered': return_distance

                                })

                                i = j  # move outer loop forward after return
                                break
                            j += 1
                    i += 1

        df_returns = pd.DataFrame(data)
        df_returns = df_returns.sort_values(by=['file', 'track', 'exit frame'])
        df_returns.to_csv(os.path.join(self.directory, 'returns.csv'), index=False)
        return df_returns
    

    # METHOD HOLE_DEPARTURES: CALCULATES THE NUMBER OF LARVAE LEAVING THE HOLE

    def hole_departures(self):

        data = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file]

            for track in df['track_id'].unique():
                track_df = df[df['track_id'] == track].sort_values(by='frame').reset_index(drop=True)
                states = track_df['within_hole'].astype(bool).values
                frames = track_df['frame'].values

                count = 0

                i = 1
                while i < len(states):
                    # Detect True → False transition (exit)
                    if states[i - 1] and not states[i]:
                        count += 1
                    i += 1

                data.append({'file': track_file, 'track': track, 'departures': count})

        hole_departures = pd.DataFrame(data)
        hole_departures = hole_departures.sort_values(by=['track'], ascending=True)
        hole_departures.to_csv(os.path.join(self.directory, 'hole_departures.csv'), index=False)

        return hole_departures
    
    # METHOD HOLE_ENTRY-DEPARTURE_LATENCY:

    def hole_entry_departure_latency(self):

        data = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file].sort_values(by='frame')

            # Get list of unique frames
            all_frames = df['frame'].unique()
            all_frames.sort()

            # Track previous inside state
            prev_inside_ids = set()

            for i, frame in enumerate(all_frames):
                current_df = df[df['frame'] == frame]
                current_inside_ids = set(current_df[current_df['within_hole']]['track_id'])

                # Detect new entries: tracks that were not inside before but are now
                new_entries = current_inside_ids - prev_inside_ids

                for entrant in new_entries:
                    # Count how many are in hole including this new one
                    count_at_entry = len(current_inside_ids)

                    # Now search forward for first time a larva leaves
                    latency = None
                    departure_happened = False

                    for j in range(i + 1, len(all_frames)):
                        next_frame = all_frames[j]
                        next_df = df[df['frame'] == next_frame]
                        next_inside_ids = set(next_df[next_df['within_hole']]['track_id'])

                        if len(next_inside_ids) < len(current_inside_ids):
                            latency = next_frame - frame
                            departure_happened = True
                            break

                    data.append({
                        'file': track_file,
                        'entry_frame': frame,
                        'number_in_hole_at_entry': count_at_entry,
                        'latency_to_next_departure': latency,
                        'departure_happened': departure_happened
                    })

                # Update previous state
                prev_inside_ids = current_inside_ids

        # Convert to DataFrame
        result = pd.DataFrame(data)
        result = result.sort_values(by=['file', 'entry_frame'])
        result.to_csv(os.path.join(self.directory, 'hole_entry_departure_latency.csv'), index=False)



    # METHOD SPEED_HOLE():

    def speed_hole(self):

        summary = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file]

            for track_id, group in df.groupby('track_id'):
                group = group.sort_values('frame')
                # Get first entry frame
                hole_frames = group[group['within_hole'] == True]['frame']
                if hole_frames.empty:
                    continue  # skip if never entered

                first_entry = hole_frames.iloc[0]

                # # Speeds before entering the hole
                # before_mask = (group['frame'] < first_entry)
                # speeds_before = group.loc[before_mask, 'speed']

                # # Speeds after entering (but not in hole)
                # after_mask = (group['frame'] > first_entry) & (~group['within_hole'])
                # speeds_after = group.loc[after_mask, 'speed']

                before_mask = (
                    (group['frame'] < first_entry) &
                    (~group['digging_outside_hole'])
                )
                speeds_before = group.loc[before_mask, 'speed']

                # Speeds after entering (exclude digging outside hole)
                after_mask = (
                    (group['frame'] > first_entry) &
                    (~group['within_hole']) &
                    (~group['digging_outside_hole'])
                )
                speeds_after = group.loc[after_mask, 'speed']

                # Compare individual larva behavior before/after

                summary.append({
                    'file': track_file,
                    'track_id': track_id,
                    'first_entry_frame': first_entry,
                    'mean_speed_before': speeds_before.mean(),
                    'mean_speed_after': speeds_after.mean(),
                    'n_frames_before': len(speeds_before),
                    'n_frames_after': len(speeds_after)
                })
        
        speed_comparison = pd.DataFrame(summary)
        speed_comparison.to_csv(os.path.join(self.directory, 'hole_speed.csv'), index=False)



    # METHOD DISTANCE_FROM_HOLE: CALCULATES DISTANCES FROM HOLE CENTROID  

    def distance_from_hole(self): 

        data = []

        for match in self.matching_pairs:  
            track_file = match['track_file']
            hole_boundary = match['hole_boundary']
            
            if hole_boundary is None:
                print(f"No hole boundary for track file: {track_file}")
                continue

            df = self.track_data[track_file]

            centroid = hole_boundary.centroid

            for index, row in df.iterrows():
                x, y = row['x_body'], row['y_body']
                distance = np.sqrt((centroid.x - x)**2 + (centroid.y - y)**2)
                data.append({'time': row.frame, 'distance_from_hole': distance, 'speed': row.speed, 'file': track_file})
    
        if not data:
            print("No distances calculated, check data")
        else:

            distance_hole_over_time = pd.DataFrame(data)
            distance_hole_over_time = distance_hole_over_time.sort_values(by=['time'], ascending=True)
            distance_hole_over_time.to_csv(os.path.join(self.directory, 'hole_distance.csv'), index=False)


    # METHOD HOLE_ORIENTATION: CALCULATES LARVAE ORIENTATION FROM THE HOLE

    def hole_orientation(self):

        def angle_calculator(vector_A, vector_B):
            # convert to an array for mathmatical ease 
            A = np.array(vector_A)
            B = np.array(vector_B)
            # calculate the dot product
            dot_product = np.dot(A, B)
            # calculate the magnitude of vector (length / norm of vector)
            magnitude_A = np.linalg.norm(vector_A)
            magnitude_B = np.linalg.norm(vector_B)
            # cosθ
            cos_theta = dot_product / (magnitude_A * magnitude_B)
            # θ in radians
            theta_radians = np.arccos(cos_theta)
            # θ in degrees
            theta_degrees = np.degrees(theta_radians)
            return theta_degrees
        
        data = []

        for match in self.matching_pairs:  
            track_file = match['track_file']
            hole_boundary = match['hole_boundary']
            
            if hole_boundary is None:
                print(f"No hole boundary for track file: {track_file}")
                continue

            df = self.track_data[track_file]

            centroid = hole_boundary.centroid

            hole = np.array([centroid.x, centroid.y])
            
            for row in df.itertuples(): # tuple of each row 

                body = np.array([row.x_body, row.y_body])
                head = np.array([row.x_head, row.y_head])

                # hole_body = np.array(centroid) - body 
                hole_body = hole - body
                body_head = head - body

                angle = angle_calculator(hole_body, body_head)

                frame = row.frame

                data.append({'time': frame, 'hole orientation': angle, 'file': track_file})

        hole_orientation_over_time = pd.DataFrame(data)
        hole_orientation_over_time = hole_orientation_over_time.sort_values(by=['time'], ascending=True)
        hole_orientation_over_time.to_csv(os.path.join(self.directory, 'hole_orientation.csv'), index=False)



        # METHOD HOLE_EUCLIDEAN_DISTANCE: EUCLIDEAN DISTANCE ACCOUNTING FOR LARVAE WITHIN THE HOLE 

    def hole_euclidean_distance(self):

        data = []

        for match in self.matching_pairs:
            track_file = match['track_file']
            hole_boundary = match['hole_boundary']

            # Ensure the perimeter polygon is available
            if hole_boundary is None:
                print(f"No perimeter polygon available for track file: {track_file}")
                continue

            track_data = self.track_data[track_file]

            for frame, fr in df.groupby('frame'):
                coords = fr[['x_body', 'y_body']].to_numpy()
                n = len(coords)
                if n < 2:
                    avg = np.nan   # not enough larvae to define a pairwise distance
                else:
                    avg = pdist(coords, metric='euclidean').mean()
                data.append({'time': frame, 'average_distance': avg, 'file': track_file})


        df = pd.DataFrame(data)
        df = df.sort_values(by=['time', 'file'], ascending=True)
        df.to_csv(os.path.join(self.directory, 'hole_euclidean_distances.csv'), index=False)





    # METHOD POTENTIAL_ENTRIES: NUMBER OF TIMES LARVAE ARE WITHIN VISCINITY OF HOLE BUT CHOSE NOT TO ENTER 
 
    def hole_entry_probability(self):
        data = []
        min_near_frames = 15

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file].sort_values(['track_id', 'frame'])

            for track_id, track_df in df.groupby('track_id'):
                near = track_df['within_10mm'].values.astype(bool)
                hole = track_df['within_hole'].values.astype(bool)
                frames = track_df['frame'].values

                i = 0
                while i <= len(near) - min_near_frames -1:
                    if all(near[i:i + min_near_frames]):

                        decision_frame = frames[i + min_near_frames]

                        # Look ahead to see if they enter the hole
                        entered = any(hole[i + min_near_frames : i + min_near_frames + 30])

                        others_inside = df[
                            (df['frame'] == decision_frame) &     # Only consider rows for the current decision frame
                            (df['track_id'] != track_id) &        # Exclude the current larva (we want others only)
                            (df['within_hole'])                   # Only count those larvae currently inside the hole
                        ].shape[0]                                # Get the number of such rows = number of others in hole

                        data.append({
                            'file': track_file,
                            'track': track_id,
                            'decision_frame': decision_frame,
                            'entry': entered,
                            'number_inside_hole': others_inside
                        })
                        
                        if entered:
                            # Skip until they leave both the hole AND the 10mm vicinity
                            j = i + min_near_frames + 30
                            while j < len(hole) and (hole[j] or near[j]):
                                j += 1
                            i = j
                        else:
                            # They didn't enter — skip ahead until they leave the 10mm vicinity
                            j = i + min_near_frames
                            while j < len(near) and near[j]:
                                j += 1
                            i = j
                    
                    else:
                        i += 1

        pd.DataFrame(data).to_csv(os.path.join(self.directory, 'hole_probability.csv'), index=False)

    
    # METHOD HOLE_STATUS: STATUS OF LARVAE REGARDING HOLE: NIAVE OR 

    def hole_status(self):

        for match in self.matching_pairs:
            track_file = match['track_file']
            df = self.track_data[track_file].sort_values(['track_id', 'frame']).copy()

            entry_frame_map = (
                df[df['within_hole']]
                .groupby('track_id')['frame']
                .min()
                .to_dict()
            )

            df['entry_frame'] = df['track_id'].map(entry_frame_map)
            

            df['hole_status'] = np.where(
                df['entry_frame'].isna(), 'naive',               # never entered
                np.where(df['frame'] < df['entry_frame'], 'naive', 'exposed')
            )

            df = df.drop(columns=['entry_frame'])
    
            self.track_data[track_file] = df
    

    # HOLE_STATUS_INTERACTION: TYPE OF INTERACTION OCCURING BETWEEN LARVAE 

    def hole_status_interactions(self, threshold=1.0):

        continue_threshold = 1.5

        def unify_interaction_type(part1, part2):
            return "_".join(sorted([part1, part2]))

        body_parts = ["head", "body", "tail"]
        interaction_pairs = list(itertools.product(body_parts, body_parts))
        unified_types = sorted(set(unify_interaction_type(p1, p2) for p1, p2 in interaction_pairs))

        bouts = []

        for track_file in self.track_files:
            df = self.track_data[track_file].copy()
            df.sort_values(by="frame", inplace=True)

            active_bouts = {}  # pair_key -> bout dict
            bout_counter = 0

            for frame in sorted(df["frame"].unique()):
                frame_data = df[df["frame"] == frame]
                track_ids = frame_data["track_id"].unique()

                coords = {
                    part: {
                        row["track_id"]: np.array([row[f"x_{part}"], row[f"y_{part}"]])
                        for _, row in frame_data.iterrows()
                    }
                    for part in body_parts
                }

                # START/true contacts (<1.0)
                interacting_pairs = {}  # pair_key -> list of interaction types
                # CONTINUE condition (<1.5)
                close_pairs = {}        # pair_key -> closest_type

                for id1, id2 in itertools.combinations(track_ids, 2):
                    interactions = []
                    min_dist = float("inf")
                    closest_type = None

                    for part1, part2 in interaction_pairs:
                        coord1 = coords[part1].get(id1)
                        coord2 = coords[part2].get(id2)
                        if coord1 is None or coord2 is None:
                            continue

                        dist = np.linalg.norm(coord1 - coord2)

                        if dist < min_dist:
                            min_dist = dist
                            closest_type = unify_interaction_type(part1, part2)

                        if dist < threshold:
                            interactions.append(unify_interaction_type(part1, part2))

                    pair_key = tuple(sorted((id1, id2)))

                    if closest_type is not None and min_dist < continue_threshold:
                        close_pairs[pair_key] = closest_type

                    if interactions:
                        interacting_pairs[pair_key] = interactions

                current_close = set(close_pairs.keys())

                # 1) END bouts no longer within 1.5mm
                for pair in list(active_bouts.keys()):
                    if pair not in current_close:
                        bout = active_bouts.pop(pair)
                        interactions_all = bout["interactions"]
                        if not interactions_all:
                            continue

                        start, end = bout["start_frame"], bout["end_frame"]
                        duration = end - start + 1
                        interaction_counts = Counter(interactions_all)
                        initial_type = interactions_all[0]
                        predominant_type = interaction_counts.most_common(1)[0][0]

                        status1_row = df[(df["track_id"] == pair[0]) & (df["frame"] == start)]
                        status2_row = df[(df["track_id"] == pair[1]) & (df["frame"] == start)]
                        status1 = status1_row["hole_status"].values[0] if not status1_row.empty else None
                        status2 = status2_row["hole_status"].values[0] if not status2_row.empty else None
                        hole_status_pair = "-".join(sorted([str(status1), str(status2)]))

                        bout_data = {
                            "file": track_file,
                            "bout_id": bout["bout_id"],
                            "track_1": pair[0],
                            "track_2": pair[1],
                            "start_frame": start,
                            "end_frame": end,
                            "interaction_duration": duration,
                            "initial_type": initial_type,
                            "predominant_type": predominant_type,
                            "hole_status_pair": hole_status_pair,
                        }
                        for t in unified_types:
                            bout_data[t] = interaction_counts.get(t, 0)
                        bouts.append(bout_data)

                # 2) UPDATE bouts still within 1.5mm
                for pair in list(active_bouts.keys()):
                    active_bouts[pair]["end_frame"] = frame
                    if pair in interacting_pairs:
                        active_bouts[pair]["interactions"].extend(interacting_pairs[pair])
                    else:
                        active_bouts[pair]["interactions"].append(close_pairs[pair])

                # 3) START new bouts ONLY if <1.0 this frame
                for pair, interactions in interacting_pairs.items():
                    if pair in active_bouts:
                        continue
                    active_bouts[pair] = {
                        "bout_id": bout_counter,
                        "start_frame": frame,
                        "end_frame": frame,
                        "interactions": interactions.copy(),
                    }
                    bout_counter += 1

            # Finalize remaining bouts at end of file
            for pair, bout in active_bouts.items():
                interactions_all = bout["interactions"]
                if not interactions_all:
                    continue

                start, end = bout["start_frame"], bout["end_frame"]
                duration = end - start + 1
                interaction_counts = Counter(interactions_all)
                initial_type = interactions_all[0]
                predominant_type = interaction_counts.most_common(1)[0][0]

                status1_row = df[(df["track_id"] == pair[0]) & (df["frame"] == start)]
                status2_row = df[(df["track_id"] == pair[1]) & (df["frame"] == start)]
                status1 = status1_row["hole_status"].values[0] if not status1_row.empty else None
                status2 = status2_row["hole_status"].values[0] if not status2_row.empty else None
                hole_status_pair = "-".join(sorted([str(status1), str(status2)]))

                bout_data = {
                    "file": track_file,
                    "bout_id": bout["bout_id"],
                    "track_1": pair[0],
                    "track_2": pair[1],
                    "start_frame": start,
                    "end_frame": end,
                    "interaction_duration": duration,
                    "initial_type": initial_type,
                    "predominant_type": predominant_type,
                    "hole_status_pair": hole_status_pair,
                }
                for t in unified_types:
                    bout_data[t] = interaction_counts.get(t, 0)
                bouts.append(bout_data)

        bout_df = pd.DataFrame(bouts).sort_values(by=["file", "bout_id"])
        bout_df.to_csv(os.path.join(self.directory, "interaction_status_type.csv"), index=False)
        return bout_df




    # def hole_status_interactions(self, threshold=1): # modified interaction_type_bout method

    #     max_gap = 4

    #     def unify_interaction_type(part1, part2):
    #         return '_'.join(sorted([part1, part2]))
        
    #     def get_closest_part_pair(coords, id1, id2):
    #         min_dist = float('inf')
    #         closest = None
    #         for p1 in coords:
    #             for p2 in coords:
    #                 coord1 = coords[p1].get(id1)
    #                 coord2 = coords[p2].get(id2)
    #                 if coord1 is None or coord2 is None:
    #                     continue
    #                 dist = np.linalg.norm(coord1 - coord2)
    #                 if dist < min_dist:
    #                     min_dist = dist
    #                     closest = unify_interaction_type(p1, p2)
    #         return closest

    #     body_parts = ['head', 'body', 'tail']
    #     interaction_pairs = list(itertools.product(body_parts, body_parts))
    #     unified_types = sorted(set(unify_interaction_type(p1, p2) for p1, p2 in interaction_pairs))

    #     bouts = []

    #     for track_file in self.track_files:
    #         df = self.track_data[track_file].copy()
    #         df.sort_values(by='frame', inplace=True)

    #         active_bouts = {}
    #         bout_counter = 0

    #         for frame in df['frame'].unique():
    #             frame_data = df[df['frame'] == frame]
    #             track_ids = frame_data['track_id'].unique()

    #             # Coordinates lookup
    #             coords = {
    #                 part: {
    #                     row['track_id']: np.array([row[f'x_{part}'], row[f'y_{part}']])
    #                     for _, row in frame_data.iterrows()
    #                 }
    #                 for part in body_parts
    #             }

    #             interacting_pairs = {}

    #             for id1, id2 in itertools.combinations(track_ids, 2):
    #                 interactions = []

    #                 for part1, part2 in interaction_pairs:
    #                     coord1 = coords[part1].get(id1)
    #                     coord2 = coords[part2].get(id2)
    #                     if coord1 is None or coord2 is None:
    #                         continue
    #                     dist = np.linalg.norm(coord1 - coord2)
    #                     if dist < threshold:
    #                         interactions.append(unify_interaction_type(part1, part2))

    #                 if interactions:
    #                     # interacting_pairs[(id1, id2)] = interactions
    #                     pair_key = tuple(sorted((id1, id2)))
    #                     interacting_pairs[pair_key] = interactions

    #             current_pairs = set(interacting_pairs)

    #             # Process ended or gap-extending bouts
    #             for pair in list(active_bouts):
    #                 if pair not in current_pairs:
    #                     active_bouts[pair]['gap_count'] += 1
    #                     if active_bouts[pair]['gap_count'] <= max_gap:
    #                         id1, id2 = pair
    #                         closest_type = None
    #                         min_dist = float('inf')
    #                         for part1, part2 in interaction_pairs:
    #                             coord1 = coords[part1].get(id1)
    #                             coord2 = coords[part2].get(id2)
    #                             if coord1 is None or coord2 is None:
    #                                 continue
    #                             dist = np.linalg.norm(coord1 - coord2)
    #                             if dist < min_dist:
    #                                 min_dist = dist
    #                                 closest_type = unify_interaction_type(part1, part2)
    #                         if closest_type:
    #                             active_bouts[pair]['interactions'].append(closest_type)
    #                             active_bouts[pair]['end_frame'] = frame
    #                     else:
    #                         # End bout
    #                         bout = active_bouts.pop(pair)
    #                         start, end = bout['start_frame'], bout['end_frame']
    #                         duration = end - start + 1
    #                         interactions = bout['interactions']
    #                         interaction_counts = Counter(interactions)
    #                         initial_type = interactions[0]
    #                         predominant_type = interaction_counts.most_common(1)[0][0]
    #                         status1 = df[(df['track_id'] == pair[0]) & (df['frame'] == start)]['hole_status'].values[0]
    #                         status2 = df[(df['track_id'] == pair[1]) & (df['frame'] == start)]['hole_status'].values[0]
    #                         hole_status_pair = '-'.join(sorted([status1, status2]))
    #                         bout_data = {
    #                             'file': track_file,
    #                             'bout_id': bout['bout_id'],
    #                             'track_1': pair[0],
    #                             'track_2': pair[1],
    #                             'start_frame': start,
    #                             'end_frame': end,
    #                             'interaction_duration': duration,
    #                             'initial_type': initial_type,
    #                             'predominant_type': predominant_type,
    #                             'hole_status_pair': hole_status_pair,
    #                         }
    #                         for t in unified_types:
    #                             bout_data[t] = interaction_counts.get(t, 0)
    #                         bouts.append(bout_data)

    #             # Update or start new bouts
    #             for pair, interactions in interacting_pairs.items():
    #                 if pair in active_bouts:
    #                     active_bouts[pair]['end_frame'] = frame
    #                     active_bouts[pair]['interactions'].extend(interactions)
    #                     active_bouts[pair]['gap_count'] = 0
    #                 else:
    #                     active_bouts[pair] = {
    #                         'bout_id': bout_counter,
    #                         'start_frame': frame,
    #                         'end_frame': frame,
    #                         'interactions': interactions.copy(),
    #                         'gap_count': 0
    #                     }
    #                     bout_counter += 1

    #         # Finalize remaining bouts
    #         for pair, bout in active_bouts.items():
    #             start, end = bout['start_frame'], bout['end_frame']
    #             duration = end - start + 1
    #             interactions = bout['interactions']
    #             interaction_counts = Counter(interactions)
    #             initial_type = interactions[0]
    #             predominant_type = interaction_counts.most_common(1)[0][0]
    #             status1 = df[(df['track_id'] == pair[0]) & (df['frame'] == start)]['hole_status'].values[0]
    #             status2 = df[(df['track_id'] == pair[1]) & (df['frame'] == start)]['hole_status'].values[0]
    #             hole_status_pair = '-'.join(sorted([status1, status2]))
    #             bout_data = {
    #                 'file': track_file,
    #                 'bout_id': bout['bout_id'],
    #                 'track_1': pair[0],
    #                 'track_2': pair[1],
    #                 'start_frame': start,
    #                 'end_frame': end,
    #                 'interaction_duration': duration,
    #                 'initial_type': initial_type,
    #                 'predominant_type': predominant_type,
    #                 'hole_status_pair': hole_status_pair,
    #             }
    #             for t in unified_types:
    #                 bout_data[t] = interaction_counts.get(t, 0)
    #             bouts.append(bout_data)

    #     bout_df = pd.DataFrame(bouts).sort_values(by=['file', 'bout_id'])
    #     bout_df.to_csv(os.path.join(self.directory, 'interaction_status_type.csv'), index=False)
    #     return bout_df
    


  #### old logic 
        #         ended_pairs = set(active_bouts.keys()) - set(interacting_pairs.keys())

        #         for pair in ended_pairs:
        #             bout = active_bouts.pop(pair)
        #             start, end = bout['start_frame'], bout['end_frame']
        #             duration = end - start + 1
        #             interactions = bout['interactions']

        #             interaction_counts = Counter(interactions)
        #             initial_type = interactions[0]
        #             predominant_type = interaction_counts.most_common(1)[0][0]

        #             # Status at start of bout
        #             status1 = df[(df['track_id'] == pair[0]) & (df['frame'] == start)]['hole_status'].values[0]
        #             status2 = df[(df['track_id'] == pair[1]) & (df['frame'] == start)]['hole_status'].values[0]
        #             hole_status_pair = '-'.join(sorted([status1, status2]))

        #             bout_data = {
        #                 'file': track_file,
        #                 'bout_id': bout['bout_id'],
        #                 'track_1': pair[0],
        #                 'track_2': pair[1],
        #                 'start_frame': start,
        #                 'end_frame': end,
        #                 'duration': duration,
        #                 'initial_type': initial_type,
        #                 'predominant_type': predominant_type,
        #                 'hole_status_pair': hole_status_pair,
        #             }

        #             for t in unified_types:
        #                 bout_data[t] = interaction_counts.get(t, 0)

        #             bouts.append(bout_data)

        #         for pair, interactions in interacting_pairs.items():
        #             if pair in active_bouts:
        #                 active_bouts[pair]['end_frame'] = frame
        #                 active_bouts[pair]['interactions'].extend(interactions)
        #             else:
        #                 active_bouts[pair] = {
        #                     'bout_id': bout_counter,
        #                     'start_frame': frame,
        #                     'end_frame': frame,
        #                     'interactions': interactions.copy(),
        #                 }
        #                 bout_counter += 1

        #     for pair, bout in active_bouts.items():
        #         start, end = bout['start_frame'], bout['end_frame']
        #         duration = end - start + 1
        #         interactions = bout['interactions']
        #         interaction_counts = Counter(interactions)
        #         initial_type = interactions[0]
        #         predominant_type = interaction_counts.most_common(1)[0][0]

        #         status1 = df[(df['track_id'] == pair[0]) & (df['frame'] == start)]['hole_status'].values[0]
        #         status2 = df[(df['track_id'] == pair[1]) & (df['frame'] == start)]['hole_status'].values[0]
        #         hole_status_pair = '-'.join(sorted([status1, status2]))

        #         bout_data = {
        #             'file': track_file,
        #             'bout_id': bout['bout_id'],
        #             'track_1': pair[0],
        #             'track_2': pair[1],
        #             'start_frame': start,
        #             'end_frame': end,
        #             'duration': duration,
        #             'initial_type': initial_type,
        #             'predominant_type': predominant_type,
        #             'hole_status_pair': hole_status_pair,
        #         }

        #         for t in unified_types:
        #             bout_data[t] = interaction_counts.get(t, 0)

        #         bouts.append(bout_data)

        # bout_df = pd.DataFrame(bouts).sort_values(by=['file', 'bout_id'])
        # bout_df.to_csv(os.path.join(self.directory, 'interaction_type.csv'), index=False)
        # return bout_df
    

    # def interactions_return(self, threshold=1):

    #     max_gap = 4

    #     def unify_interaction_type(p1, p2):
    #         return '_'.join(sorted([p1, p2]))
        
    #     def get_closest_part_pair(coords, id1, id2):
    #         min_dist = float('inf')
    #         closest = None
    #         for p1 in coords:
    #             for p2 in coords:
    #                 coord1 = coords[p1].get(id1)
    #                 coord2 = coords[p2].get(id2)
    #                 if coord1 is None or coord2 is None:
    #                     continue
    #                 dist = np.linalg.norm(coord1 - coord2)
    #                 if dist < min_dist:
    #                     min_dist = dist
    #                     closest = unify_interaction_type(p1, p2)
    #         return closest

    #     body_parts = ['head', 'body', 'tail']
    #     interaction_pairs = list(itertools.product(body_parts, body_parts))
    #     unified_types = sorted(set(unify_interaction_type(p1, p2) for p1, p2 in interaction_pairs))

    #     bouts = []

    #     for track_file in self.track_files:
    #         df = self.track_data[track_file].copy()
    #         df.sort_values(by='frame', inplace=True)

    #         # Get exposed larvae (they have an entry_frame)
    #         entry_frames = df[df['within_hole']].groupby('track_id')['frame'].min()
    #         exposed_ids = entry_frames.index.tolist()

    #         for larva_id in exposed_ids: #filter for larvae which entered the hole
    #             sub = df[df['track_id'] == larva_id].copy()
    #             sub = sub.sort_values(by='frame')

    #             # Find all exit (True→False) and return (False→True) transitions
    #             sub['prev'] = sub['within_hole'].shift()
    #             exits = sub[(sub['within_hole'] == False) & (sub['prev'] == True)]
    #             returns = sub[(sub['within_hole'] == True) & (sub['prev'] == False)]

    #             exit_frames = exits['frame'].values
    #             return_frames = returns['frame'].values

    #             # Handle matching exits to returns
    #             for i, exit_frame in enumerate(exit_frames):
    #                 later_returns = return_frames[return_frames > exit_frame]
    #                 if len(later_returns) > 0:
    #                     reentry_frame = later_returns[0]
    #                     returned = True
    #                     return_duration = reentry_frame - exit_frame + 1
    #                 else:
    #                     reentry_frame = sub['frame'].max()
    #                     returned = False
    #                     return_duration = False
                        
    #                 # Get all frames between exit and reentry
    #                 window = df[
    #                     (df['frame'] >= exit_frame) & 
    #                     (df['frame'] <= reentry_frame)
    #                 ]

    #                 # Get interaction bouts in this window involving the exiting larva
    #                 active_bouts = {}
    #                 bout_counter = 0
    #                 bouts_this_exit = []

    #                 for frame in window['frame'].unique():
    #                     frame_data = window[window['frame'] == frame]
    #                     track_ids = frame_data['track_id'].unique()

    #                     coords = {
    #                         part: {
    #                             row['track_id']: np.array([row[f'x_{part}'], row[f'y_{part}']])
    #                             for _, row in frame_data.iterrows()
    #                         }
    #                         for part in body_parts
    #                     }

    #                     interacting_pairs = {}
    #                     for id1, id2 in itertools.combinations(track_ids, 2):
    #                         if larva_id not in (id1, id2):
    #                             continue  # Only care about interactions involving this larva

    #                         partner_id = id2 if id1 == larva_id else id1  # Get the other larva
    #                         partner_row = frame_data[frame_data['track_id'] == partner_id]

    #                         if partner_row.empty or partner_row['within_hole'].values[0]:
    #                             continue  # Skip if partner is in hole

    #                         interactions = []
    #                         for part1, part2 in interaction_pairs:
    #                             coord1 = coords[part1].get(id1)
    #                             coord2 = coords[part2].get(id2)
    #                             if coord1 is None or coord2 is None:
    #                                 continue
    #                             if np.linalg.norm(coord1 - coord2) < threshold:
    #                                 interactions.append(unify_interaction_type(part1, part2))

    #                         if interactions:
    #                             # interacting_pairs[(id1, id2)] = interactions
    #                             pair_key = tuple(sorted((id1, id2)))
    #                             interacting_pairs[pair_key] = interactions


    #                     current_pairs = set(interacting_pairs)
                        

    #                     # Process ended or extended gaps
    #                     for pair in list(active_bouts):
    #                         if pair not in current_pairs:
    #                             active_bouts[pair]['gap_count'] += 1
    #                             if active_bouts[pair]['gap_count'] <= max_gap:
    #                                 id1, id2 = pair
    #                                 fallback = get_closest_part_pair(coords, id1, id2)
    #                                 if fallback:
    #                                     active_bouts[pair]['interactions'].append(fallback)
    #                                 active_bouts[pair]['end_frame'] = frame
    #                             else:
    #                                 # End bout
    #                                 bout = active_bouts.pop(pair)
    #                                 start, end = bout['start_frame'], bout['end_frame']
    #                                 interactions = bout['interactions']
    #                                 interaction_counts = Counter(interactions)
    #                                 initial_type = interactions[0]
    #                                 predominant_type = interaction_counts.most_common(1)[0][0]
    #                                 partner = [x for x in pair if x != larva_id][0]
    #                                 partner_status = df[(df['track_id'] == partner) & (df['frame'] == start)]['hole_status'].values[0]

    #                                 bout_data = {
    #                                     'file': track_file,
    #                                     'exiting_larva': larva_id,
    #                                     'exit_index': i,
    #                                     'returned_to_hole': returned,
    #                                     'start_frame': start,
    #                                     'end_frame': end,
    #                                     'return_time': return_duration,
    #                                     'interacted': True,
    #                                     'partner': partner,
    #                                     'partner_status': partner_status,
    #                                     'duration': end - start + 1,
    #                                     'initial_type': initial_type,
    #                                     'predominant_type': predominant_type,
    #                                 }
    #                                 for t in unified_types:
    #                                     bout_data[t] = interaction_counts.get(t, 0)
    #                                 bouts_this_exit.append(bout_data)

    #                     # Extend or start bouts
    #                     for pair, interactions in interacting_pairs.items():
    #                         if pair in active_bouts:
    #                             active_bouts[pair]['end_frame'] = frame
    #                             active_bouts[pair]['interactions'].extend(interactions)
    #                             active_bouts[pair]['gap_count'] = 0
    #                         else:
    #                             active_bouts[pair] = {
    #                                 'bout_id': bout_counter,
    #                                 'start_frame': frame,
    #                                 'end_frame': frame,
    #                                 'interactions': interactions.copy(),
    #                                 'gap_count': 0
    #                             }
    #                             bout_counter += 1

    #                 # Close remaining bouts
    #                 for pair, bout in active_bouts.items():
    #                     start, end = bout['start_frame'], bout['end_frame']
    #                     interactions = bout['interactions']
    #                     interaction_counts = Counter(interactions)
    #                     initial_type = interactions[0]
    #                     predominant_type = interaction_counts.most_common(1)[0][0]
    #                     partner = [x for x in pair if x != larva_id][0]
    #                     partner_status = df[(df['track_id'] == partner) & (df['frame'] == start)]['hole_status'].values[0]

    #                     bout_data = {
    #                         'file': track_file,
    #                         'exiting_larva': larva_id,
    #                         'exit_index': i,
    #                         'returned_to_hole': returned,
    #                         'start_frame': start,
    #                         'end_frame': end,
    #                         'return_time': return_duration,
    #                         'interacted': True,
    #                         'partner': partner,
    #                         'partner_status': partner_status,
    
    #                         'duration': end - start + 1,
    #                         'initial_type': initial_type,
    #                         'predominant_type': predominant_type,
    #                     }
    #                     for t in unified_types:
    #                         bout_data[t] = interaction_counts.get(t, 0)
    #                     bouts_this_exit.append(bout_data)

    #                 if not bouts_this_exit:
    #                     fallback = {
    #                         'file': track_file,
    #                         'exiting_larva': larva_id,
    #                         'exit_index': i,
    #                         'returned_to_hole': returned,
    #                         'start_frame': exit_frame,
    #                         'end_frame': reentry_frame,
    #                         'return_time': return_duration,
    #                         'interacted': False,
    #                         'partner': None,
    #                         'partner_status': None,
    #                         'duration': None,
    #                         'initial_type': None,
    #                         'predominant_type': None,
    #                     }
    #                     for t in unified_types:
    #                         fallback[t] = None
    #                     bouts_this_exit.append(fallback)

    #                 bouts.extend(bouts_this_exit)

    #     bout_df = pd.DataFrame(bouts).sort_values(by=['file', 'exiting_larva', 'exit_index', 'start_frame'])
    #     bout_df.to_csv(os.path.join(self.directory, 'interactions_return.csv'), index=False)
    #     return bout_df

    def interactions_return(self, threshold=1.0):

        continue_threshold = 1.5

        def unify_interaction_type(p1, p2):
            return "_".join(sorted([p1, p2]))

        body_parts = ["head", "body", "tail"]
        interaction_pairs = list(itertools.product(body_parts, body_parts))
        unified_types = sorted(set(unify_interaction_type(p1, p2) for p1, p2 in interaction_pairs))

        bouts = []

        for track_file in self.track_files:
            df = self.track_data[track_file].copy()
            df.sort_values(by="frame", inplace=True)

            # exposed larvae = have an entry frame
            entry_frames = df[df["within_hole"]].groupby("track_id")["frame"].min()
            exposed_ids = entry_frames.index.tolist()

            for larva_id in exposed_ids:
                sub = df[df["track_id"] == larva_id].copy().sort_values(by="frame")

                # exits (True->False) and returns (False->True)
                sub["prev"] = sub["within_hole"].shift()
                exits = sub[(sub["within_hole"] == False) & (sub["prev"] == True)]
                returns = sub[(sub["within_hole"] == True) & (sub["prev"] == False)]

                exit_frames = exits["frame"].values
                return_frames = returns["frame"].values

                for i, exit_frame in enumerate(exit_frames):
                    later_returns = return_frames[return_frames > exit_frame]
                    if len(later_returns) > 0:
                        reentry_frame = later_returns[0]
                        returned = True
                        return_duration = reentry_frame - exit_frame + 1
                    else:
                        reentry_frame = sub["frame"].max()
                        returned = False
                        return_duration = False

                    # frames between exit and reentry
                    window = df[(df["frame"] >= exit_frame) & (df["frame"] <= reentry_frame)].copy()

                    active_bouts = {}   # pair_key -> bout dict
                    bout_counter = 0
                    bouts_this_exit = []
                    had_bout = False

                    for frame in sorted(window["frame"].unique()):
                        frame_data = window[window["frame"] == frame]
                        track_ids = frame_data["track_id"].unique()

                        coords = {
                            part: {
                                row["track_id"]: np.array([row[f"x_{part}"], row[f"y_{part}"]])
                                for _, row in frame_data.iterrows()
                            }
                            for part in body_parts
                        }

                        # START: any <1.0 contacts
                        interacting_pairs = {}  # pair_key -> list of interaction types (<1.0)
                        # CONTINUE: min_dist <1.5
                        close_pairs = {}        # pair_key -> closest_type

                        for id1, id2 in itertools.combinations(track_ids, 2):
                            if larva_id not in (id1, id2):
                                continue  # only bouts involving this larva

                            partner_id = id2 if id1 == larva_id else id1
                            partner_row = frame_data[frame_data["track_id"] == partner_id]

                            # skip if partner missing or partner in hole
                            if partner_row.empty or partner_row["within_hole"].values[0]:
                                continue

                            interactions = []
                            min_dist = float("inf")
                            closest_type = None

                            for part1, part2 in interaction_pairs:
                                coord1 = coords[part1].get(id1)
                                coord2 = coords[part2].get(id2)
                                if coord1 is None or coord2 is None:
                                    continue

                                dist = np.linalg.norm(coord1 - coord2)

                                if dist < min_dist:
                                    min_dist = dist
                                    closest_type = unify_interaction_type(part1, part2)

                                if dist < threshold:
                                    interactions.append(unify_interaction_type(part1, part2))

                            pair_key = tuple(sorted((id1, id2)))

                            if closest_type is not None and min_dist < continue_threshold:
                                close_pairs[pair_key] = closest_type

                            if interactions:
                                interacting_pairs[pair_key] = interactions

                        current_close = set(close_pairs.keys())

                        # 1) END bouts no longer within 1.5mm
                        for pair in list(active_bouts.keys()):
                            if pair not in current_close:
                                bout = active_bouts.pop(pair)
                                interactions_all = bout["interactions"]
                                if not interactions_all:
                                    continue

                                start, end = bout["start_frame"], bout["end_frame"]
                                interaction_counts = Counter(interactions_all)
                                initial_type = interactions_all[0]
                                predominant_type = interaction_counts.most_common(1)[0][0]

                                partner = [x for x in pair if x != larva_id][0]
                                partner_status_row = df[(df["track_id"] == partner) & (df["frame"] == start)]
                                partner_status = (
                                    partner_status_row["hole_status"].values[0]
                                    if not partner_status_row.empty else None
                                )

                                bout_data = {
                                    "file": track_file,
                                    "exiting_larva": larva_id,
                                    "exit_index": i,
                                    "returned_to_hole": returned,
                                    "start_frame": start,
                                    "end_frame": end,
                                    "return_time": return_duration,
                                    "interacted": True,
                                    "partner": partner,
                                    "partner_status": partner_status,
                                    "duration": end - start + 1,
                                    "initial_type": initial_type,
                                    "predominant_type": predominant_type,
                                }
                                for t in unified_types:
                                    bout_data[t] = interaction_counts.get(t, 0)

                                bouts_this_exit.append(bout_data)
                                had_bout = True

                        # 2) UPDATE bouts still within 1.5mm
                        for pair in list(active_bouts.keys()):
                            active_bouts[pair]["end_frame"] = frame
                            if pair in interacting_pairs:
                                active_bouts[pair]["interactions"].extend(interacting_pairs[pair])
                            else:
                                active_bouts[pair]["interactions"].append(close_pairs[pair])

                        # 3) START new bouts only if <1.0 this frame
                        for pair, interactions in interacting_pairs.items():
                            if pair in active_bouts:
                                continue
                            active_bouts[pair] = {
                                "bout_id": bout_counter,
                                "start_frame": frame,
                                "end_frame": frame,
                                "interactions": interactions.copy(),
                            }
                            bout_counter += 1

                    # finalize remaining bouts in this window
                    for pair, bout in active_bouts.items():
                        interactions_all = bout["interactions"]
                        if not interactions_all:
                            continue

                        start, end = bout["start_frame"], bout["end_frame"]
                        interaction_counts = Counter(interactions_all)
                        initial_type = interactions_all[0]
                        predominant_type = interaction_counts.most_common(1)[0][0]

                        partner = [x for x in pair if x != larva_id][0]
                        partner_status_row = df[(df["track_id"] == partner) & (df["frame"] == start)]
                        partner_status = (
                            partner_status_row["hole_status"].values[0]
                            if not partner_status_row.empty else None
                        )

                        bout_data = {
                            "file": track_file,
                            "exiting_larva": larva_id,
                            "exit_index": i,
                            "returned_to_hole": returned,
                            "start_frame": start,
                            "end_frame": end,
                            "return_time": return_duration,
                            "interacted": True,
                            "partner": partner,
                            "partner_status": partner_status,
                            "duration": end - start + 1,
                            "initial_type": initial_type,
                            "predominant_type": predominant_type,
                        }
                        for t in unified_types:
                            bout_data[t] = interaction_counts.get(t, 0)

                        bouts_this_exit.append(bout_data)
                        had_bout = True

                    # fallback row if no interactions occurred in this exit->reentry window
                    if not had_bout:
                        fallback = {
                            "file": track_file,
                            "exiting_larva": larva_id,
                            "exit_index": i,
                            "returned_to_hole": returned,
                            "start_frame": exit_frame,
                            "end_frame": reentry_frame,
                            "return_time": return_duration,
                            "interacted": False,
                            "partner": None,
                            "partner_status": None,
                            "duration": None,
                            "initial_type": None,
                            "predominant_type": None,
                        }
                        for t in unified_types:
                            fallback[t] = None
                        bouts_this_exit.append(fallback)

                    bouts.extend(bouts_this_exit)

        bout_df = pd.DataFrame(bouts).sort_values(by=["file", "exiting_larva", "exit_index", "start_frame"])
        bout_df.to_csv(os.path.join(self.directory, "interactions_return.csv"), index=False)
        return bout_df




                        
                            

        #                 ended_pairs = set(active_bouts.keys()) - set(interacting_pairs.keys())

        #                 for pair in ended_pairs:
        #                     bout = active_bouts.pop(pair)
        #                     start, end = bout['start_frame'], bout['end_frame']
        #                     duration = end - start + 1
        #                     interactions = bout['interactions']
        #                     interaction_counts = Counter(interactions)
        #                     initial_type = interactions[0]
        #                     predominant_type = interaction_counts.most_common(1)[0][0]

        #                     other_id = [i for i in pair if i != larva_id][0]
        #                     # status1 = df[(df['track_id'] == larva_id) & (df['frame'] == start)]['hole_status'].values[0]
        #                     partner_status = df[(df['track_id'] == other_id) & (df['frame'] == start)]['hole_status'].values[0]
        #                     # hole_status_pair = '-'.join(sorted([status1, status2]))

        #                     bout_data = {
        #                         'file': track_file,
        #                         'exiting_larva': larva_id,
        #                         'exit_index': i,
        #                         'returned_to_hole': returned,
        #                         'return_time': return_duration,
        #                         # 'track_1': pair[0],
        #                         'interacted': True,
        #                         'partner': other_id,
        #                         'start_frame': start,
        #                         'end_frame': end,
        #                         'duration': duration,
        #                         'initial_type': initial_type,
        #                         'predominant_type': predominant_type,
        #                         'partner_status': partner_status,
        #                     }
        #                     for t in unified_types:
        #                         bout_data[t] = interaction_counts.get(t, 0)

        #                     bouts.append(bout_data)

        #                 # Extend or start bouts
        #                 for pair, interactions in interacting_pairs.items():
        #                     if pair in active_bouts:
        #                         active_bouts[pair]['end_frame'] = frame
        #                         active_bouts[pair]['interactions'].extend(interactions)
        #                     else:
        #                         active_bouts[pair] = {
        #                             'start_frame': frame,
        #                             'end_frame': frame,
        #                             'interactions': interactions.copy(),
        #                         }
                        
        #                 # ✅ After processing all frames in this window, add a fallback row if no interactions occurred
        #                 has_bouts = any(
        #                     (row['exiting_larva'] == larva_id) and (row['exit_index'] == i)
        #                     for row in bouts)

        #                 if not has_bouts:
        #                     bouts.append({
        #                         'file': track_file,
        #                         'exiting_larva': larva_id,
        #                         'exit_index': i,
        #                         'returned_to_hole': returned,
        #                         'return_time': False,
        #                         'interacted': False,
        #                         'partner': None,
        #                         'start_frame': exit_frame,
        #                         'end_frame': reentry_frame,
        #                         'duration': reentry_frame - exit_frame + 1,
        #                         'initial_type': None,
        #                         'predominant_type': None,
        #                         'partner_status': None,
        #                         **{t: None for t in unified_types}
        #                     })

        #             # Close any active bouts left at the end
        #             for pair, bout in active_bouts.items():
        #                 start, end = bout['start_frame'], bout['end_frame']
        #                 duration = end - start + 1
        #                 interactions = bout['interactions']
        #                 interaction_counts = Counter(interactions)
        #                 initial_type = interactions[0]
        #                 predominant_type = interaction_counts.most_common(1)[0][0]

        #                 other_id = [i for i in pair if i != larva_id][0]
        #                 # status1 = df[(df['track_id'] == larva_id) & (df['frame'] == start)]['hole_status'].values[0]
        #                 partner_status = df[(df['track_id'] == other_id) & (df['frame'] == start)]['hole_status'].values[0]
        #                 # hole_status_pair = '-'.join(sorted([status1, status2]))

        #                 bout_data = {
        #                     'file': track_file,
        #                     'exiting_larva': larva_id,
        #                     'exit_index': i,
        #                     'returned_to_hole': returned,
        #                     # 'track_1': pair[0],
        #                     'interacted': True,
        #                     'partner': other_id,
        #                     'start_frame': start,
        #                     'end_frame': end,
        #                     'duration': duration,
        #                     'initial_type': initial_type,
        #                     'predominant_type': predominant_type,
        #                     'partner_status': partner_status,
        #                 }
        #                 for t in unified_types:
        #                     bout_data[t] = interaction_counts.get(t, 0)

        #                 bouts.append(bout_data)

        # # Final output
        # bout_df = pd.DataFrame(bouts).sort_values(by=['file', 'exiting_larva', 'exit_index', 'start_frame'])
        # bout_df.to_csv(os.path.join(self.directory, 'interactions_return.csv'), index=False)
        # return bout_df
    



    # METHOD PRE_POST_HOLE_INTERACTION:

    # def pre_post_hole_interactions(self, threshold=1):
        
    #     max_gap = 4

    #     def unify_interaction_type(p1, p2):
    #         return '_'.join(sorted([p1, p2]))
        
    #     def get_closest_part_pair(coords, id1, id2):
    #         min_dist = float('inf')
    #         closest = None
    #         for p1 in coords:
    #             for p2 in coords:
    #                 coord1 = coords[p1].get(id1)
    #                 coord2 = coords[p2].get(id2)
    #                 if coord1 is None or coord2 is None:
    #                     continue
    #                 dist = np.linalg.norm(coord1 - coord2)
    #                 if dist < min_dist:
    #                     min_dist = dist
    #                     closest = unify_interaction_type(p1, p2)
    #         return closest

    #     body_parts = ['head', 'body', 'tail']
    #     interaction_pairs = list(itertools.product(body_parts, body_parts))
    #     unified_types = sorted(set(unify_interaction_type(p1, p2) for p1, p2 in interaction_pairs))

    #     bouts = []

    #     for track_file in self.track_files:
    #         df = self.track_data[track_file].copy()
    #         df.sort_values(by='frame', inplace=True)

    #         entry_map = df[df['within_hole']].groupby('track_id')['frame'].min().to_dict()


    #         exit_map = (
    #             df[df['track_id'].isin(entry_map.keys())]
    #             .assign(prev=lambda d: d.groupby('track_id')['within_hole'].shift())
    #             .query('within_hole == False and prev == True')
    #             .groupby('track_id')['frame'].min()
    #             .to_dict()
    #         )

    #         df = df[df['within_hole'] == False].copy()

    #         for track_id in entry_map:
    #             if track_id not in exit_map:
    #                 continue

    #             entry_frame = entry_map[track_id]
    #             exit_frame = exit_map[track_id]


    #             pre_mask = (df['track_id'] == track_id) & (df['frame'] < entry_frame) & (~df['within_hole'])
    #             post_mask = (df['track_id'] == track_id) & (df['frame'] > exit_frame) & (~df['within_hole'])

    #             duration_pre = pre_mask.sum()
    #             duration_post = post_mask.sum()
    #             duration = min(duration_pre, duration_post)

    #             if duration < 1:
    #                 continue

    #             pre_frames = df[pre_mask].nlargest(duration, 'frame')['frame'] # frames closest to hole before entry 
    #             post_frames = df[post_mask].nsmallest(duration, 'frame')['frame'] # frames closest to hole

    #             for label, frames in [('pre', pre_frames), ('post', post_frames)]:
    #                 subset = df[df['frame'].isin(frames)].copy()
    #                 active_bouts = {}

    #                 for frame in subset['frame'].unique():
    #                     frame_data = subset[subset['frame'] == frame]
    #                     track_ids = frame_data['track_id'].unique()

    #                     coords = {
    #                         part: {
    #                             row['track_id']: np.array([row[f'x_{part}'], row[f'y_{part}']])
    #                             for _, row in frame_data.iterrows()
    #                         }
    #                         for part in body_parts
    #                     }

    #                     interacting_pairs = {}
    #                     for id1, id2 in itertools.combinations(track_ids, 2):
    #                         if track_id not in (id1, id2):
    #                             continue

    #                         partner_id = id2 if id1 == track_id else id1  # The other larva in the pair
    #                         partner_row = frame_data[frame_data['track_id'] == partner_id]

    #                         if partner_row.empty or partner_row['within_hole'].values[0]:
    #                             continue  # Partner is either missing or currently in hole — skip this interaction
                            

    #                         interactions = []
    #                         for part1, part2 in interaction_pairs:
    #                             coord1 = coords[part1].get(id1)
    #                             coord2 = coords[part2].get(id2)
    #                             if coord1 is None or coord2 is None:
    #                                 continue
    #                             if np.linalg.norm(coord1 - coord2) < threshold:
    #                                 interactions.append(unify_interaction_type(part1, part2))

    #                         if interactions:
    #                             # interacting_pairs[(id1, id2)] = interactions
    #                             pair_key = tuple(sorted((id1, id2)))
    #                             interacting_pairs[pair_key] = interactions


                            
    #                     current_pairs = set(interacting_pairs.keys())

    #                     for pair in list(active_bouts.keys()):
    #                         if pair not in current_pairs:
    #                             active_bouts[pair]['gap_count'] += 1
    #                             if active_bouts[pair]['gap_count'] <= max_gap:
    #                                 id1, id2 = pair
    #                                 fallback = get_closest_part_pair(coords, id1, id2)
    #                                 if fallback:
    #                                     active_bouts[pair]['interactions'].append(fallback)
    #                             else:
    #                                 bout = active_bouts.pop(pair)
    #                                 start, end = bout['start_frame'], bout['end_frame']
    #                                 duration_bout = end - start + 1
    #                                 interaction_counts = Counter(bout['interactions'])
    #                                 initial_type = bout['interactions'][0]
    #                                 predominant_type = interaction_counts.most_common(1)[0][0]
    #                                 partner = [i for i in pair if i != track_id][0]
    #                                 partner_status = df[(df['track_id'] == partner) & (df['frame'] == start)]['hole_status'].values[0] if not df[(df['track_id'] == partner) & (df['frame'] == start)].empty else None

    #                                 bout_data = {
    #                                     'file': track_file,
    #                                     'track': track_id,
    #                                     'status_hole': label,
    #                                     'duration_analysed': duration,
    #                                     'interacted': True,
    #                                     'start_frame': start,
    #                                     'end_frame': end,
    #                                     'duration': duration_bout,
    #                                     'partner': partner,
    #                                     'partner_status': partner_status,
    #                                     'initial_type': initial_type,
    #                                     'predominant_type': predominant_type,
    #                                 }
    #                                 for t in unified_types:
    #                                     bout_data[t] = interaction_counts.get(t, 0)
    #                                 bouts.append(bout_data)

    #                     for pair, interactions in interacting_pairs.items():
    #                         if pair in active_bouts:
    #                             active_bouts[pair]['end_frame'] = frame
    #                             active_bouts[pair]['interactions'].extend(interactions)
    #                             active_bouts[pair]['gap_count'] = 0
    #                         else:
    #                             active_bouts[pair] = {
    #                                 'start_frame': frame,
    #                                 'end_frame': frame,
    #                                 'interactions': interactions.copy(),
    #                                 'gap_count': 0,
    #                             }

    #                 for pair, bout in active_bouts.items():
    #                     start, end = bout['start_frame'], bout['end_frame']
    #                     duration_bout = end - start + 1
    #                     interaction_counts = Counter(bout['interactions'])
    #                     initial_type = bout['interactions'][0]
    #                     predominant_type = interaction_counts.most_common(1)[0][0]
    #                     partner = [i for i in pair if i != track_id][0]
    #                     partner_status = df[(df['track_id'] == partner) & (df['frame'] == start)]['hole_status'].values[0] if not df[(df['track_id'] == partner) & (df['frame'] == start)].empty else None

    #                     bout_data = {
    #                         'file': track_file,
    #                         'track': track_id,
    #                         'status_hole': label,
    #                         'duration_analysed': duration,
    #                         'interacted': True,
    #                         'start_frame': start,
    #                         'end_frame': end,
    #                         'duration': duration_bout,
    #                         'partner': partner,
    #                         'partner_status': partner_status,
    #                         'initial_type': initial_type,
    #                         'predominant_type': predominant_type,
    #                     }
    #                     for t in unified_types:
    #                         bout_data[t] = interaction_counts.get(t, 0)
    #                     bouts.append(bout_data)

    #                 if not any((b['track'] == track_id and b['status_hole'] == label) for b in bouts):
    #                     bout_data = {
    #                         'file': track_file,
    #                         'track': track_id,
    #                         'status_hole': label,
    #                         'duration_analysed': duration,
    #                         'interacted': False,
    #                         'start_frame': frames.min(),
    #                         'end_frame': frames.max(),
    #                         'duration': None,
    #                         'partner': None,
    #                         'partner_status': None,
    #                         'initial_type': None,
    #                         'predominant_type': None,
    #                     }
    #                     for t in unified_types:
    #                         bout_data[t] = None
    #                     bouts.append(bout_data)

    #     bout_df = pd.DataFrame(bouts).sort_values(by=['file', 'track', 'status_hole', 'start_frame'])
    #     bout_df.to_csv(os.path.join(self.directory, 'pre_post_interactions.csv'), index=False)
    #     return bout_df

    def pre_post_hole_interactions(self, threshold=1.0):

        continue_threshold = 1.5  # once started, can CONTINUE while min_dist < this

        def unify_interaction_type(p1, p2):
            return "_".join(sorted([p1, p2]))

        body_parts = ["head", "body", "tail"]
        interaction_pairs = list(itertools.product(body_parts, body_parts))
        unified_types = sorted(
            set(unify_interaction_type(p1, p2) for p1, p2 in interaction_pairs)
        )

        bouts = []

        for track_file in self.track_files:
            df = self.track_data[track_file].copy()
            df.sort_values(by="frame", inplace=True)

            entry_map = df[df["within_hole"]].groupby("track_id")["frame"].min().to_dict()

            exit_map = (
                df[df["track_id"].isin(entry_map.keys())]
                .assign(prev=lambda d: d.groupby("track_id")["within_hole"].shift())
                .query("within_hole == False and prev == True")
                .groupby("track_id")["frame"]
                .min()
                .to_dict()
            )

            # analyse only outside-hole frames (your original behaviour)
            df = df[df["within_hole"] == False].copy()

            for track_id in entry_map:
                if track_id not in exit_map:
                    continue

                entry_frame = entry_map[track_id]
                exit_frame = exit_map[track_id]

                pre_mask = (df["track_id"] == track_id) & (df["frame"] < entry_frame)
                post_mask = (df["track_id"] == track_id) & (df["frame"] > exit_frame)

                duration_pre = pre_mask.sum()
                duration_post = post_mask.sum()
                duration = min(duration_pre, duration_post)

                if duration < 1:
                    continue

                pre_frames = df[pre_mask].nlargest(duration, "frame")["frame"]
                post_frames = df[post_mask].nsmallest(duration, "frame")["frame"]

                for label, frames in [("pre", pre_frames), ("post", post_frames)]:
                    subset = df[df["frame"].isin(frames)].copy()

                    active_bouts = {}  # pair_key -> bout dict
                    bout_counter = 0

                    for frame in sorted(subset["frame"].unique()):
                        frame_data = subset[subset["frame"] == frame]
                        track_ids = frame_data["track_id"].unique()

                        coords = {
                            part: {
                                row["track_id"]: np.array(
                                    [row[f"x_{part}"], row[f"y_{part}"]]
                                )
                                for _, row in frame_data.iterrows()
                            }
                            for part in body_parts
                        }

                        # START/true contacts (<1.0)
                        interacting_pairs = {}  # pair_key -> list of interaction types
                        # CONTINUE condition (<1.5)
                        close_pairs = {}  # pair_key -> closest_type

                        for id1, id2 in itertools.combinations(track_ids, 2):
                            if track_id not in (id1, id2):
                                continue

                            partner_id = id2 if id1 == track_id else id1
                            partner_row = frame_data[frame_data["track_id"] == partner_id]
                            if partner_row.empty or partner_row["within_hole"].values[0]:
                                continue

                            interactions = []
                            min_dist = float("inf")
                            closest_type = None

                            for part1, part2 in interaction_pairs:
                                coord1 = coords[part1].get(id1)
                                coord2 = coords[part2].get(id2)
                                if coord1 is None or coord2 is None:
                                    continue

                                dist = np.linalg.norm(coord1 - coord2)

                                if dist < min_dist:
                                    min_dist = dist
                                    closest_type = unify_interaction_type(part1, part2)

                                if dist < threshold:
                                    interactions.append(unify_interaction_type(part1, part2))

                            pair_key = tuple(sorted((id1, id2)))

                            # continuation condition: within 1.5mm
                            if closest_type is not None and min_dist < continue_threshold:
                                close_pairs[pair_key] = closest_type

                            # start/true-contact condition: any <1mm
                            if interactions:
                                interacting_pairs[pair_key] = interactions

                        current_close = set(close_pairs.keys())

                        # 1) END bouts that are no longer within 1.5mm
                        for pair in list(active_bouts.keys()):
                            if pair not in current_close:
                                bout = active_bouts.pop(pair)
                                interactions_all = bout["interactions"]
                                if not interactions_all:
                                    continue

                                interaction_counts = Counter(interactions_all)
                                start, end = bout["start_frame"], bout["end_frame"]
                                partner = [i for i in pair if i != track_id][0]

                                partner_status = (
                                    df[(df["track_id"] == partner) & (df["frame"] == start)][
                                        "hole_status"
                                    ].values[0]
                                    if not df[
                                        (df["track_id"] == partner) & (df["frame"] == start)
                                    ].empty
                                    else None
                                )

                                bout_data = {
                                    "file": track_file,
                                    "track": track_id,
                                    "status_hole": label,
                                    "duration_analysed": duration,
                                    "interacted": True,
                                    "start_frame": start,
                                    "end_frame": end,
                                    "duration": end - start + 1,
                                    "partner": partner,
                                    "partner_status": partner_status,
                                    "initial_type": interactions_all[0],
                                    "predominant_type": interaction_counts.most_common(1)[0][0],
                                }
                                for t in unified_types:
                                    bout_data[t] = interaction_counts.get(t, 0)
                                bouts.append(bout_data)

                        # 2) UPDATE existing bouts that are still within 1.5mm
                        for pair in list(active_bouts.keys()):
                            # pair must be in close_pairs
                            active_bouts[pair]["end_frame"] = frame

                            if pair in interacting_pairs:
                                active_bouts[pair]["interactions"].extend(
                                    interacting_pairs[pair]
                                )
                            else:
                                # filler closest type (1.0-1.5mm band)
                                active_bouts[pair]["interactions"].append(close_pairs[pair])

                        # 3) START new bouts ONLY if they hit <1mm this frame
                        for pair, interactions in interacting_pairs.items():
                            if pair in active_bouts:
                                continue
                            active_bouts[pair] = {
                                "bout_id": bout_counter,
                                "start_frame": frame,
                                "end_frame": frame,
                                "interactions": interactions.copy(),
                            }
                            bout_counter += 1

                    # Finalize remaining bouts at end of this pre/post window
                    for pair, bout in active_bouts.items():
                        interactions_all = bout["interactions"]
                        if not interactions_all:
                            continue

                        interaction_counts = Counter(interactions_all)
                        start, end = bout["start_frame"], bout["end_frame"]
                        partner = [i for i in pair if i != track_id][0]

                        partner_status = (
                            df[(df["track_id"] == partner) & (df["frame"] == start)][
                                "hole_status"
                            ].values[0]
                            if not df[(df["track_id"] == partner) & (df["frame"] == start)].empty
                            else None
                        )

                        bout_data = {
                            "file": track_file,
                            "track": track_id,
                            "status_hole": label,
                            "duration_analysed": duration,
                            "interacted": True,
                            "start_frame": start,
                            "end_frame": end,
                            "duration": end - start + 1,
                            "partner": partner,
                            "partner_status": partner_status,
                            "initial_type": interactions_all[0],
                            "predominant_type": interaction_counts.most_common(1)[0][0],
                        }
                        for t in unified_types:
                            bout_data[t] = interaction_counts.get(t, 0)
                        bouts.append(bout_data)

                    # Fallback row if no bouts for this (track_id, label)
                    if not any((b["track"] == track_id and b["status_hole"] == label) for b in bouts):
                        bout_data = {
                            "file": track_file,
                            "track": track_id,
                            "status_hole": label,
                            "duration_analysed": duration,
                            "interacted": False,
                            "start_frame": frames.min(),
                            "end_frame": frames.max(),
                            "duration": None,
                            "partner": None,
                            "partner_status": None,
                            "initial_type": None,
                            "predominant_type": None,
                        }
                        for t in unified_types:
                            bout_data[t] = None
                        bouts.append(bout_data)

        bout_df = pd.DataFrame(bouts).sort_values(by=["file", "track", "status_hole", "start_frame"])
        bout_df.to_csv(os.path.join(self.directory, "pre_post_interactions.csv"), index=False)
        return bout_df







        #                 ended = set(active_bouts) - set(interacting_pairs)

        #                 for pair in ended:
        #                     bout = active_bouts.pop(pair)
        #                     start, end = bout['start_frame'], bout['end_frame']
        #                     duration_bout = end - start + 1
        #                     interactions = bout['interactions']
        #                     interaction_counts = Counter(interactions)
        #                     initial_type = interactions[0]
        #                     predominant_type = interaction_counts.most_common(1)[0][0]

        #                     partner = [i for i in pair if i != track_id][0]
        #                     partner_status = df[
        #                         (df['track_id'] == partner) & (df['frame'] == start)
        #                     ]['hole_status'].values[0] if not df[(df['track_id'] == partner) & (df['frame'] == start)].empty else None

        #                     bout_data = {
        #                         'file': track_file,
        #                         'track': track_id,
        #                         'status_hole': label,
        #                         'duration_analysed': duration,
        #                         'interacted': True,
        #                         'start_frame': start,
        #                         'end_frame': end,
        #                         'duration': duration_bout,
        #                         'partner': partner,
        #                         'partner_status': partner_status,
        #                         'initial_type': initial_type,
        #                         'predominant_type': predominant_type,
        #                     }
        #                     for t in unified_types:
        #                         bout_data[t] = interaction_counts.get(t, 0)
        #                     bouts.append(bout_data)

        #                 for pair, interactions in interacting_pairs.items():
        #                     if pair in active_bouts:
        #                         active_bouts[pair]['end_frame'] = frame
        #                         active_bouts[pair]['interactions'].extend(interactions)
        #                     else:
        #                         active_bouts[pair] = {
        #                             'start_frame': frame,
        #                             'end_frame': frame,
        #                             'interactions': interactions.copy(),
        #                         }

        #             if not any((b['track'] == track_id and b['status_hole'] == label) for b in bouts):
                    
                
        #                 bout_data = {
        #                     'file': track_file,
        #                     'track': track_id,
        #                     'status_hole': label,
        #                     'duration_analysed': duration,
        #                     'interacted': False,
        #                     'start_frame': frames.min(),
        #                     'end_frame': frames.max(),
        #                     'duration': frames.max() - frames.min() + 1,
        #                     'partner': None,
        #                     'partner_status': None,
        #                     'initial_type': None,
        #                     'predominant_type': None,
        #                 }
        #                 for t in unified_types:
        #                     bout_data[t] = None
        #                 bouts.append(bout_data)

        # bout_df = pd.DataFrame(bouts).sort_values(by=['file', 'track', 'status_hole', 'start_frame'])
        # bout_df.to_csv(os.path.join(self.directory, 'pre_post_interactions.csv'), index=False)
        # return bout_df


















            

























# DIGGING IN ISOLATION/ HOW TO COMBINE THE MAN-MADE HOLE WITH THE NUMBER DIGGING OUTSIDE OF THIS
    # IDEK HOW TO DO THIS

# METHOD CASTING: 

# METHOD FOR TRACK OVERLAY IMAGES AND VIDEOS 










