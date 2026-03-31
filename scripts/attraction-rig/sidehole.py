
import tkinter as tk
from hole_gui import HoleGui  # Import the HoleGui class from the hole_gui module
from behavioural_analysis import HoleAnalysis 
import os 
import seaborn as sns
import matplotlib.pyplot as plt


def run_gui():
    root = tk.Tk()  # Create the main root window
    app = HoleGui(root)  # Create an instance of the SideHoleGui class with the root window
    root.mainloop()  # Start the Tkinter main loop
    print("GUI closed.")


def perform_analysis(directory):
    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        return

    print("Starting analysis...")
    # analysis = HoleAnalysis()
    analysis = HoleAnalysis(directory)

    analysis.merged_dataframes()

    # analysis.post_processing()
    # analysis.quality_control()
    # analysis.shorten(frame=600)


    ####### --- LARVAL BEHAVIOUR --- ######

    ## REMOVE DIGGING FROM ANALYSIS 

    analysis.digging_mask()


    ### PSEUDO POPULATION MODEL

    # analysis.pseudo_population_model()
    # analysis.pseudo_population_model(number_of_iterations=1, number_of_animals=2)

    ## BEHAVIOURAL DYNAMICS 

    # analysis.trajectory()
    # analysis.time_average_msd(list(range(1, 101, 1)))
    # analysis.speed()
    # analysis.ensemble_msd()
    # analysis.acceleration()
    # analysis.euclidean_distance()
    # analysis.euclidean_distance_variance(200, 600) # currently plotting the plataeu but arbitary time start for plateu 
    # analysis.distance_from_centre()
    # analysis.trajectory()
    # analysis.movement_direction()

    ## PROXIMITY AND INTERACTION DYNAMICS 

    # analysis.potential_interactions(threshold=5.0)
    # analysis.individual_approach_responses(threshold=2)
    # analysis.individual_approach_responses(threshold=3)
    # analysis.individual_approach_responses(threshold=4)
    # analysis.individual_approach_responses(threshold=5)
    # analysis.individual_approach_responses(threshold=6)
    # analysis.individual_approach_responses(threshold=7)
    # analysis.individual_approach_responses(threshold=8)
    # analysis.individual_approach_responses(threshold=9)
    # analysis.individual_approach_responses(threshold=10)

    # analysis.individual_approach_responses_consistent_approach_angle(threshold=5)
    # analysis.individual_approach_responses_consistent_approach_angle(threshold=10)



    # analysis.nearest_neighbour()
    # analysis.interactions()
    # analysis.interaction_types()
    # analysis.interaction_types_closest()
    # analysis.interaction_type_bout()
    # analysis.interaction_bout_dynamics() 
    # analysis.contacts(proximity_threshold=5)


    ## GH AND SI SAME PLATE ANALYSIS
    # analysis.nearest_neighbour() # has track id, speed, acceleration, angles etc
    # analysis.GH_SI_interaction_types_closest(threshold=1) # has track id, speed, acceleration, angles etc
    
    ## HEAD-HEAD ANALYSIS
    # analysis.head_head_interaction_type()
    # analysis.head_head_interaction_type_over_time()
    # analysis.interaction_types()
    # analysis.euclidean_distance()
    # analysis.trajectory()
    # analysis.speed()
    # analysis.acceleration()
    # analysis.nearest_neighbour()
    # analysis.contacts()
    # analysis.movement_direction()
    # analysis.head_head_approach_angle()
    # analysis.head_head_first_contact_kinematics()
    # analysis.head_head_contacts_kinematics_over_time()
    

        


    ####### --- DIGGING IN ABSENCE OF HOLES --- ######
    analysis.total_digging(cleaned=True) #HAVE TO MODIFY???
    analysis.digging_behaviour()


    ####### --- HOLES --- ######
    ## REQUIRES COMPUTE HOLE
    # analysis.compute_hole() 

    # analysis.hole_counter()
    # analysis.hole_frame_counts()
    # analysis.returns()
    # analysis.hole_departures()
    # analysis.time_to_enter()
    # analysis.hole_entry_probability()
    # analysis.hole_entry_departure_latency()
    # analysis.speed_hole() # internal logic for hole / digging

    ## REQUIRES HOLE STATUS 
    # analysis.hole_status() 
    # analysis.interactions_return() ### SURELY THESE NEED HOLE MASK???
    # analysis.pre_post_hole_interactions()
    

    ## REQUIRES HOLE MASK 1) REMOVES LARVAE IN HOLE 2) REMOVES DIGGING OUTSIDE HOLE
    # analysis.hole_mask() 
    # analysis.hole_orientation()
    # analysis.distance_from_hole()
    # analysis.hole_euclidean_distance()

    ## REQUIRES HOLE STATUS AND MASK
    # analysis.hole_status_interactions() 



    print("Analysis Completed")


if __name__ == "__main__":

    run_gui_option = input("Do you want to run the GUI for hole drawing? (y/n): ").strip().lower()
    if run_gui_option == 'y':
        run_gui()    


    # perform_analysis("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n1/group-housed")
    # perform_analysis("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n1/socially-isolated")
    # perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/group-housed")
    # perform_analysis("/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/n2/socially-isolated")
    # perform_analysis("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/group-housed")
    # perform_analysis("/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/socially-isolated")

    # perform_analysis('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/n10/grouped+isolated')


    # perform_analysis('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/GROUP-HOUSED')
    # perform_analysis('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/SOCIAL-ISOLATION')

    # perform_analysis('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-digging-2/n2')

    # perform_analysis('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/social-isolation/holes/N10-1-HOLE/test-delete')

    # perform_analysis('/Volumes/lab-windingm/home/users/cochral/AttractionRig/analysis/testing-methods/test-digging-mask/diff-video')


    perform_analysis('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/group-housed/starved-starved')
    perform_analysis('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/group-housed/fed-starved')
    perform_analysis('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/group-housed/fed-fed')
    perform_analysis('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/socially-isolated/starved-starved')
    perform_analysis('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/socially-isolated/fed-starved')
    perform_analysis('/Volumes/lab-windingm/home/users/cochral/LRS/AttractionRig/analysis/social-isolation/head-head/2/socially-isolated/fed-fed')





    








    





