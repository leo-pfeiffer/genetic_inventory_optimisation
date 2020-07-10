## Learning from Nature: Using Genetic Algorithms for Inventory Optimisation

The code is split into three folders, model, pilotstudy, and simulations.<br />

1. model
    * GenericGA: Contains all scripts connected to the generic GA used in the thesis. FitnessFunctions.py contains 
    the functions to be optimised, GenericGenAlg.py contains the code for the algorithm, and RunSimpleGenAlg.py can
    be run to see the Generic GA do its magic (Algorithm 1).
    * GenAlg_thesis.py: Contains all classes and functions related to the main GA configurations of the thesis (Algorithm 3, 4, 5)
    * RandomSearch.py: Contains all classes and functions related to the RS of the thesis
    * SCsettings_thesis.py: Contains all SC settings and some functions related to them
    * SupplyChain_thesis: All classes and functions related to the SC model (Algorithm 2)
    
2. pilotstudy
    * results: Contains CSV files that were obtained as output of PilotStudy.py and PilotStudyRS.py
    * pilotstudy.ipynb: Jupyter Notebook to find optimal parameters for GA1 and create Figure 6.1
    * pilotstudy_SubparGAs.ipynb: Jupyter Notebook to find optimal parameters for GA2 and GA3
    * PilotStudy.py: Script used for the pilot study of the GA
    * PilotStudyRS.py: Script used for the pilot study of the RS
   
3. simulations:
    * GA1, GA2, GA3, RS: csv files created by Simulate.py for the main simulations
    * plots: Contains plots created by simulation_visual.ipynb corresponding to Figures 6.2 - 6.5
    * simulation_numeric.ipynb: Jupyter Notebook for numeric analysis of the simulation results. Creates Table 6.6
    * simulation_visual.ipynb: Jupyter Notebook for visual analysis of the simulation results. Saved in /plots. 
   