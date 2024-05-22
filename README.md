# MFM_WS2223

## Project Structure
### top-level directory
    .
    ├── /ClusteredResult                        # Directory to store all the clustered results with specified R and K
    ├── /Data                                   # Directory to store the datasets, including nodes, arcs, map related data
    ├── /Solution                               # Directory to store all the clustered-MILP result
    ├── Cluster_MILP.ipynb                      # Notebook to run the cluster method and the MILP on the clustered result
    ├── Cluster_MILP.py                         # Script that stores all the necessary methods used in preprocess clusters and run MILP
    ├── Model_Run.py                            # Script to run the original MILP model
    ├── Models.py                               # Script that stores the MILP model
    ├── Print_Routes.ipynb                      # Notebook to print the final clustered-MILP result
    ├── Run_Algo_revised.ipynb                  # Notebook to run and store the clustered-MILP on all datasets
    ├── cluster.py                              # Script that stores all the clustering methods
    ├── distance_generator.ipynb                # Notebook to create arcs dataset from the node datasets
    └── README.md

### other directory
#### ClusteredResult
    .
    ├── ...
    ├── /ClusteredResult                                    # Directory to store all the clustered results with specified R and K
    │   ├── /old file                                       # Directory to store old clustered results (Paris)
    │   ├── LA_r9_k5_kmeans_cluster_result.json        
    │   ├── LA_r9_k5_kmeans_cluster_result.pickle       
    │   ├── Paris_r2_k5_kmedoids_cluster_result.json         
    │   ├── Paris_r2_k5_kmedoids_cluster_result.pickle       
    │   ├── Shanghai_r8_k(1, 5)_SOM_cluster_result.pickle 
    │   └── Shanghai_r8_k(1, 5)_SOM_result_dict.json
    └── ...

#### Data
    .
    ├── ...
    ├── /Data                                               # Directory to store the datasets, including nodes, arcs, map related data
    │   ├── Data_Parser.py                                  # Script to parse the .nodes and .routes file 
    │   ├── generate_new_paris_data.py                      # Script to parse the Paris dataset
    │   ├── parse_amazon_data.py                            # Script to parse the Amazon dataset
    │   ├── ... .nodes                                      # .nodes file stores all the customer nodes and the demand for each node
    │   ├── ... .routes                                     # .routes file stores all the routes, distance, and path connecting two nodes
    │   ├── ... .osm                                        # .osm file is a file storing all the path-relevant data from open street map 
    │   └── ...                                             # etc.
    └── ...

#### Solution
    .
    ├── ...
    ├── /Solution                                           # Directory to store all the clustered-MILP result
    │   ├── /Map                                            # Directory to store the Map visualization of the clustered-MILP result in html format
    │   ├── Map_Printer.py                                  # Script to print Maps from the clustered-MILP result 
    │   ├── Result_Comparision.ipynby                       # Notebook to compare the clustered-MILP result
    │   ├── ... .csv                                        # .csv and .xlsx files store clustered-MILP result in text format
    │   └── ...                                             # etc.
    └── ...

