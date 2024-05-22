import random
from Data.Data_Parser import fetch_all_dataset
from Models import run_F4
from Solution.Map_Printer import osm_print_tours
import json


if __name__ == "__main__":
    network, vertices, arcs, V_d, V_c, F, alpha, K, Q, demands = fetch_all_dataset()
    objective_value, solution, runtime, mip_gap = run_F4(demands, vertices, arcs, V_d, V_c, F, alpha, K, Q,
                                                         runtime_limit=10800, print_out=True)
    print(objective_value)
    print(runtime)
    print(mip_gap)
    file_name = "F4_All"
    osm_print_tours(network, solution, file_name)
    result_dict = {
        "obj_value": objective_value,
        "runtime": runtime,
        "mip_gap": mip_gap,
        "solution": solution
    }
    # json.dump(result_dict, open(".\\Solution\\paris_result.json", 'w'))
    # result_dict = json.load(open(".\\Solution\\paris_result.json"))
