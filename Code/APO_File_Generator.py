import itertools
import time
import warnings
from itertools import product 
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import StandardScaler
from pmlb import fetch_data
import re
import multiprocessing
from collections.abc import Iterable
from pathlib import Path

tmp_chunks = "tmp_chunks"
tmp_results = "tmp_results"
length = 3

Path(tmp_chunks).mkdir(parents=True, exist_ok=True)
Path(tmp_results).mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")

operators = {"Pow": {"arity":2, "symmetric": False}, "Mul": {"arity":2, "symmetric": True},
             "Add": {"arity":2, "symmetric": True}, "Sub": {"arity":2, "symmetric": False},
             "Div": {"arity":2, "symmetric": False},}

def Pow(op1, op2):
    return np.power(np.abs(op1),op2)

def Mul(op1, op2):
    return op1 * op2

def Add(op1, op2):
    return op1 + op2

def Sub(op1, op2):
    return op1 - op2

def Div(op1, op2):
    return op1 / op2

class Node:
    def __init__(self, symbol="UNFILLED", parent=None):
        self.symbol = symbol
        self.parent = parent
        self.children = []
        self.on_variable_path = False

def kexp_to_tree(kexp, ordered_symbols = True):
    kexp = list(kexp)
    root = Node()
    queue = [root]  # FIFO queue
    seen = [root]  # Tracker
    for symbol in kexp:
        if not queue:
            break
        cur_Node = queue[0]
        queue = queue[1:]
        cur_Node.symbol = symbol
        if symbol in operators:
            no_of_children = operators[symbol]["arity"]
            cur_Node.children = [Node(parent=cur_Node) for i in range(no_of_children)]
            queue.extend(cur_Node.children)
            seen.extend(cur_Node.children)
        elif symbol in ("R",):
            pass
    if ordered_symbols:
        queue = [root]  # FIFO queue
        all_nodes = [root]
        while queue:
            cur_Node = queue[0]
            queue = queue[1:]
            queue.extend(cur_Node.children)
            all_nodes.extend(cur_Node.children)
        for node in all_nodes:
            if node.symbol not in ("R",) and operators[node.symbol]["symmetric"]:
                node.children = sorted(node.children, key=lambda x:x.symbol)
    return root

def tree_to_exp(node):
    symbol = node.symbol
    if node.symbol in operators:
        return (
            symbol
            + "("
            + "".join([tree_to_exp(child) + "," for child in node.children])[:-1]
            + ")"
        )
    else:
        return symbol

def get_exp_set(k_exp_front_length = 3):
    exhuastive_symbol_set = [i for i in operators]+["R"]
    k_exp_list = [list(i)+["R"]*(k_exp_front_length+1) for i in product(exhuastive_symbol_set,repeat=k_exp_front_length)]
    exp_list = [tree_to_exp(kexp_to_tree(list(i)+["R"]*(k_exp_front_length+1))) for i in product(exhuastive_symbol_set,repeat=k_exp_front_length)]
    exp_set = set(exp_list)
    return exp_set

def cost(x, xdata, ydata, lambda_string): 
    y_pred = eval(lambda_string)(x, xdata)
    return np.mean(((y_pred - ydata))**2) 

def cust_pred(x, xdata, ydata, lambda_string): 
    y_pred = eval(lambda_string)(x, xdata)
    if isinstance(y_pred, Iterable):
        return y_pred
    else:
        return [y_pred]*len(ydata)

def save_strings_to_file(strings, filename):
    filename = tmp_chunks + '/' + filename
    with open(filename, 'w') as file:
        for string in strings:
            file.write(string + '\n')

def load_strings_from_file(filename):
    filename = tmp_chunks + '/' + filename
    with open(filename, 'r') as file:
        for line in file:
            yield line.strip()
    
def process_eq(chunk_idx, dataset_name, train_X, train_y, random_state, method="BFGS"):
    results = []
    total_time_chunk = 0
    eq_list_chunk = load_strings_from_file(f'strings_data_chunk_{chunk_idx}.txt')
    for test_eq in eq_list_chunk:
        eq_len = test_eq.count("[") + test_eq.count("(")
        ERC_count = test_eq.count("x[")
        lambda_string = "lambda x,xdata:" + test_eq
        np.random.seed(random_state)
        if ERC_count:
            try:
                start_time = time.time()
                res = minimize(cost,
                               x0=(2*np.random.rand(ERC_count)-1),
                               args=(train_X.T, train_y, lambda_string),
                               method=method)
                total_time_chunk+=time.time() - start_time
                optimized_cost = cost(res.x, train_X.T, train_y, lambda_string)
                y_pred = cust_pred(res.x, train_X.T, train_y, lambda_string)
                r2 = r2_score(train_y,y_pred)
                results.append((eq_len,test_eq, lambda_string, res.x, res.nit, optimized_cost,r2))
            except (RuntimeError,ValueError):
                results.append((eq_len,test_eq, lambda_string, None, None, None, None))
        else:
            try:
                optimized_cost = cost(None, train_X.T, train_y, lambda_string)
                y_pred = cust_pred(None, train_X.T, train_y, lambda_string)
                r2 = r2_score(train_y,y_pred)
                results.append((eq_len,test_eq, lambda_string, None, None, optimized_cost,r2))
            except (RuntimeError,ValueError):
                results.append((eq_len,test_eq, lambda_string, None, None, None, None))
    return results, total_time_chunk

def test_SR(dataset_name="1027_ESL",method="BFGS",length=3,random_state=11284):
    exp_set = get_exp_set(length)
    total_time = 0
    X, y = fetch_data(dataset_name,return_X_y=True)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y_mean, y_std = np.mean(y), np.std(y)
    y = (y-y_mean)/y_std
    train_X, train_y = X, y
    np.random.seed(random_state)
    shuffle_idx = np.random.permutation(len(train_y))
    train_X = train_X[shuffle_idx]
    train_y = train_y[shuffle_idx]
    
    master_list=[]
    eq_list = []
    
    num_of_feature = train_X.shape[1]
    
    mse_tuple = tuple()

    exp_set = list(exp_set)
    np.random.seed(random_state)
    np.random.shuffle(exp_set)

    for test_eq in exp_set:
        test_eq_orig = test_eq
        R_count = test_eq.count("R")
        
        for combi_var in itertools.product(range(num_of_feature+1), repeat=R_count):
            test_eq=test_eq_orig
            for i in combi_var:
                if i==num_of_feature:
                    test_eq = test_eq.replace("R", "erc", 1)
                else:
                    test_eq = test_eq.replace("R", f"xdata[{i}]", 1)
            match = re.search(r"\w{3}\(erc,erc\)", test_eq)
            while match:
                test_eq = test_eq.replace(match.group(),"erc")
                match = re.search(r"\w{3}\(erc,erc\)", test_eq)
            ERC_count = test_eq.count("erc")
            for i in range(ERC_count):
                test_eq = test_eq.replace("erc", f"x[{i}]", 1)
            eq_list.append(test_eq)

    eq_list = list(set(eq_list))

    num_processes = multiprocessing.cpu_count()-1
    eq_list_len = len (eq_list)
    chunk_size = eq_list_len // num_processes
    for idx, i in enumerate(range(0, eq_list_len, chunk_size)):
        save_strings_to_file(eq_list[i:i + chunk_size], f'strings_data_chunk_{idx}.txt')
    del eq_list

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_eq, [(chunk_idx, dataset_name, train_X, train_y,random_state,method) for chunk_idx in range(num_processes)])

    for result_chunk, total_time_chunk in results:
        master_list.extend(result_chunk)
        total_time+=total_time_chunk
        
    print(total_time)
    df = pd.DataFrame(master_list)
    df.columns = ["EquationLength","EquationStructure","EquationLambda","EquationParameters","NumericalIterations","MSE","R2"]
    df = df.sort_values(by=["R2","EquationLength","EquationStructure"], ascending=[False,True,False], na_position='last')
    df["MSE"] *= y_std**2
    df.to_csv(f"{tmp_results}/{dataset_name}_{method}_{length}_{random_state}.csv", index=False)
    return 

total_time = 0

for random_state in [11284, 11964, 15795, 21575, 22118, 23654, 29802,  5390,  6265, 860]:
    for dataset_name in ['1027_ESL', '649_fri_c0_500_5', '523_analcatdata_neavote', '712_chscase_geyser1', '557_analcatdata_apnea1', '579_fri_c0_250_5', '617_fri_c3_500_5', '631_fri_c1_500_5', '228_elusage', '547_no2', '561_cpu', '659_sleuth_ex1714', '706_sleuth_case1202', '210_cloud', '192_vineyard', '611_fri_c3_100_5', '522_pm10', '485_analcatdata_vehicle', '678_visualizing_environmental', '519_vinnie', '687_sleuth_ex1605', '613_fri_c3_250_5', '594_fri_c2_100_5', '656_fri_c1_100_5', '596_fri_c2_250_5', '597_fri_c2_500_5', '601_fri_c1_250_5', '690_visualizing_galaxy', '1096_FacultySalaries', '665_sleuth_case2002', '624_fri_c0_100_5', '663_rabe_266', '230_machine_cpu', '556_analcatdata_apnea2']:
            for method in ["BFGS", "CG","L-BFGS-B","Nelder-Mead","Powell", "SLSQP","TNC","trust-constr"]:
                print(dataset_name,method,length,random_state)
                start_time = time.time()
                DistilSR_solution = test_SR(dataset_name=dataset_name,method=method,length=length,random_state=random_state)
                print(f"{time.time() - start_time=}")