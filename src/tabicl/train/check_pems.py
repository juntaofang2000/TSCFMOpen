
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from tabicl.prior.data_reader import DataReader

try:
    reader = DataReader(UEA_data_path="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UEAData/", UCR_data_path="/data0/fangjuntao2025/CauKer/CauKerOrign/CauKer-main/UCRdata/")
    X_train, y_train = reader.read_dataset("PEMS-SF", which_set="train")
    print(f"PEMS-SF Train shape: {X_train.shape}")
    X_test, y_test = reader.read_dataset("PEMS-SF", which_set="test")
    print(f"PEMS-SF Test shape: {X_test.shape}")
except Exception as e:
    print(f"Error: {e}")
