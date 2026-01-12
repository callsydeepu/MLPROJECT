import os
import pickle


def save_object(file_path, obj):
    """
    Save any Python object to disk using pickle.
    Automatically creates parent directories if needed.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)


def load_object(file_path):
    """
    Load a pickled Python object from disk.
    """
    with open(file_path, "rb") as file:
        return pickle.load(file)
