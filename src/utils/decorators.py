from pathlib import Path

def make_output_path_obj(func):
    def inner(*args, **kwargs):
         

         
        # getting the returned value
        returned_value = Path(func(*args, **kwargs))

         
        # returning the value to the original frame
        return returned_value
         
    return inner