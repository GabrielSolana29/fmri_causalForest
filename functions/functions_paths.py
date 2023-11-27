


def path_functions(linux:bool=False):
    import os
    directory = os.getcwd()
    
    if linux == True:          
        directory_functions = str(directory +"/functions/")
        path = directory_functions
    else:
        path = str(directory +"\\functions\\")              
    
    return path

def path_CSV(linux:bool=False):
    import os
    directory = os.getcwd()        
    
    if linux == True:     
        directory_functions = str(directory +"/CSV/")
        path = directory_functions        
    else:
        path = str(directory +"\\CSV\\")              
    
    return path


def path_saved_models_params(linux:bool=False):
    import os
    directory = os.getcwd()
    if linux == True:        
        directory_functions = str(directory +"/saved_model_params/")
        path = directory_functions        
    else:
        path = str(directory +"\\saved_model_params\\")              
    return path
    

def path_figures(linux:bool=False):
    import os
    directory = os.getcwd()
    if linux == True:        
        directory_functions = str(directory +"/Figures/")
        path = directory_functions        
    else:
        path = str(directory +"\\Figures\\")                 
    return path
    