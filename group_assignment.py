#################################################
# This is a file to define the function group_assign()
# This will assign participants into 2 groups automatically and evenly.
##################################################


from random import randint
import pandas as pd
import os

def group_assign(filename):
    '''
    Assign participants into 2 groups during the first session.
    filename: name of file recording each participant' information
    group: 0 - short blocks group; 1 - long blocks group
    id: Participants' initials+age+gender. eg. SS25F

    '''
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # get current path of this script
    RESULT_PATH = os.path.join(SCRIPT_DIR, "results")
    columns = ['id', 'group', 'gender', 'age', 'session_comp']
    
    if not os.path.exists(filename):
        # file not exists，1st session of the 1st participant
        curr_group = 0
        row_number = 1

        # ensure path exists
        if not os.path.exists(RESULT_PATH):
            os.makedirs(RESULT_PATH)
        
        # create a form
        try:
            info = pd.DataFrame(columns=columns) # header
            info.to_csv(filename, index=False)
        except Exception as e:
            print(f'Debug: Problems saving subject info file. Error: {e}')
            
        return curr_group, row_number # int, int
    
    # if file exists
    try:
        df = pd.read_csv(filename)
        
        # if only header, 1st session of the 1st participant
        if df.empty:
            return 0, 1
            
        # count number of participants in each group 
        group_counts = df['group'].value_counts()
        count0 = group_counts.get(0, 0)
        count1 = group_counts.get(1, 0)

        # assign
        if count0 < count1:
            curr_group = 0
        elif count1 < count0:
            curr_group = 1
        else:
            curr_group = randint(0, 1)

        # Get the row number for the new participant
        row_number = len(df) + 1
            
    except Exception as e:
        print(f"Debug: Error during group assignment: {e}")
        curr_group = randint(0, 1)
        row_number = len(df) + 1 if 'df' in locals() and df is not None else 0

    return curr_group, row_number # int, int
    