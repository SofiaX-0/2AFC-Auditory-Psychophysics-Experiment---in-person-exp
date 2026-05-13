'''
2AFC Auditory Psychophysics Experiment
==================================================================================
Sound stimulus template - 300ms,10ms fade in and out @ 400Hz, sampleRate = 44100Hz, 
generated with Audacity

Sound amplitude: sampled from 2 distributions based on: 

Menichini, E., Pajot-Moric, Q., Low, R., Pedrosa, V., Pourdehghan, A., 
Vincent, P., Zhou, L., Teachen, L., & Akrami, A. (2025). 
Different learning algorithms achieve shared optimal outcomes in humans, rats, 
and mice. bioRxiv, 2023.01.30.526119. https://doi.org/10.1101/2023.01.30.526119


Interstimulus intervals: 
        350ms Fixation;
        (300ms stimulus);
        4000ms waiting for response;
        350ms Feedback
    -------------------------------
    Total Max.: 5000 ms / trial

Number of sessions: 7
Number of trials: (30 training) + 600 trials/session
Number of subject groups: 2
-> group 1: block size: [50-60]
-> group 2: block size: [130-210]

===============================================================================    
'''
###############################################################################
# This is the main file for the experiment.
# Params to change: quit -> line 436-437, line 262, NOT ALLOWED IN REAL EXPERIMENT
# RESULT_PATH (line 76 & 568); Windowsize: line 108; - REMEMBER TO check the 'results' folder first.
###############################################################################

import pandas as pd
import numpy as np
from psychopy import core, event, visual, logging, prefs, gui, sound
import os
from datetime import datetime
from random import randint
from stimulus_generator import sample_in_block
from group_assignment import group_assign
from block_size import block_size

logging.console.setLevel(logging.CRITICAL)
prefs.hardware['audioLib'] = ['sounddevice', 'ptb', 'pyo']
prefs.hardware['audioLatencyMode'] = 0


# clear command prompt 
def clear_output():
    os.system('cls' if os.name == 'nt' else 'clear') # cls on Windows; others: clear 
    print("Debug: clear screen")

def experiment_update():
    '''
    1) control the user interface;
    2) record subject response;
    3) update file information
    '''
    ## basic settings
    TOTAL_SESSION = 7
    response_limit = 4
    fixation_duration = 0.35
    feedback = 0.35
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # get current path of this script
    RESULT_PATH = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(RESULT_PATH, exist_ok=True) # create results folder if not exists
    FILENAME_INFO = os.path.join(RESULT_PATH, "Subject_info.csv") 
  
    if os.path.exists(FILENAME_INFO):
        df_master = pd.read_csv(FILENAME_INFO)
    else:
        df_master = pd.DataFrame(columns=['id', 'group', 'age', 'gender','session_comp'])
        df_master.to_csv(FILENAME_INFO, index=False)

    ## generate base stimuli
    def generate_white_noise(duration=0.3, sample_rate=44100, amplitude=1.0):
        '''
        Generate white noise (no fade in / fade out):
        Parameters:
        duration : float (default = 0.3s)
        sample_rate : int (default = 44100Hz)
        amplitude (0-1) : float (default = 1.0)

        Returns:
        psychopy.sound.Sound
        
        '''
        n_samples = int(duration * sample_rate)
        white_noise = np.random.normal(0, 1, n_samples)
        max_val = np.max(np.abs(white_noise))
        if max_val > 0:
            white_noise = white_noise / max_val
        white_noise = white_noise * amplitude
        white_noise = white_noise.astype(np.float32)
        sound_obj = sound.Sound(value=white_noise, 
                           stereo=True, 
                           sampleRate=sample_rate)
        return sound_obj
    
    
    ## feedback images
    img_correct_path = os.path.join(SCRIPT_DIR, "images","correct.png")
    img_wrong_path = os.path.join(SCRIPT_DIR, "images","wrong.png")


    Instruction = (
    f"Welcome to this experiment!\n\nPlease DO NOT CHANGE the device volume during the experiment.\n\n"
    f"In each trial, you will hear a sound. Your task is to classify the sound "
    f"into one of the two categories: A or B.\n\n"
    f"Press [←] LEFT arrow key for Category A.\n"
    f"Press [→] RIGHT arrow key for Category B.\n\n"
    f"You have {response_limit} seconds to respond. "
    f"If no response is made, it will be marked as incorrect.\n\n"
    f"Note: the reward is only granted upon completion of all {TOTAL_SESSION} sessions.\n\n"
    f"Press SPACE to start."
    )
    
    choice_A = "Category A: Press [←]"
    choice_B = "Category B: Press [→]"
    response_text = 'Please classify the sound by pressing an arrow key.'
    
    ### initialization
    curr_sess = 0
    trial_resp = 0 # number of trials with a response in the whole session
    trial_resp_blk = 0 # number of trials with a response in current block
    curr_trial_sess = 0 # current trial no. in the whole session
    corr_counter_block = 0
    corr_counter_session = 0
    
    ## ===================================================================
    ## INFO PAGE ---------------------------------------------------------
    ## => need to read & store information of subjects:
    ## === initials, age, gender (store OR match with IDs in Subject_info.csv);
    ## === check which session is this one (do not store for now), and whether it's correct

    def get_verified_info():
        while True:
        # 1. Information Input Stage
            while True:
                dlg = gui.Dlg(title="Demographics")
            
                if hasattr(dlg, 'labelStack'):
                    dlg.labelStack = []

                dlg.addText("Initials (e.g., sa) *")
                dlg.addField(key='initials', label=' ', initial='') 
            
                dlg.addText("Age (e.g., 22) *")
                dlg.addField(key='age', label=' ', initial='')
            
                dlg.addText("Gender *")
                dlg.addField(key='gender', label=' ', choices=['Male', 'Female', 'Other'])
            
                dlg.addText("Current Session (1, 2, 3...) *")
                session_choices = [str(i) for i in range(1, TOTAL_SESSION + 1)]
                dlg.addField(key='session', label=' ', choices=session_choices)

                if hasattr(dlg, 'cancelButton'):
                    dlg.cancelButton = None 

                dlg.show() 
                data = dlg.data 
            
                if data is None or not str(data[0]).strip() or not str(data[1]).strip():
                    err = gui.Dlg(title="Input Error")
                    err.show()
                    continue 

                try:
                    temp_initials = str(data[0]).upper().strip()
                    temp_age = int(data[1]) 
                    temp_gender = data[2]
                    temp_sess = int(data[3])
                    base_id = f"{temp_initials}{temp_age}{temp_gender[0]}".upper()
                    break 
                except (ValueError, TypeError, IndexError):
                    err = gui.Dlg(title="Input Error")
                    err.addText("Age must be a valid number!")
                    err.show()
                    continue

            # 2. Double Check Stage
            conf_dlg = gui.Dlg(title="Double Check")
            if hasattr(conf_dlg, 'labelStack'):
                conf_dlg.labelStack = []
            
            conf_dlg.addText("Please check the information below:")
            conf_dlg.addText(f"*Initials: {temp_initials}")
            conf_dlg.addText(f"*Age: {temp_age}")
            conf_dlg.addText(f"*Gender: {temp_gender}")
            conf_dlg.addText(f"*Current Session: {temp_sess}")
            conf_dlg.addText("\nClick OK to confirm, Cancel to re-type.")
        
            conf_dlg.show()
            if not conf_dlg.OK:
                continue 

            # 3. Logic & Session Consistency Check
            temp_id = base_id
            error_msg_text = ""

            if os.path.exists(FILENAME_INFO):
                # --- Check if the BASE ID already exists ---
                # We look for the exact base_id or any suffixed versions
                existing_records = df_master[df_master['id'].str.startswith(base_id)]
            
                if not existing_records.empty:
                    # If the user input Session 1, but this person already exists
                    if temp_sess == 1:
                        # Check if the exact base_id has already done session 1
                        # We need to ask: Is this a REALLY new person with same initials, or a mistake?
                        error_msg_text = f"ID {base_id} already exists. If you are this person, Session cannot be 1. If you are a NEW person, please alert the experimenter to add a suffix (e.g., 1_) to your initials."
                
                    else:
                        # If Session > 1, we must find which specific ID (base or suffixed) 
                        # is ready for this session (i.e., session_comp == temp_sess - 1)
                        found_correct_id = False
                        for _, row in existing_records.iterrows():
                            if int(row['session_comp']) == (temp_sess - 1):
                                temp_id = row['id']
                                found_correct_id = True
                                break
                    
                        if not found_correct_id:
                            # Find the max session completed among all matching IDs to give a helpful hint
                            max_comp = existing_records['session_comp'].max()
                            error_msg_text = f"Session mismatch! For ID {base_id}, the furthest completed session is {max_comp}. You cannot start Session {temp_sess}."
            
                else:
                    # Case: ID does not exist in master file at all
                    if temp_sess != 1:
                        error_msg_text = f"New ID detected ({base_id}), but Session is not 1! Please check your input."
        
            else:
                # Case: No master file exists yet
                if temp_sess != 1:
                    error_msg_text = "Master file not found. First ever participant must be Session 1!"

            # 4. Final Verification
            if error_msg_text == "":
                # Success: temp_id is now either the original or the correct suffixed one
                return temp_id, temp_sess, temp_age, temp_gender
            else:
                # Show the specific error message
                logic_err = gui.Dlg(title="Registration Error")
                logic_err.addText(error_msg_text)
                logic_err.show()

    # --- Execution ---
    subj_id, curr_sess, subj_age, subj_gender = get_verified_info()

    df_master = pd.read_csv(FILENAME_INFO)
    match_row = df_master[df_master['id'] == subj_id]
    if not match_row.empty:
        part_group = int(match_row.iloc[0]['group'])
        part_no = int(match_row.index[0]) + 1
    else:
        part_group, part_no = group_assign(FILENAME_INFO) # part_no: the number of the participant (1,2,...)
        new_record = {
            'id': subj_id, 
            'group': part_group, 
            'age': subj_age,
            'gender': subj_gender,
            'session_comp': 0
        }
        df_master = pd.concat([df_master, pd.DataFrame([new_record])], ignore_index=True)
        df_master.to_csv(FILENAME_INFO, index=False)

    print(f'DEBUG: Subject {subj_id} is in Group {part_group}.')

    ## ===================================================================
    ## BEGIN & INSTRUCTION PAGE ------------------------------------------
    win0 = visual.Window(size=[1920, 1080], screen = 0, monitor='testMonitor',
                         color= (1,1,1), fullscr=True, # white
                         winType='pyglet',
                         allowGUI=False,
                         waitBlanking=True) 
    win0.mouseVisible = False 
    text_stim = visual.TextStim(win0, font='Arial', color=(-1, -1, -1), units='pix',
                                height=40, wrapWidth=1400, alignText='left', anchorHoriz='center', anchorVert='center')
    text_stim.setText(Instruction)
    text_stim.wrapWidth = 1600
  
    event.clearEvents()
    text_stim.draw()
    win0.flip()

    continue_loop = True
    while continue_loop:
        text_stim.draw()
        win0.flip()
    
        keys = event.getKeys(keyList=['space', 'q'])
        if 'space' in keys:
            continue_loop = False
        ## quit option for test
        if 'q' in keys:
            win0.close()
            core.quit()
    
    win0.color = (1, 1, 1)
    win0.flip()
    event.clearEvents()
    ## =========================================================================
    ## SESSION CONDITIONS -------------------------------------------------------
    ## 1) feedback imgs & Block size arrangement; => curr_block_arr, curr_total_block
    ## 2) Whether to include training trials. [dep. on: current session]
    ## 3) initialize finish and last_trial flags.
    ## 4) side rule: 0 = A (left) for lower amp ; 1 = A (left) for higher amp

    ### feedback images setting
    img_correct = visual.ImageStim(win0, image=img_correct_path, 
                                   size=(406, 512), units='pix')
    img_incorrect = visual.ImageStim(win0, image=img_wrong_path, 
                                     size=(339, 512), units='pix')

    # BLOCK ARRANGEMENT
    curr_block_arr, curr_total_block = block_size(part_group)
    print(f'DEBUG: block sizes: {curr_block_arr}; total block: {curr_total_block}')

    train = False
    if curr_sess == 1:
        curr_total_block+= 1 # add a training 'block'
        curr_block_arr.insert(0, 30)
        train = True
    
    finish = False
    last_trial = False

    # side rule
    if part_no % 2 == 0:
        side_rule = 0 # A for lower amp correct
    else:
        side_rule = 1 # A for higher amp correct

    ## =========================================================================
    ## MAIN LOOP

    for block_no in range(1, curr_total_block+1):
        event.clearEvents()
        trial_resp_blk = 0
        corr_counter_block = 0
        updated_exp_date = datetime.today().strftime('%Y%m%d')
        updated_block_no = block_no # current block
        if train == True:
            if block_no == 1:
                updated_dist = 0 # uniform
            if block_no > 1:
                updated_dist = randint(1, 2) # 1 = hard-A, 2 = hard-B
        else: # no training 'block'
            updated_dist = randint(1, 2)
        # generate samples for this block
        # sample_in_block(id, session, distribution_type, num_block, num_trials, physical_min, physical_max)
        # physical_min/max are measured in dba
        df_block_sample = sample_in_block(subj_id,curr_sess,
                                          updated_dist, updated_block_no,curr_block_arr[block_no-1])
        amplitudes = df_block_sample['Target_dba']
        logical_values = df_block_sample['Logical_Value']
        corr_sides = df_block_sample['Side']
        distances = df_block_sample['Distance_toB_in_dba']
        boundary = df_block_sample['Physical_Boundary'].iloc[0]


        for trial_no in range(1, curr_block_arr[block_no-1] + 1):
            event.clearEvents()
            ### update file info
            updated_exp_time = datetime.today().strftime('%H%M%S')
            updated_blk_trial = trial_no # current trial in block
            curr_trial_sess += 1
            
            ### Fixation
            fixation = visual.ShapeStim(win0, 
                    vertices=((0, -30), (0, 30), (0, 0), (-30, 0), (30, 0)),
                    lineWidth=5,
                    closeShape=False,
                    lineColor='black',
                    units='pix'
                    )
            fixation.draw()
            win0.flip()
            core.wait(fixation_duration)

            ### Listen
            text_stim.setText('Listen')
            text_stim.setHeight(72)
            text_stim.alignText = 'center'
            text_stim.draw()
            win0.flip()
            #### set stimulus amplitude
            updated_amp = amplitudes.iloc[trial_no-1]
            # temporary linear mapping from dBA to PsychoPy volume!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            playback_volume = ((updated_amp - 45)/(75 - 45))

            playback_volume = np.clip(playback_volume, 0.0, 1.0)

            updated_logical_value = logical_values.iloc[trial_no-1]
            distance = distances.iloc[trial_no-1]
            true_cat = corr_sides.iloc[trial_no-1]
            ### create a sound object
            base_noise = generate_white_noise(duration=0.3, 
                                        sample_rate=44100, 
                                        amplitude=1.0)
            
            probe = base_noise

            probe.setVolume(playback_volume)
            probe.play()
            core.wait(probe.getDuration())

            ### Respond
            # countdown bar
            bar_width = 1000
            bar_height = 20
            bar_y_pos = 350
            countdown_bg = visual.Rect(win0, width=bar_width, height=bar_height, 
                           pos=(0, bar_y_pos), fillColor='grey', 
                           lineColor='black', units='pix')
            countdown_bar = visual.Rect(win0, width=bar_width, height=bar_height, 
                            pos=(0, bar_y_pos), fillColor='blue', 
                            lineColor='blue', units='pix')
            
            text_stim_resp = visual.TextStim(win0, text=response_text, font='Arial', 
                                 pos=(0, 200), height=45, color=(-1, -1, -1), units='pix', wrapWidth=1600,
                                 alignText='center')
            text_stim_A = visual.TextStim(win0, text=choice_A, font='Arial', 
                              pos=(-400, -150), height=50, color=(-1, -1, -1), units='pix', wrapWidth=1600,
                              alignText='center')
            text_stim_B = visual.TextStim(win0, text=choice_B, font='Arial', 
                              pos=(400, -150), height=50, color=(-1, -1, -1), units='pix', wrapWidth=1600,
                              alignText='center')

            resp_clock = core.Clock()
            event.clearEvents()
            updated_sub_response = -1
            updated_feed = 0
            rt = -1
            while resp_clock.getTime() < response_limit:
                time_left = response_limit - resp_clock.getTime()
                ratio = max(0, time_left / response_limit)
                new_width = bar_width * ratio
                countdown_bar.width = new_width
                countdown_bar.pos = (-(bar_width - new_width) / 2, bar_y_pos)
                countdown_bg.draw()
                countdown_bar.draw()

                text_stim_resp.draw()
                text_stim_A.draw()
                text_stim_B.draw()
                win0.flip()

                keys = event.getKeys(keyList=['left', 'right', 'q'], timeStamped=resp_clock)
                if keys:
                    key_name, key_time = keys[0]
                    rt = key_time
                    trial_resp += 1
                    trial_resp_blk += 1

                    if key_name == 'left':
                        updated_sub_response = 0
                        if true_cat == 0: # lower amp
                            if side_rule == 0: # lower amp on the left
                                print('DEBUG: Subject Correct.')
                                updated_feed = 1
                                corr_counter_block += 1
                                corr_counter_session += 1
                            else: # lower amp on the right
                                print('DEBUG: Subject Incorrect.')
                                updated_feed = 0
                        elif true_cat == 1: # higher amp
                            if side_rule == 0:
                                print('DEBUG: Subject Incorrect.')
                                updated_feed = 0
                            else: # lower amp on the right
                                print('DEBUG: Subject Correct.')
                                updated_feed = 1
                                corr_counter_block += 1
                                corr_counter_session += 1
            
                        print('Debug: Trial Number: %d' %(trial_no))
                        print('Debug: A chosen.')

                    elif key_name == 'right':
                        updated_sub_response = 1
                        if true_cat == 0: # lower amp
                            if side_rule == 0: # lower amp on the left
                                print('DEBUG: Subject Incorrect.')
                                updated_feed = 0
                            else: # lower amp on the right
                                updated_feed = 1
                                corr_counter_block += 1
                                corr_counter_session += 1
                                print('DEBUG: Subject Correct.')
                        elif true_cat == 1: # higher amp
                            if side_rule == 0: # lower amp on the left
                                updated_feed = 1
                                corr_counter_block += 1
                                corr_counter_session += 1
                                print('DEBUG: Subject Correct.')
                            else: # lower amp on the right
                                print('DEBUG: Subject Incorrect.')
                                updated_feed = 0

                        print('DEBUG: Trial Number: %d' %(trial_no))
                        print('DEBUG: B chosen.')
                    elif key_name == 'q':
                        win0.close()
                        core.quit()
                    break
            
            if rt == -1:
                updated_feed = 0
                print(f"DEBUG: Trial {trial_no} - Timed Out! Auto-switching to feedback.")

            win0.flip()
            event.clearEvents()


            updated_output = pd.DataFrame({'ID':[subj_id],
                             'GROUP': [part_group],
                             'SIDE_RULE': [side_rule],
                             'SESSION': [curr_sess],
                             'DATE':[updated_exp_date],
                             'TIME':[updated_exp_time],
                             'TRIAL_BLK':[updated_blk_trial], 
                             'BLK_NO': [updated_block_no], 
                             'DISTRIBUTION': [updated_dist],
                             'AMP':[updated_amp],
                             'LOGICAL_AMP':[updated_logical_value],
                             'PHY_BOUND': [boundary],
                             'DISTANCE':[distance],
                             'TRUE_CAT': [true_cat],
                             'SUB RESPONSE':[updated_sub_response],
                             'FEEDBACK': [updated_feed],
                             'RT': [rt],
                             'CORRECT_B':[corr_counter_block],
                             'CORRECT_S':[corr_counter_session],
                             'TOTAL_BLK': [curr_total_block],
                             'TOTAL_TRIAL_INBLK': [curr_block_arr[block_no-1]],
                             'TRIAL_SSE':[curr_trial_sess]
                            })
            ### Feedback
            msg_feedback = visual.TextStim(win0, text='Feedback', pos=(0, 150), height=30, units='pix')
            msg_feedback.draw()
            if updated_feed == 0:
                img_to_draw = img_incorrect
            else:
                img_to_draw = img_correct
            img_to_draw.draw()
            win0.flip()

            core.wait(feedback)
            ### Save record
            file_name = f"{subj_id}_{part_group}.csv"
            pull_file = os.path.join(RESULT_PATH, file_name)
            if curr_sess == 1 and trial_no == 1 and block_no == 1:
                if os.path.exists(pull_file):
                    print(f'DEBUG: File {pull_file} already exists! Skipping creation.')
                else:
                    print('DEBUG: Continue saving the first record.')
                    create_exp_file(subj_id,part_group,updated_output)
            elif trial_no == 1 and block_no == 1:
                p_columns = ['pDISTRIBUTION','pAMP','pphyBOUND','pDISTANCE','pTRUE_CAT',
                            'pSUB RESPONSE','pFEEDBACK','pRT']
                for col in p_columns:
                    updated_output[col] = "---"
                
                columns = ['ID','GROUP','SIDE_RULE','SESSION','DATE','TIME','TRIAL_BLK','BLK_NO', 
                    'DISTRIBUTION','AMP','LOGICAL_AMP','PHY_BOUND','DISTANCE','TRUE_CAT',
                    'SUB RESPONSE', 'FEEDBACK', 'RT', 'CORRECT_B', 'CORRECT_S',
                    'pDISTRIBUTION','pAMP','pphyBOUND','pDISTANCE','pTRUE_CAT',
                    'pSUB RESPONSE','pFEEDBACK','pRT','TOTAL_BLK', 
                    'TOTAL_TRIAL_INBLK','TRIAL_SSE']
                final_output = updated_output[columns]

                final_output.to_csv(pull_file, mode='a', header=False, index=False, encoding='utf-8-sig')
                print(f"DEBUG: Session {curr_sess} started. P-columns initialized with '---'.")
                
            else: # do not need to create a new file
                old_df = pd.read_csv(pull_file)
                last_trial = old_df.iloc[-1]
                updated_output['pDISTRIBUTION'] = last_trial['DISTRIBUTION']
                updated_output['pAMP']          = last_trial['AMP']
                updated_output['pLOGICAL_AMP']  = last_trial['LOGICAL_AMP']
                updated_output['pphyBOUND']        = last_trial['PHY_BOUND']
                updated_output['pDISTANCE']     = last_trial['DISTANCE']
                updated_output['pTRUE_CAT']     = last_trial['TRUE_CAT']
                updated_output['pSUB RESPONSE'] = last_trial['SUB RESPONSE']
                updated_output['pFEEDBACK']     = last_trial['FEEDBACK']
                updated_output['pRT']           = last_trial['RT']

                columns = ['ID','GROUP','SIDE_RULE','SESSION','DATE','TIME','TRIAL_BLK','BLK_NO', 
                    'DISTRIBUTION','AMP','LOGICAL_AMP','PHY_BOUND','DISTANCE','TRUE_CAT',
                    'SUB RESPONSE', 'FEEDBACK', 'RT', 'CORRECT_B', 'CORRECT_S',
                    'pDISTRIBUTION','pAMP','pphyBOUND','pDISTANCE','pTRUE_CAT',
                    'pSUB RESPONSE','pFEEDBACK','pRT','TOTAL_BLK', 
                    'TOTAL_TRIAL_INBLK','TRIAL_SSE']
                updated_output = updated_output[columns]

                updated_output.to_csv(pull_file, mode='a', header=False, index=False)
        
        if trial_no == curr_block_arr[block_no-1]:
            last_trial= True
        
    if block_no == curr_total_block and last_trial == True:
        finish = True 

    if finish:# INFO table,session_comp + 1
        try:
            info_df = pd.read_csv(FILENAME_INFO)
            idx = info_df[info_df['id'] == subj_id].index
            if not idx.empty:
                info_df.loc[idx, 'session_comp'] += 1
                info_df.to_csv(FILENAME_INFO, index=False, encoding='utf-8-sig')
                print(f"DEBUG: Session completed for {subj_id}. \
                      Total sessions: {info_df.loc[idx[0], 'session_comp']}")
            else:
                print(f"WARNING: ID {subj_id} not found in {FILENAME_INFO}")
        except FileNotFoundError:
            print(f"ERROR: Info file {FILENAME_INFO} not found. Cannot update session count.")
        except Exception as e:
            print(f"An error occurred: {e}")
    
        ########################################################################################
        ### FINISH PAGE
        text_stim.setText("Experiment Finished!\n\nThank you for your participation.")
        text_stim.setHeight(72)
        text_stim.pos = (0, 0)
        text_stim.alignText = 'center'
        text_stim.draw()
        win0.flip()
        core.wait(5.0)
        win0.close()
        core.quit()


# save to file
def create_exp_file(id,group,first_record):
    '''
   create a new file for a new participant and save the record of the first trial

    '''
    ## New id, create a new file
    ## file name: ID+Group
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # get current path of this script
    RESULT_PATH = os.path.join(SCRIPT_DIR, "results")

    file_name = f"{id}_{group}.csv"
    pull_file = os.path.join(RESULT_PATH, file_name)
    # previous trial data column 
    p_columns = ['pDISTRIBUTION','pAMP','pLOGICAL_AMP','pphyBOUND','pDISTANCE','pTRUE_CAT',
                'pSUB RESPONSE','pFEEDBACK','pRT']
    for col in p_columns:
        first_record[col] = "---"
        
    columns = ['ID','GROUP','SIDE_RULE','SESSION','DATE','TIME','TRIAL_BLK','BLK_NO', 
                    'DISTRIBUTION','AMP','LOGICAL_AMP','PHY_BOUND','DISTANCE','TRUE_CAT',
                    'SUB RESPONSE', 'FEEDBACK', 'RT', 'CORRECT_B', 'CORRECT_S',
                    'pDISTRIBUTION','pAMP','pphyBOUND','pDISTANCE','pTRUE_CAT',
                    'pSUB RESPONSE','pFEEDBACK','pRT','TOTAL_BLK', 
                    'TOTAL_TRIAL_INBLK','TRIAL_SSE']
        
    final_to_save = first_record[columns]
        
    ## csv file
    try:
        final_to_save.to_csv(pull_file, index=False, encoding='utf-8-sig')
        pure_file_path = os.path.join(RESULT_PATH, f"{id}{group}")
        print(f'Debug: to {pull_file}.')
        return pure_file_path
    except Exception as e:
            print('Debug: Problems saving file.')



# experiment
if __name__=='__main__':
    clear_output()
    experiment_update()