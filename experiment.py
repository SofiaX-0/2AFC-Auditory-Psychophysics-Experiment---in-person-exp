###############################################################################
# This is the main file for the experiment. 
###############################################################################

'''
2AFC Auditory Psychophysics Experiment
==================================================================================
Sound stimulus template - 300ms,10ms @ 400Hz, sampleRate = 44100Hz

Sound amplitude: sampled from 2 distributions (hard-A and hard-B) based on: 

Menichini, E., Pajot-Moric, Q., Low, R., Pedrosa, V., Pourdehghan, A., 
Vincent, P., Zhou, L., Teachen, L., & Akrami, A. (2023). 
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
Number of trials: (30 training) + 600 ~ about 1500 trials/session, fix 50 min per session
Number of subject groups: 2
-> group 1: block size: [40-60]
-> group 2: block size: [150-250]

===============================================================================    
'''

import pandas as pd
import numpy as np
from psychopy import prefs
prefs.hardware['audioLib'] = ['ptb']
prefs.hardware['audioLatencyMode'] = 2
# prefs.hardware['audioDevice'] = 'Speakers (8- US-4x4)' --> used in our lab
prefs.hardware['audioDevice'] = "扬声器 (Realtek(R) Audio)"
# prefs.hardware['audioDevice'] = "Speakers (FiiO K7)"
# import sounddevice as sd
# print('sound device:', sd.default.device)
# for i, dev in enumerate(sd.query_devices()):
#     print (i, dev['name'])
import os
from datetime import datetime
from random import randint
from calibration import SimpleCalibration
from psychopy import core, event, visual, logging, gui, sound
from psychopy.hardware import keyboard

from stimulus_generator import sample_in_block
from group_assignment import group_assign
from block_size import block_size


logging.console.setLevel(logging.CRITICAL)
calibration = SimpleCalibration()
## get fit parameters from calibration
with open("calibration_fit.txt", "r") as f:
    lines = f.readlines()

calibration.fit_a = float(lines[0])
calibration.fit_b = float(lines[1])


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
    TOTAL_TIME = 50 * 60  ## 50 mins
    AUTO_TIMEOUT_THRESHOLD = 3
    AUTO_EASY_ERROR_THRESHOLD = 3
    # bonus tier
    TIER1 = 0.90
    TIER2 = 0.80
    TIER3 = 0.70
    TIER4 = 0.60
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # get current path of this script
    RESULT_PATH = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(RESULT_PATH, exist_ok=True) # create results folder if not exists
    FILENAME_INFO = os.path.join(RESULT_PATH, "Subject_info.csv") 
  
    if os.path.exists(FILENAME_INFO):
        df_master = pd.read_csv(FILENAME_INFO)
    else:
        df_master = pd.DataFrame(columns=['id', 'group', 'age', 'gender','session_comp'])
        df_master.to_csv(FILENAME_INFO, index=False)
    
    ## feedback images
    img_correct_path = os.path.join(SCRIPT_DIR, "images","correct.png")
    img_wrong_path = os.path.join(SCRIPT_DIR, "images","wrong.png")


    Instruction_1 = (f"Welcome to this experiment!\n\nPress SPACE to continue.")
    Instruction_2 = (f"In each trial, you will hear a sound. Your task is to classify the sound into "
                     f"one of two categories: Category A or Category B.\n\nPress SPACE to continue.")
    Instruction_3 = (f"Press the [S] key for Category A. Press the [K] key for Category B."
                     f"\n\nPress SPACE to continue.")
    Instruction_4 = (f"You have {response_limit} seconds to respond. If no response is made, "
                     f"the trial will be marked as incorrect. \n\nPress SPACE to continue.")
    Instruction_5 = (f"You may pause the experiment by pressing ESC. "
                     f"You may take a short pause of up to 1 minute at any time during the experiment. "
                     f"However, only 2 long breaks of up to 5 minutes are allowed. "
                     f"Feel free to pause at any time. If a trial is interrupted by a pause, it will be presented again when the experiment resumes.\n\n"
                     f"Press SPACE to continue.")
    Instruction_6 = (f"You need to complete 50 minutes of the task. "
                     f"Pauses and breaks do not count towards the 50 minutes.\n\nPress SPACE to continue.")
    Instruction_7 = (f"Press [B] to switch between Light Mode and Dark Mode.\n\nPress SPACE to continue.")
    Instruction_8 = (f"Ready?\n\nPress SPACE to start!")
    
    choice_A = "Category A: Press [S]"
    choice_B = "Category B: Press [K]"
    response_text = 'Please classify the sound by pressing a key.'
    rest_text = "Press ESC to pause/take a long break."
    
    ### initialization
    curr_sess = 0
    trial_resp = 0 # number of trials with a response in the whole session
    trial_resp_blk = 0 # number of trials with a response in current block
    curr_trial_sess = 0 # current trial no. in the whole session
    corr_counter_block = 0
    corr_counter_session = 0
    # pause automatically when subject is not paying attention
    consecutive_timeout = 0
    consecutive_easy_error = 0
    recent_responses = []

    # =========================
    # emergency_quit
    # =========================
    def emergency_quit():
        pressed = kb.getKeys(
            keyList=['f12'],
            waitRelease=False
        )

        if pressed:
            finish_experiment()

    # =========================
    # End page
    # =========================
    def finish_experiment():
        try:
            probe.stop()
        except:
            pass

        text_stim.setText(
            f"Experiment Finished!\n\n"
            f"Thank you for your participation."
        )

        text_stim.setHeight(72*SY)
        text_stim.pos = (0, 0)
        text_stim.alignText = 'center'
        text_stim.draw()
        win0.flip()
        core.wait(5)
        win0.close()
        core.quit()

    # =========================
    # break settings
    # =========================
    break_count = 0
    MAX_BREAK = 2
    PAUSE_LIMIT = 1*60 # 1 minite short pause
    BREAK_LIMIT = 5*60 # 5 minutes long break
    total_pause_time = 0
    task_time = 0 # precise control for task time

    # ============================================================
    # BREAK FUNCTION
    # ============================================================
    def choose_break_type(break_count):
        nonlocal dark_mode, bg_color, text_color
        if break_count >= MAX_BREAK:
            return 'N'
        else:
            text_stim.setHeight(60*SY)
            text_stim.pos = (0, 0)
            text_stim.alignText = 'center'
            text_stim.setText(
                f"Pause Menu\n\n"
                f"Press [Y] for a long break (5 minutes).\n"
                f"Remaining long breaks: {MAX_BREAK - break_count}\n\n"
                f"Press [N] for a short pause (1 minute).\n\n"
                f"Press [B] to switch between Light Mode and Dark Mode."
            )

        text_stim.draw()
        win0.flip()

        while True:
            text_stim.draw()
            win0.flip()
            emergency_quit()
            keys = event.getKeys()
            for key in keys:
                if key.lower() == 'b':
                    dark_mode, bg_color, text_color = toggle_display_mode(dark_mode)
                    apply_display_mode()

                    event.clearEvents()

                elif key.lower() == 'y':
                    return 'Y'
                elif key.lower() == 'n':
                    return 'N'

            core.wait(0.05)

    def run_break(break_choice: str, remain_time: int, break_count: int):
        '''
        Control breaks and pauses

        break_choice:
            Y = long break
            N = short pause

        remain_time:
            remaining experiment time (seconds)

        break_count:
            number of long breaks already taken
        '''

        break_clock = core.Clock()
        nonlocal dark_mode, bg_color, text_color
        if break_choice == "Y" or break_choice == "y":
            time_limit = BREAK_LIMIT
            break_count += 1
        else:
            time_limit = PAUSE_LIMIT

        remain_min = remain_time // 60
        remain_sec = remain_time % 60

        r_count = 0

        # countdown period
        while break_clock.getTime() < time_limit:
            emergency_quit()
            remaining = int(time_limit - break_clock.getTime())

            mins = remaining // 60
            secs = remaining % 60

            text_stim.setText(
                f"Experiment Paused\n"
                f"Press [R] twice to continue the experiment.\n\n"
                f"Remaining break time: {mins:02d}:{secs:02d}\n"
                f"Long breaks used: {break_count}/{MAX_BREAK}\n"
                f"Remaining task time: "
                f"{remain_min:02d}:{remain_sec:02d}\n"
                f"Press [B] to switch between Light Mode and Dark Mode.\n\n"
                f"Keep focusing on accuracy to qualify for the Tier 1 bonus reward!\n\nCurrent performance level:"
            )

            text_stim.setHeight(45*SY)
            text_stim.alignText = 'center'
            text_stim.pos = (0, 120 * SY)

            update_performance_ring()
            text_stim.draw()
            performance_bg.draw()
            performance_bar.draw()

            tier4_text.draw()
            tier1_text.draw()

            win0.flip()

            keys = event.getKeys()
            for key in keys:
                if key.lower() == 'b':
                    dark_mode, bg_color, text_color = toggle_display_mode(dark_mode)
                    apply_display_mode()

                    event.clearEvents()

                elif key.lower() == 'r':
                    r_count += 1
                    if r_count == 2:
                        event.clearEvents()
                        pause_duration = break_clock.getTime()
                        return break_count, pause_duration
                else:
                    r_count = 0

            core.wait(0.05)

        # countdown finished
        text_stim.setText(
            "Time over!\n\n"
            "Press [R] twice to continue the experiment."
        )
        text_stim.setHeight(72*SY)
        text_stim.alignText = 'center'
        text_stim.pos = (0, 0)
        r_count = 0
        while r_count < 2:
            emergency_quit()
            text_stim.draw()
            win0.flip()
            keys = event.getKeys()
            for key in keys:
                if key.lower() == 'r':
                    r_count += 1
                else:
                    r_count = 0
            core.wait(0.05)

        event.clearEvents()

        pause_duration = break_clock.getTime()
        return break_count, pause_duration

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
    # ---------- display mode ----------
    dark_mode = False
    win0 = visual.Window(size=[1920, 1080], screen = 0, monitor='testMonitor',
                         fullscr=True,
                         winType='pyglet',
                         allowGUI=False,
                         waitBlanking=True)

    win0.mouseVisible = False

    # ====================================
    # UI scaling
    # ====================================

    screen_w, screen_h = win0.size

    SX = screen_w / 1920
    SY = screen_h / 1080

    print(f"Screen: {screen_w} x {screen_h}")

    ## fixation text
    ### Fixation
    fixation = visual.ShapeStim(win0, 
            vertices=((0, -30*SY), (0, 30*SY), (0, 0), (-30*SX, 0), (30*SX, 0)),
            lineWidth=5 * min(SX, SY) ,
            closeShape=False,
            lineColor='black',
            units='pix'
            )
    
    ## countdown objects
    # countdown bar
    bar_width = 1000 * SX
    bar_height = 20 * SY
    bar_y_pos = 350 * SY
    countdown_bg = visual.Rect(win0, width=bar_width, height=bar_height, 
                           pos=(0, bar_y_pos), fillColor= 'grey', 
                           lineColor= 'grey', units='pix')
    countdown_bar = visual.Rect(win0, width=bar_width, height=bar_height, 
                            pos=(0, bar_y_pos), fillColor=(0.4, 0.7, 1), 
                            lineColor=(0.4, 0.7, 1), units='pix')
            
    text_stim_resp = visual.TextStim(win0, text=response_text,
                            pos=(0, 200*SY), height=45*SY, units='pix', wrapWidth=screen_w * 0.9,
                            alignText='center')
    text_stim_A = visual.TextStim(win0, text=choice_A,
                            pos=(-400*SX, -150*SY), height=50*SY, units='pix', wrapWidth=screen_w * 0.9,
                            alignText='center')
    text_stim_B = visual.TextStim(win0, text=choice_B,
                            pos=(400*SX, -150*SY), height=50*SY, units='pix', wrapWidth=screen_w * 0.9,
                            alignText='center')
    rest_text = visual.TextStim(
        win0,
        text=rest_text,
        pos=(-screen_w/2 + 30*SX,
            screen_h/2 - 40*SY),
        height=30*SY,
        units='pix',
        wrapWidth=1500*SX,
        alignText='left',
        anchorHoriz='left'
    )

    text_stim = visual.TextStim(
    win0,
    color=(-1, -1, -1),
    units='pix',
    height=60*SY,
    wrapWidth=screen_w * 0.9,
    alignText='center',
    anchorHoriz='center',
    anchorVert='center'
    )

    tier4_text = visual.TextStim(
    win0,
    text="Tier 4",
    pos=(-250*SX, -250*SY),
    height=30*SY,
    units='pix'
    )

    tier1_text = visual.TextStim(
        win0,
        text="Tier 1",
        pos=(250*SX, -250*SY),
        height=30*SY,
        units='pix'
    )


    performance_bg = visual.Rect(
    win0,
    width=500*SX,
    height=40*SY,
    pos=(0, -300*SY),
    fillColor='grey',
    lineColor='grey',
    units='pix'
    )

    performance_bar = visual.Rect(
        win0,
        width=0,
        height=40*SY,
        pos=(-250*SX, -300*SY),
        fillColor=(0.4, 0.7, 1),
        lineColor=(0.4, 0.7, 1),
        units='pix'
    )

    def toggle_display_mode(dark_mode):
        dark_mode = not dark_mode
        if dark_mode:
            bg_color = (-1, -1, -1)
            text_color = (1, 1, 1)

        else:
            bg_color = (0.8, 0.8, 0.8)
            text_color = (-1, -1, -1)

        return dark_mode, bg_color, text_color


    def apply_display_mode():
        win0.color = bg_color
        text_stim.color = text_color
        text_stim_resp.color = text_color
        text_stim_A.color = text_color
        text_stim_B.color = text_color
        rest_text.color = text_color
        fixation.lineColor = text_color
        performance_bg.lineColor = text_color
        tier4_text.color = text_color
        tier1_text.color = text_color

    def update_performance_ring():
        accuracy = corr_counter_session / max(1, curr_trial_sess - 1)
        progress = max(0, min((accuracy - TIER4) / (TIER1 - TIER4), 1.0))

        bar_width = 500 * SX * progress

        performance_bar.width = bar_width
        performance_bar.pos = (-250 * SX + bar_width/2, -300 * SY)

        if accuracy >= TIER1:
            performance_bar.fillColor = "gold"
            performance_bar.lineColor = "gold"

        elif accuracy >= TIER2:
            performance_bar.fillColor = "limegreen"
            performance_bar.lineColor = "limegreen"
        elif accuracy >= TIER3:
            performance_bar.fillColor = "red"
            performance_bar.lineColor = "red"
        else:
            performance_bar.fillColor = (0.4, 0.7, 1)
            performance_bar.lineColor = (0.4, 0.7, 1)

    kb = keyboard.Keyboard()
    probe = sound.Sound(
        value=np.zeros(int(44100 * 0.3), dtype=np.float32),
        stereo=True,
        sampleRate=44100
    )


    # ---------- instruction pages ----------

    instruction_pages = [
        Instruction_1,
        Instruction_2,
        Instruction_3,
        Instruction_4,
        Instruction_5,
        Instruction_6,
        Instruction_7,
        Instruction_8
    ]

    # ---------- default mode ----------

    bg_color = (0.8, 0.8, 0.8)
    text_color = (-1, -1, -1)
    apply_display_mode()

    win0.color = bg_color
    text_stim.color = text_color
    event.clearEvents()
    page_idx = 0
    while page_idx < len(instruction_pages):
        emergency_quit() # experimenter emergency quit: Ctrl + Shift + Q
        
        # current instruction page
        text_stim.setText(instruction_pages[page_idx])
        win0.color = bg_color
        text_stim.color = text_color
        text_stim.draw()
        win0.flip()
        keys = event.getKeys(keyList=['space', 'b'])
        # next page
        if 'space' in keys:
            page_idx += 1
            event.clearEvents()
            core.wait(0.15)
        # toggle dark/light mode
        elif 'b' in keys:
            dark_mode, bg_color, text_color = toggle_display_mode(dark_mode)
            apply_display_mode()
            event.clearEvents()
    # ---------- keep selected theme ----------
    win0.color = bg_color
    text_stim.color = text_color
    fixation.lineColor = text_color
    text_stim_resp.color = text_color
    text_stim_A.color = text_color
    text_stim_B.color = text_color
    text_stim_B.color = text_color
    rest_text.color = text_color
    event.clearEvents()

    # ====================================
    # Experiment timer
    # ====================================

    experiment_clock = core.Clock()
    experiment_clock.reset()

    ## =========================================================================
    ## SESSION CONDITIONS -------------------------------------------------------
    ## 1) feedback imgs & Block size arrangement; => curr_block_arr, curr_total_block
    ## 2) Whether to include training trials. [dep. on: current session]
    ## 3) initialize finish and last_trial flags.
    ## 4) side rule: 0 = A (left) for lower amp ; 1 = A (left) for higher amp

    ### feedback images setting
    img_correct = visual.ImageStim(win0, image=img_correct_path, 
                                   size=(500*SX, 500*SY), units='pix')
    img_incorrect = visual.ImageStim(win0, image=img_wrong_path, 
                                     size=(500*SX, 500*SY), units='pix')

    # BLOCK ARRANGEMENT
    curr_block_arr, curr_total_block = block_size(part_group)
    print(f'DEBUG: block sizes: {curr_block_arr}; total block: {curr_total_block}')
    if part_group == 1:
        if randint(0, 1) == 0:
            first_dist = 1
        else:
            first_dist = 2

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
        if finish:
            break
        event.clearEvents()
        trial_resp_blk = 0
        corr_counter_block = 0
        updated_exp_date = datetime.today().strftime('%Y%m%d')
        updated_block_no = block_no # current block
        if part_group == 1:
            # ---------- Long-block group ----------
            if train and block_no == 1:
                updated_dist = 0
            else:
                if train:
                    formal_idx = block_no - 2
                else:
                    formal_idx = block_no - 1
                if formal_idx % 2 == 0:
                    updated_dist = first_dist
                else:
                    updated_dist = 3 - first_dist
        else:
            # ---------- Short-block group ----------
            if train:
                if block_no == 1:
                    updated_dist = 0
                else:
                    updated_dist = randint(1, 2)
            else:
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
        easy_threshold = df_block_sample['Easy_threshold'].iloc[0]

        for trial_no in range(1, curr_block_arr[block_no-1] + 1):
            effective_time = task_time

            if task_time >= TOTAL_TIME:
                finish = True
                break
            event.clearEvents()
            ### update file info
            updated_exp_time = datetime.today().strftime('%H%M%S')
            updated_blk_trial = trial_no # current trial in block
            curr_trial_sess += 1

            trial_finished = False
            while not trial_finished:
            
                ### fixation
                fixation.draw()
                win0.flip()
                core.wait(fixation_duration)

                ### Listen
                #### set stimulus amplitude
                updated_amp = amplitudes.iloc[trial_no-1]
                playback_volume = calibration.dba_to_amplitude(updated_amp)
                playback_volume = np.clip(playback_volume, 0.0, 1.0) # ensure it stays valid
                updated_logical_value = logical_values.iloc[trial_no-1]
                distance = distances.iloc[trial_no-1]
                true_cat = corr_sides.iloc[trial_no-1]
                # play
                signal = calibration.generate_white_noise(playback_volume, duration=0.3)
                probe.setSound(signal) # use the same noise in calibration
                probe.play()
                sound_clock = core.Clock()
                paused = False
                while sound_clock.getTime() < probe.getDuration():
                        emergency_quit()
                        keys = event.getKeys()

                        if 'escape' in keys:
                            try:
                                probe.stop()
                            except:
                                pass
                            choice = choose_break_type(break_count)
                            break_count, pause_duration = run_break(choice, int(TOTAL_TIME - effective_time), break_count)
                            total_pause_time += pause_duration
                            paused = True
                            event.clearEvents()
                            break

                if paused:
                    continue

                ### Respond
                resp_clock = core.Clock()
                event.clearEvents()
                updated_sub_response = -1
                updated_feed = 0
                rt = -1
                responded = False
                while resp_clock.getTime() < response_limit:
                    emergency_quit()
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
                    rest_text.draw()
                    win0.flip()

                    keys = event.getKeys(timeStamped=resp_clock)
                    if keys:
                        for key_name, key_time in keys:
                            # ====================================
                            # BREAK
                            # ====================================

                            if key_name == 'escape':
                                try:
                                    probe.stop()
                                except:
                                    pass
                                remain_time = int(TOTAL_TIME - effective_time)
                                choice = choose_break_type(break_count)
                                break_count, pause_duration = run_break(choice, int(TOTAL_TIME - effective_time), break_count)
                                total_pause_time += pause_duration
                                paused = True
                                event.clearEvents()
                                break

                            # ====================================
                            # response handling
                            # ====================================

                            if key_name not in ['s', 'k']:
                                continue

                            rt = key_time
                            trial_resp += 1
                            trial_resp_blk += 1

                            if key_name == 's':

                                updated_sub_response = 0

                                if true_cat == 0: # low intensity

                                    if side_rule == 0: # low left; high right

                                        print('DEBUG: Subject Correct.')
                                        updated_feed = 1
                                        corr_counter_block += 1
                                        corr_counter_session += 1

                                    else: # low right, high left

                                        print('DEBUG: Subject Incorrect.')
                                        updated_feed = 0

                                elif true_cat == 1: # high intensity

                                    if side_rule == 0: # low left; high right

                                        print('DEBUG: Subject Incorrect.')
                                        updated_feed = 0

                                    else: # low right, high left

                                        print('DEBUG: Subject Correct.')
                                        updated_feed = 1
                                        corr_counter_block += 1
                                        corr_counter_session += 1

                                print('Debug: Trial Number: %d' %(trial_no))
                                print('Debug: A chosen.')

                                responded = True
                                recent_responses.append((updated_sub_response, updated_feed))
                                if len(recent_responses) > 8:
                                    recent_responses.pop(0)
                                consecutive_timeout = 0

                                break

                            elif key_name == 'k':

                                updated_sub_response = 1 # pressed right key

                                if true_cat == 0: # low intensity

                                    if side_rule == 0:

                                        print('DEBUG: Subject Incorrect.')
                                        updated_feed = 0

                                    else:

                                        updated_feed = 1
                                        corr_counter_block += 1
                                        corr_counter_session += 1

                                        print('DEBUG: Subject Correct.')

                                elif true_cat == 1: # hight intensity

                                    if side_rule == 0:

                                        updated_feed = 1
                                        corr_counter_block += 1
                                        corr_counter_session += 1

                                        print('DEBUG: Subject Correct.')

                                    else:

                                        print('DEBUG: Subject Incorrect.')
                                        updated_feed = 0

                                print('DEBUG: Trial Number: %d' %(trial_no))
                                print('DEBUG: B chosen.')

                                responded = True
                                recent_responses.append((updated_sub_response, updated_feed))
                                if len(recent_responses) > 8:
                                    recent_responses.pop(0)
                                consecutive_timeout = 0

                                break
                        if paused or responded:
                            break
                if rt == -1:
                    if paused:
                        continue
                    updated_feed = 0
                    consecutive_timeout += 1
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
                if updated_feed == 0:
                    img_to_draw = img_incorrect
                else:
                    img_to_draw = img_correct
                img_to_draw.draw()
                win0.flip()

                core.wait(feedback)
                task_time += fixation_duration
                task_time += probe.getDuration()
                task_time += feedback

                if rt == -1:
                    task_time += response_limit
                else:
                    task_time += rt

                ## count easy trial error
                if rt != -1 and updated_feed == 0 and distance >= easy_threshold:
                    consecutive_easy_error += 1
                else:
                    consecutive_easy_error = 0
                ## stop when several consecutive no response
                if consecutive_timeout >= AUTO_TIMEOUT_THRESHOLD:
                    text_stim.setHeight(60*SY)
                    text_stim.pos = (0,0)
                    text_stim.alignText = 'center'
                    text_stim.setText(
                        f"Three consecutive trials received no response.\n\n"
                        f"The experiment has been paused automatically."
                    )

                    text_stim.draw()
                    win0.flip()
                    core.wait(3)
                    event.clearEvents()
                    choice = "N"

                    break_count, pause_duration = run_break(
                        choice,
                        int(TOTAL_TIME - effective_time),
                        break_count
                    )

                    total_pause_time += pause_duration

                    consecutive_timeout = 0
                    recent_responses.clear()
                
                ## stop when 3 consecutive easy trial errors
                if consecutive_easy_error >= AUTO_EASY_ERROR_THRESHOLD:
                    text_stim.setHeight(60*SY)
                    text_stim.pos = (0,0)
                    text_stim.alignText = 'center'
                    text_stim.setText(
                        f"Several easy trials were answered incorrectly.\n\n"
                        f"Please take a short pause and refocus."
                    )

                    text_stim.draw()
                    win0.flip()
                    core.wait(3)
                    event.clearEvents()
                    choice = "N"

                    break_count, pause_duration = run_break(
                        choice,
                        int(TOTAL_TIME - effective_time),
                        break_count
                    )

                    total_pause_time += pause_duration

                    consecutive_easy_error = 0
                    recent_responses.clear()

                ## stop when getting 8 same responses with 4 wrongs
                if len(recent_responses) == 8:
                    same_key = all(
                        r[0] == recent_responses[0][0]
                        for r in recent_responses
                    )

                    error_count = sum(
                        1 for r in recent_responses
                        if r[1] == 0
                    )

                    if same_key and error_count >= 4:
                        text_stim.setHeight(60*SY)
                        text_stim.pos = (0,0)
                        text_stim.alignText = 'center'
                        text_stim.setText(
                        f"A repetitive response pattern was detected.\n\n"
                        f"Please take a short pause and refocus."
                    )

                        text_stim.draw()
                        win0.flip()
                        core.wait(3)
                        event.clearEvents()
                        choice = "N"

                        break_count, pause_duration = run_break(
                            choice,
                            int(TOTAL_TIME - effective_time),
                            break_count
                        )

                        total_pause_time += pause_duration

                        recent_responses.clear()

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
                trial_finished = True
        
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

        finish_experiment()
    
        ########################################################################################

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