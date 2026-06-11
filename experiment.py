###############################################################################
# This is the main file for the experiment. 
###############################################################################

'''
2AFC Auditory Psychophysics Experiment
==================================================================================
Sound stimulus template -
- For calibration: 4s calibration stimulus (10 repetitions of a fixed 400ms white-noise segment), sampleRate = 44000Hz
- Stimuli: fixed 400ms white-noise, sampleRate = 44000Hz


Sound amplitude: sampled from 2 distributions (hard-A and hard-B) based on: 

Menichini, E., Pajot-Moric, Q., Low, R., Pedrosa, V., Pourdehghan, A., 
Vincent, P., Zhou, L., Teachen, L., & Akrami, A. (2023). 
Different learning algorithms achieve shared optimal outcomes in humans, rats, 
and mice. bioRxiv, 2023.01.30.526119. https://doi.org/10.1101/2023.01.30.526119


Interstimulus intervals:
        100ms pre-fixation;
        500ms fixation;
        100ms post-fixation;
        (400ms stimulus) with up to 5000ms response window;
        500ms feedback;
        up to 3000ms next-trial page
    --------------------------------
    Estimated duration: 3000-9200 ms / trial

Number of sessions: 5; extra sessions 6-7 available
Number of trials:
    Session 1: 30 warm-up + 600 formal trials
    Session 2-7: 600 formal trials
Number of subject groups: 2
-> group 1: block size: [40-60]
-> group 2: block size: [150-250]

===============================================================================    
'''

import pandas as pd
import numpy as np
from psychopy import prefs
prefs.hardware['audioLib'] = ['ptb']
prefs.hardware['audioLatencyMode'] = 3
# prefs.hardware['audioDevice'] = 'Speakers (8- US-4x4)' --> used in our lab
prefs.hardware['audioDevice'] = "扬声器 (Realtek(R) Audio)"
# prefs.hardware['audioDevice'] = "Speakers (FiiO K7)" --> used in our lab
# import sounddevice as sd
# print('sound device:', sd.default.device)
# for i, dev in enumerate(sd.query_devices()):
#     print (i, dev['name'])
import os
from scipy.io.wavfile import read
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
    FORMAL_TRIAL_NUMBER = 600
    response_limit = 5
    fixation_duration = 0.5
    feedback = 0.5
    PRE_FIXATION = 0.1
    POST_FIXATION = 0.1
    AUTO_TIMEOUT_THRESHOLD = 3
    AUTO_EASY_ERROR_THRESHOLD = 3
    # bonus tier
    TIER1 = 0.95
    TIER2 = 0.90
    TIER3 = 0.80
    TIER4 = 0.70
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # get current path of this script
    RESULT_PATH = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(RESULT_PATH, exist_ok=True) # create results folder if not exists
    FILENAME_INFO = os.path.join(RESULT_PATH, "Subject_info.csv") 
  
    if os.path.exists(FILENAME_INFO):
        df_master = pd.read_csv(
            FILENAME_INFO,
            dtype={'id': str}
        )
    else:
        df_master = pd.DataFrame(columns=['id', 'group', 'age', 'gender','session_comp'])
        df_master.to_csv(FILENAME_INFO, index=False)
    
    ## feedback images
    img_correct_path = os.path.join(SCRIPT_DIR, "images","correct.png")
    img_wrong_path = os.path.join(SCRIPT_DIR, "images","wrong.png")
    
    choice_A = "A"
    choice_B = "B"
    
    ### initialization
    curr_sess = 0
    trial_resp = 0 # number of trials with a response in the whole session
    trial_resp_blk = 0 # number of trials with a response in current block
    curr_trial_sess = 0 # current trial no. in the whole session
    corr_counter_block = 0
    corr_counter_session = 0
    formal_trial_count = 0
    formal_correct_count = 0
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
    break_used = False

    # ============================================================
    # BREAK FUNCTION
    # ============================================================

    def run_break():
        resume_rect, resume_text = create_button(
            "Resume",
            pos=(0, -120 * SY),
            width=260,
            height=80
        )
        break_clock = core.Clock()
        time_limit = 5 * 60

        # countdown period
        while True:
            if resume_rect.contains(mouse):
                resume_rect.fillColor = (0.4, 0.7, 1)
            else:
                resume_rect.fillColor = (0.8, 0.8, 0.8)

            if mouse.isPressedIn(resume_rect):
                while mouse.getPressed()[0]:
                    core.wait(0.01)
                event.clearEvents()
                mouse.clickReset()
                core.wait(0.2)

                return
            
            emergency_quit()
            remaining = max(0,int(time_limit - break_clock.getTime()))

            mins = remaining // 60
            secs = remaining % 60

            if remaining > 0:

                text_stim.setText(
                    f"Experiment Paused\n"
                    f"Remaining break time: {mins:02d}:{secs:02d}\n"
                    f"Keep focusing on accuracy to qualify for the Tier 1 bonus reward!"
                )

            else:

                text_stim.setText(
                    "Break finished.\n\n"
                    "Click Resume to continue.\n\n"
                )

            text_stim.setHeight(45*SY)
            text_stim.alignText = 'center'
            text_stim.pos = (0, 120 * SY)

            text_stim.draw()

            resume_rect.draw()
            resume_text.draw()

            win0.flip()
            core.wait(0.01)
       

    ## ===================================================================
    ## INFO PAGE ---------------------------------------------------------
    ## => need to read & store information of subjects:
    ## === initials, age, gender (store OR match with IDs in Subject_info.csv);
    ## === check which session is this one (do not store for now), and whether it's correct

    def get_verified_info():

        while True:

            # ====================================
            # Experimenter setup
            # ====================================

            setup_dlg = gui.Dlg(title="Participant Setup")

            if hasattr(setup_dlg, 'labelStack'):
                setup_dlg.labelStack = []

            setup_dlg.addText("Participant ID *")
            setup_dlg.addField("ID:", "")

            setup_dlg.addText("Current Session *")
            session_choices = [str(i) for i in range(1, TOTAL_SESSION + 1)]
            setup_dlg.addField("Session:", choices=session_choices)

            setup_dlg.show()

            if not setup_dlg.OK:
                core.quit()

            subj_id = (str(setup_dlg.data[0]).strip().upper())

            if subj_id == "":
                err = gui.Dlg(title="Input Error")
                err.addText("Participant ID cannot be empty.")
                err.show()
                continue

            temp_sess = int(setup_dlg.data[1])

            # ====================================
            # Session consistency check
            # ====================================

            error_msg_text = ""

            if os.path.exists(FILENAME_INFO):

                df_master_local = pd.read_csv(
                    FILENAME_INFO,
                    dtype={'id': str}
                )

                existing_records = df_master_local[
                    df_master_local['id'] == subj_id
                ]

                if not existing_records.empty:

                    session_comp = int(
                        existing_records.iloc[0]['session_comp']
                    )

                    expected_session = session_comp + 1

                    if temp_sess != expected_session:

                        error_msg_text = (
                            f"Session mismatch!\n\n"
                            f"Completed sessions: {session_comp}\n"
                            f"Expected session: {expected_session}\n"
                            f"Selected session: {temp_sess}"
                        )

                else:

                    if temp_sess != 1:

                        error_msg_text = (
                            f"New ID detected ({subj_id}).\n\n"
                            f"Session must be 1."
                        )

            else:

                if temp_sess != 1:

                    error_msg_text = (
                        "Master file not found.\n\n"
                        "First participant must start Session 1."
                    )

            if error_msg_text != "":

                logic_err = gui.Dlg(title="Registration Error")
                logic_err.addText(error_msg_text)
                logic_err.show()

                continue

            # ====================================
            # Participant information
            # ====================================

            while True:

                dlg = gui.Dlg(title="Participant Information")

                if hasattr(dlg, 'labelStack'):
                    dlg.labelStack = []

                dlg.addText(f"Participant ID: {subj_id}")
                dlg.addText(f"Session: {temp_sess}")

                dlg.addText("Age *")
                dlg.addField(key='age', label=' ', initial='')

                dlg.addText("Gender *")
                dlg.addField(
                    key='gender',
                    label=' ',
                    choices=['Male', 'Female', 'Other']
                )

                if hasattr(dlg, 'cancelButton'):
                    dlg.cancelButton = None

                dlg.show()

                data = dlg.data

                if data is None or not str(data[0]).strip():

                    err = gui.Dlg(title="Input Error")
                    err.addText("Age cannot be empty.")
                    err.show()

                    continue

                try:

                    temp_age = int(data[0])
                    temp_gender = data[1]

                except (ValueError, TypeError):

                    err = gui.Dlg(title="Input Error")
                    err.addText("Age must be a valid number.")
                    err.show()

                    continue

                # ====================================
                # Double Check
                # ====================================

                conf_dlg = gui.Dlg(title="Double Check")

                if hasattr(conf_dlg, 'labelStack'):
                    conf_dlg.labelStack = []

                conf_dlg.addText("Please check the information below:")

                conf_dlg.addText(f"Participant ID: {subj_id}")
                conf_dlg.addText(f"Session: {temp_sess}")
                conf_dlg.addText(f"Age: {temp_age}")
                conf_dlg.addText(f"Gender: {temp_gender}")

                conf_dlg.addText(
                    "\nClick OK to confirm, Cancel to re-enter."
                )

                conf_dlg.show()

                if conf_dlg.OK:

                    return (
                        subj_id,
                        temp_sess,
                        temp_age,
                        temp_gender
                    )

    # --- Execution ---
    subj_id, curr_sess, subj_age, subj_gender = get_verified_info()

    df_master = pd.read_csv(
        FILENAME_INFO,
        dtype={'id': str}
    )
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

    # print(f'DEBUG: Subject {subj_id} is in Group {part_group}.')

    ## ===================================================================
    ## BEGIN & INSTRUCTION PAGE ------------------------------------------
    # ---------- display mode ----------
    dark_mode = False
    win0 = visual.Window(size=[2560,1440], screen = 0, monitor='testMonitor', # 2560*1440 in lab
                         fullscr=False,
                         winType='pyglet',
                         allowGUI=False,
                         waitBlanking=True)
    mouse = event.Mouse(
        visible=True,
        win=win0
    )
    win0.mouseVisible = True

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
            
    text_stim_A = visual.TextStim(win0, text=choice_A,
                            pos=(-400*SX, -150*SY), height=50*SY, units='pix', wrapWidth=screen_w * 0.9,
                            alignText='center')
    text_stim_B = visual.TextStim(win0, text=choice_B,
                            pos=(400*SX, -150*SY), height=50*SY, units='pix', wrapWidth=screen_w * 0.9,
                            alignText='center')

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
    pos=(-320*SX, 340*SY),
    height=30*SY,
    units='pix',
    color=(-1, -1, -1)
    )

    tier1_text = visual.TextStim(
        win0,
        text="Tier 1",
        pos=(320*SX, 340*SY),
        height=30*SY,
        units='pix',
        color=(-1, -1, -1)
    )


    performance_bg = visual.Rect(
    win0,
    width=500*SX,
    height=40*SY,
    pos=(0, 340*SY),
    fillColor=(0.75, 0.75, 0.75),
    lineColor=(0.75, 0.75, 0.75),
    units='pix'
    )

    performance_bar = visual.Rect(
        win0,
        width=0,
        height=40*SY,
        pos=(-250*SX, 340*SY),
        fillColor=(0.55, 0.75, 0.9),
        lineColor=(0.55, 0.75, 0.9),
        units='pix'
    )

    # ====================================
    # Trial Progress Bar
    # ====================================

    trial_progress_bg = visual.Rect(
        win0,
        width=700 * SX,
        height=25 * SY,
        pos=(0, 480 * SY),
        fillColor=(0.65, 0.65, 0.65),
        lineColor=None,
        units='pix'
    )

    trial_progress_bar = visual.Rect(
        win0,
        width=1,
        height=25 * SY,
        pos=(-350 * SX, 480 * SY),
        fillColor=(0.4, 0.7, 1),
        lineColor=None,
        units='pix'
    )


    def update_performance_bar():
        accuracy = formal_correct_count / max(1, formal_trial_count)
        progress = max(0, min((accuracy - TIER4) / (TIER1 - TIER4), 1.0))

        bar_width = 500 * SX * progress

        performance_bar.width = bar_width
        performance_bar.pos = (-250 * SX + bar_width/2, 340 * SY)

        if accuracy >= TIER1:
            performance_bar.fillColor = (0.85, 0.75, 0.35)
            performance_bar.lineColor = (0.85, 0.75, 0.35)

        elif accuracy >= TIER2:
            performance_bar.fillColor = (0.50, 0.75, 0.50)
            performance_bar.lineColor = (0.50, 0.75, 0.50)
        elif accuracy >= TIER3:
            performance_bar.fillColor = (0.85, 0.55, 0.55)
            performance_bar.lineColor = (0.85, 0.55, 0.55)
        else:
            performance_bar.fillColor = (0.60, 0.75, 0.90)
            performance_bar.lineColor = (0.60, 0.75, 0.90)

    ## Progress bar
    def update_trial_progress(progress):

        progress = max(0, min(progress, 1))

        width_now = 700 * SX * progress

        trial_progress_bar.width = max(1, width_now)

        trial_progress_bar.pos = (
            -350 * SX + width_now / 2,
            480 * SY
        )
    
    ### keyboard
    kb = keyboard.Keyboard()

    ### Buttons
    def create_button(text,
                  pos=(0, 0),
                  width=220,
                  height=70):

        button_rect = visual.Rect(
            win0,
            width=width * SX,
            height=height * SY,
            pos=pos,
            fillColor=(0.8, 0.8, 0.8),
            lineColor=(-1, -1, -1),
            lineWidth=2,
            units='pix'
        )

        button_text = visual.TextStim(
            win0,
            text=text,
            pos=pos,
            height=32 * SY,
            units='pix',
            color=(-1, -1, -1),
        )

        return button_rect, button_text
    

    def wait_for_button(button_rect,
                    button_text,
                    page_text=None):
        win0.mouseVisible = True
        text_stim.pos = (0,0)
        mouse.clickReset()

        while True:

            emergency_quit()

            if page_text is not None:
                text_stim.setText(page_text)
                text_stim.pos = (0, 150 * SY)
                text_stim.draw()
            
            button_text.color = text_color
            button_rect.lineColor = text_color
            
            if button_rect.contains(mouse):

                button_rect.fillColor = (0.4, 0.7, 0.1)

            else:

                button_rect.fillColor = (0.8, 0.8, 0.8)

            button_rect.draw()
            button_text.draw()

            win0.flip()

            if mouse.isPressedIn(button_rect):
                text_stim.pos = (0, 0)

                core.wait(0.2)

                while mouse.getPressed()[0]:
                    core.wait(0.01)

                return

            core.wait(0.01)
    

    ### Choice: A/B buttons
    A_rect, A_text = create_button(
        "A",
        pos=(-250 * SX, -150 * SY),
        width=320,
        height=120
    )

    B_rect, B_text = create_button(
        "B",
        pos=(250 * SX, -150 * SY),
        width=320,
        height=120
    )

    
        
    ## Instructions at the begining
    def show_instruction_page(
        text,
        button_label="Next"
    ):
        button_width = max(260, len(button_label) * 22)

        btn_rect, btn_text = create_button(
            button_label,
            pos=(0, -250 * SY),
            width=button_width
        )

        wait_for_button(
            btn_rect,
            btn_text,
            text,
        )


    def next_trial_page(in_warmup):
        win0.mouseVisible = True

        nonlocal break_used

        next_rect, next_text = create_button(
            "Next Trial",
            pos=(0, 0),
            width=280,
            height=80
        )

        if not break_used and not in_warmup:

            break_rect, break_text = create_button(
                "Take Break",
                pos=(-750*SX, 450*SY),
                width=280,
                height=80
            )

        page_clock = core.Clock()

        mouse.clickReset()

        while page_clock.getTime() < 3:

            emergency_quit()
            if not in_warmup:
                update_performance_bar()

                text_stim.setText("Current performance level")
                text_stim.pos = (0, 220 * SY)
                text_stim.height = 40 * SY

                text_stim.draw()

                performance_bg.draw()
                performance_bar.draw()

                tier4_text.draw()
                tier1_text.draw()
                        # ---------- Next button ----------
            if next_rect.contains(mouse):
                next_rect.fillColor = (0.4, 0.7, 1)
            else:
                next_rect.fillColor = (0.8, 0.8, 0.8)

            next_rect.draw()
            next_text.draw()

            # ---------- Break button ----------
            if not break_used and not in_warmup:

                if break_rect.contains(mouse):
                    break_rect.fillColor = (0.4, 0.7, 1)
                else:
                    break_rect.fillColor = (0.8, 0.8, 0.8)

                break_rect.draw()
                break_text.draw()

            win0.flip()

            # ---------- Next ----------
            if mouse.isPressedIn(next_rect):

                while mouse.getPressed()[0]:
                    core.wait(0.01)

                return

            # ---------- Break ----------
            if (not break_used
                    and not in_warmup
                    and mouse.isPressedIn(break_rect)):

                while mouse.getPressed()[0]:
                    core.wait(0.01)

                run_break()

                break_used = True

                return

            core.wait(0.01)


    # ---------- default mode ----------

    bg_color = (0.8, 0.8, 0.8)
    text_color = (-1, -1, -1)
    win0.color = bg_color
    text_stim.color = text_color
    event.clearEvents()
    show_instruction_page(
        "Welcome!\n\n"
        "In this experiment, you will hear sounds and make choices by clicking either Button A or Button B.\n"
        "You will receive feedback after each response.\n"
        "Please DO NOT change the volume during the experiment.",
        "Next"
    )

    show_instruction_page(
        "Your goal is to make as many correct responses as possible.\n\n"
        "Please pay attention throughout the experiment. "
        "If your performance is too low, the experiment may be paused automatically. ",
        "Next"
    )

    show_instruction_page(
        "During the experiment, you can click [Next Trial] to start the next trial immediately.\n\n"
        "If you do not click the button, the next trial will start automatically after a short delay. ",
        "Next"
    )

    if curr_sess == 1:
        show_instruction_page(
            "You will first complete 30 warm-up trials.\n\n"
            "Please respond as accurately as possible.",
            "Start Warm-up"
        )
    else:

        show_instruction_page(
            "Click the button below when you are ready to begin.\n\n"
            "You can choose to take ONE 5-minute break during the experiment. "
            "Try to make as many correct responses as possible and get the highest tier reward!",
            "Start Experiment"
        )

    # ====================================
    # Audio warm-up (PTB initialization)
    # ====================================

    warmup_sound = sound.Sound(
        "stimulus_400ms.wav"
    )

    warmup_sound.setVolume(0.0)

    warmup_sound.play()

    core.wait(0.5)
    
    # ---------- keep selected theme ----------
    win0.color = bg_color
    text_stim.color = text_color
    fixation.lineColor = text_color
    text_stim_A.color = text_color
    text_stim_B.color = text_color
    event.clearEvents()

    ## =========================================================================
    ## SESSION CONDITIONS -------------------------------------------------------
    ## 1) feedback imgs & Block size arrangement; => curr_block_arr, curr_total_block
    ## 2) Whether to include training trials. [dep. on: current session]
    ## 3) initialize finish and last_trial flags.
    ## 4) side rule: 0 = A (left) for lower amp ; 1 = A (left) for higher amp

    ### feedback images setting
    img_correct = visual.ImageStim(win0, image=img_correct_path, 
                                   size=(250*SX, 250*SY),  pos=(0, 150*SY), units='pix')
    img_incorrect = visual.ImageStim(win0, image=img_wrong_path, 
                                     size=(250*SX, 250*SY),  pos=(0, 150*SY), units='pix')

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
    main_trial_counter = 0
    
    finish = False
    last_trial = False

    ## =========================================================================
    ## MAIN LOOP

    for block_no in range(1, curr_total_block+1):
        if finish:
            break

        event.clearEvents()
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

            event.clearEvents()
            ### update file info
            updated_exp_time = datetime.today().strftime('%H%M%S')
            updated_blk_trial = trial_no # current trial in block
            curr_trial_sess += 1
            if not (train and block_no == 1):
                main_trial_counter += 1
            if train and block_no == 1:

                # Warm-up
                progress = (trial_no / 30)

            else:

                # Main Experiment
                progress = (main_trial_counter / FORMAL_TRIAL_NUMBER)

            trial_finished = False
            button_layout = randint(0, 1)
            while not trial_finished:
                ### button layout
                if button_layout == 0:
                    A_rect.pos = (-250 * SX, -150 * SY)
                    A_text.pos = (-250 * SX, -150 * SY)

                    B_rect.pos = (250 * SX, -150 * SY)
                    B_text.pos = (250 * SX, -150 * SY)

                else:

                    A_rect.pos = (250 * SX, -150 * SY)
                    A_text.pos = (250 * SX, -150 * SY)

                    B_rect.pos = (-250 * SX, -150 * SY)
                    B_text.pos = (-250 * SX, -150 * SY)
                

                ### fixation
                core.wait(PRE_FIXATION)
                fixation.draw()
                win0.flip()
                core.wait(fixation_duration)
                core.wait(POST_FIXATION)

                ### Listen
                #### set stimulus amplitude
                updated_amp = amplitudes.iloc[trial_no-1]
                playback_volume = calibration.dba_to_amplitude(updated_amp)
                playback_volume = np.clip(playback_volume, 0.0, 1.0) # ensure it stays valid

                ### DEBUG
                print(
                    f"Trial={curr_trial_sess}, "
                    f"Block={block_no}, "
                    f"Volume={playback_volume:.6f}"
                )

                updated_logical_value = logical_values.iloc[trial_no-1]
                distance = distances.iloc[trial_no-1]
                true_cat = corr_sides.iloc[trial_no-1]

                ### Respond
                resp_clock = core.Clock()

                stim_started = False
                event.clearEvents()
                updated_sub_response = -1
                updated_feed = -1
                rt = -1
                responded = False
               
                while resp_clock.getTime() < response_limit:
                    if not stim_started:

                        probe = sound.Sound(
                            "stimulus_400ms.wav"
                        )

                        probe.setVolume(playback_volume)

                        probe.play()

                        stim_started = True

                    win0.mouseVisible = True
                    emergency_quit()

                    # A hover
                    if A_rect.contains(mouse):

                        A_rect.fillColor = (0.4, 0.7, 0.1)

                    else:

                        A_rect.fillColor = (0.8, 0.8, 0.8)

                    # B hover
                    if B_rect.contains(mouse):

                        B_rect.fillColor = (0.4, 0.7, 0.1)

                    else:

                        B_rect.fillColor = (0.8, 0.8, 0.8)
                
                    update_trial_progress(progress)
                    trial_progress_bg.draw()
                    trial_progress_bar.draw()
                    A_rect.draw()
                    A_text.draw()

                    B_rect.draw()
                    B_text.draw()
                    win0.flip() 

                    # mouse response
                    if mouse.isPressedIn(A_rect):

                        rt = resp_clock.getTime()

                        updated_sub_response = 0

                        responded = True

                        break

                    elif mouse.isPressedIn(B_rect):

                        rt = resp_clock.getTime()

                        updated_sub_response = 1


                        responded = True

                        break

                if responded:

                    trial_resp += 1
                    trial_resp_blk += 1

                    # A = category A
                    if updated_sub_response == true_cat:
                        if not (train and block_no == 1):
                            formal_correct_count += 1
                        updated_feed = 1
                        corr_counter_block += 1
                        corr_counter_session += 1

                    else:

                        updated_feed = 0

                    recent_responses.append(
                        (updated_sub_response, updated_feed)
                    )

                    if len(recent_responses) > 8:
                        recent_responses.pop(0)

                    consecutive_timeout = 0   
                    
                
                if rt == -1:
                    consecutive_timeout += 1
                    # print(f"DEBUG: Trial {trial_no} - Timed Out! Auto-switching to feedback.")
                if not (train and block_no == 1):
                        formal_trial_count += 1
                win0.flip()
                event.clearEvents()


                updated_output = pd.DataFrame({'ID':[subj_id],
                                'GROUP': [part_group],
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
                # Highlight selected button
                if updated_sub_response == 0:

                    A_rect.fillColor = (0.4, 0.7, 1)
                    B_rect.fillColor = (0.8, 0.8, 0.8)

                elif updated_sub_response == 1:

                    B_rect.fillColor = (0.4, 0.7, 1)
                    A_rect.fillColor = (0.8, 0.8, 0.8)

                ### Feedback
                if updated_feed != -1:

                    if updated_feed == 0:
                        img_to_draw = img_incorrect
                    else:
                        img_to_draw = img_correct

                    update_trial_progress(progress)

                    trial_progress_bg.draw()
                    trial_progress_bar.draw()

                    A_rect.draw()
                    A_text.draw()

                    B_rect.draw()
                    B_text.draw()

                    img_to_draw.draw()

                    win0.flip()

                    core.wait(feedback)
                in_warmup = train and block_no == 1
                next_trial_page(in_warmup)

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
                    run_break()
                
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
                    run_break()
                    
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
                        run_break()
                      
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
                    
                    columns = ['ID','GROUP','SESSION','DATE','TIME','TRIAL_BLK','BLK_NO', 
                        'DISTRIBUTION','AMP','LOGICAL_AMP','PHY_BOUND','DISTANCE','TRUE_CAT',
                        'SUB RESPONSE', 'FEEDBACK', 'RT', 'CORRECT_B', 'CORRECT_S',
                        'pDISTRIBUTION','pAMP','pphyBOUND','pDISTANCE','pTRUE_CAT',
                        'pSUB RESPONSE','pFEEDBACK','pRT','TOTAL_BLK', 
                        'TOTAL_TRIAL_INBLK','TRIAL_SSE']
                    final_output = updated_output[columns]

                    final_output.to_csv(pull_file, mode='a', header=False, index=False, encoding='utf-8-sig')
                    print(f"DEBUG: Session {curr_sess} started. P-columns initialized with '---'.")
                    
                else: # do not need to create a new file
                    old_df = pd.read_csv(
                        pull_file,
                        dtype={'ID': str}
                    )
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

                    columns = ['ID','GROUP','SESSION','DATE','TIME','TRIAL_BLK','BLK_NO', 
                        'DISTRIBUTION','AMP','LOGICAL_AMP','PHY_BOUND','DISTANCE','TRUE_CAT',
                        'SUB RESPONSE', 'FEEDBACK', 'RT', 'CORRECT_B', 'CORRECT_S',
                        'pDISTRIBUTION','pAMP','pphyBOUND','pDISTANCE','pTRUE_CAT',
                        'pSUB RESPONSE','pFEEDBACK','pRT','TOTAL_BLK', 
                        'TOTAL_TRIAL_INBLK','TRIAL_SSE']
                    updated_output = updated_output[columns]

                    updated_output.to_csv(pull_file, mode='a', header=False, index=False)
                trial_finished = True

        # Warm-up block finished

        if train and block_no == 1:

            show_instruction_page(
                "End of Warm-up\n\n"
                "Please prepare to begin the main experiment.\n"
                "You can choose to take ONE 5-minute break during the experiment.\n"
                "Ready? Let's begin!",
                "Start"
            )

        if trial_no == curr_block_arr[block_no-1]:
            last_trial= True
        
    if block_no == curr_total_block and last_trial == True:
        finish = True 

    if finish:# INFO table,session_comp + 1
        try:
            info_df = pd.read_csv(
                FILENAME_INFO,
                dtype={'id': str}
            )
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
        
    columns = ['ID','GROUP','SESSION','DATE','TIME','TRIAL_BLK','BLK_NO', 
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