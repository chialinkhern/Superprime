from psychopy import event, visual, core, parallel, prefs, voicekey
import librosa
import pandas as pd
import csv
import random
import os


class SuperPrime:

    def __init__(self):
        # variables in this chunk are for logging purposes- they are changed by self.create_log()
        self.FILE_NAME = ""
        self.EXP_NAME = ""

        # this chunk sets values for visual.textStim()
        self.EVENT_TEXT_HEIGHT = 0.1
        self.EVENT_TEXT_FONT = "Arial"
        self.EVENT_TEXT_COLOR = "white"
        self.INSTRUCTION_TEXT_HEIGHT = 0.06
        self.INSTRUCTION_FONT = "Arial"
        self.INSTRUCTION_TEXT_COLOR = "white"

        # this chunk inherits experiment attributes from conditions.csv
        self.condition_dict = self.load_dict("conditions.csv")
        self.CONDITION = self.condition_dict["trial_events"]
        self.ITEM_LIST = self.condition_dict["items"]
        self.SUBJECT_ID = self.condition_dict["subj_id"]

        # this chunk inherits experiment attributes from config.csv
        self.exp_config_dict = self.load_dict("config.csv")
        self.RAND_BLOCKS = self.exp_config_dict["RAND_BLOCKS"]
        self.RAND_WITHIN_BLOCKS = self.exp_config_dict["RAND_WITHIN_BLOCKS"]
        self.TIME_OUT = float(self.exp_config_dict["TIMEOUT"])/1000  # is in seconds because event.waitKeys() looks for seconds
        self.KEY_LIST = self.exp_config_dict["KEY"].split(" ")
        self.BLOCK_NAMES_LIST = self.exp_config_dict["BLOCK_NAMES"].split()
        self.TASK = self.exp_config_dict["TASK"]
        self.EEG = self.exp_config_dict["EEG"]
        self.screen_refresh_rate = float(self.exp_config_dict["REFRESH_RATE"])

        self.stimuli_df = self.load_df("Stimuli/Item_Lists/" + self.ITEM_LIST + ".csv")
        self.trial_config_list = self.load_list('Events/' + self.CONDITION + '.csv')

        # these attributes keep tabs on the status of the experiment
        self.current_block_num = None
        self.current_trial_num = None
        self.current_event_num = None
        self.corr_resp = None
        self.related = None
        self.item_num = None
        self.key_press = None
        self.reaction_time = None
        self.current_trial_series = None

        # this chunk instantiates the psychopy objects that are needed
        self.window = visual.Window(size=(1920, 1080), color=(-1, -1, -1), fullscr=True)
        event.globalKeys.clear()
        event.globalKeys.add(key='q', modifiers=['ctrl', 'alt'], func=self.quit)
        self.stimulus_text = visual.TextStim(self.window, text=" ",
                                             height=self.EVENT_TEXT_HEIGHT,
                                             pos=(0.0, 0.0),
                                             color=self.EVENT_TEXT_COLOR,
                                             bold=False,
                                             italic=False)
        self.stimulus_text.autoDraw = True
        self.instr_text = visual.TextStim(self.window, text=" ",
                                          height=self.INSTRUCTION_TEXT_HEIGHT,
                                          pos=(0.0, 0.0),
                                          color=self.INSTRUCTION_TEXT_COLOR,
                                          bold=False,
                                          italic=False)
        if self.TASK=="NAMING":
            voicekey.pyo_init(rate=48000, buffersize=5)  # initializing pyo server for audio processing
            prefs.hardware["audioLib"] = ["PTB"]  #for audio hardware pref; PTB best

        # this chunk: 1) sets up port if EEG, 2) creates the subject log and 3) runs the experiment
        self.port = None
        self.detect_eeg()
        self.create_log()
        self.experiment()

    def create_log(self):
        """ This procedure creates a log for the subject's data. """
        header_row = []
        header_row.extend(('ExpName', 'SubjectID', 'Item_List', 'Condition'))
        header_row.extend(('BlockID', 'TrialID'))
        header_row.extend(self.stimuli_df.columns.values.tolist()[0:])
        header_row.extend(("Key_press", "RT"))

        file = os.path.basename(__file__)
        self.EXP_NAME = os.path.splitext(file)[0]

        task_rp_list = self.ITEM_LIST.split("_")
        task = task_rp_list[0]
        rp = task_rp_list[1]
        list_num = task_rp_list[2]


        if self.TASK=="NAMING":  # naming data folder will contain not just csv, but also audio clips
            dir_name = "{}_{}_{}_{}_{}_{}".format(self.TASK, task, self.CONDITION, rp,
                                                   list_num, str(self.SUBJECT_ID))  # task var is actually part of item list. TODO confusing
            os.makedirs("Output/Data/"+dir_name, exist_ok=True)
            self.FILE_NAME = dir_name + "/" + dir_name  # messy, but re-using dirname as filename
        else:
            dir_name = ""
            self.FILE_NAME = self.EXP_NAME + '_' + task + '_' + self.CONDITION + '_' + rp + '_' + list_num + '_' + \
                             str(self.SUBJECT_ID)

        with open("Output/Data/" + self.FILE_NAME + '.csv', 'w', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(header_row)

    def detect_eeg(self):
        if self.EEG == "TRUE":
            self.port = parallel.ParallelPort(address=0x3ff8)

    def display_block(self, block_dataframe):
        """
        This procedure groups trials together to present them as blocks.
        """
        num_trials = len(block_dataframe)
        for i in range(num_trials):
            self.current_trial_num = i+1  # i+1 so that current_trial_num starts at 1

            self.current_trial_series = block_dataframe.iloc[i]
            self.display_trial(self.current_trial_series)

    def display_instructions(self, filepath, name=None):
        """
        Displays instructions line by line, and waits for space bar to continue.
        """
        if name is None:  # generic instructions that are applicable to any condition
            with open(filepath) as f:
                instruction_list = f.readlines()
                for i in range(len(instruction_list)):
                    instruction_text = instruction_list[i]
                    self.instr_text.text = instruction_text.replace(r"\n", "\n")
                    self.instr_text.draw()
                    self.window.flip()
                    event.waitKeys(keyList=["space"])
        else:  # instructions that are specific to condition
            instruction_dict = {}
            f = open(filepath)
            for line in f:
                line = line.strip("\n").split("#")
                instruction_dict[line[0]] = line[1]
            instruction_text = instruction_dict[name]
            self.instr_text.text = instruction_text.replace(r"\n", "\n")
            self.instr_text.draw()
            self.window.flip()
            event.waitKeys(keyList=["space"])

    def display_text(self, num_frames=None, text=None, response=False, eeg_trigger=None):
        """
        This procedure displays text for the allotted number of frames.
        """
        timer = core.Clock()  # to record latencies
        timer.reset()

        # EEG triggers sent here are in accordance to conventions Kara and I have agreed upon (refer to description of
        # self.send_eeg_trigger())
        if text is None:  # waits for the allotted number of frames, but does not display anything.
            for frameN in range(num_frames):
                if frameN == 0:
                    self.stimulus_text.text = " "
                self.window.flip()
        else:  # displays text for the allotted number of frames.
            if response is False:
                if self.EEG == "TRUE":
                    frames_to_display = self.time_to_frames(200)
                    for frameN in range(num_frames):
                        if frameN == 0:
                            self.stimulus_text.text = text
                            self.window.flip()
                            self.send_eeg_trigger(eeg_trigger)
                            continue
                        if frameN <= frames_to_display:
                            self.window.flip()
                        else:
                            self.stimulus_text.text = " "
                            self.window.flip()
                else:
                    for frameN in range(num_frames):
                        if frameN == 0:
                            self.stimulus_text.text = text
                            self.window.flip()
                        else:
                            self.window.flip()
            else:  # displays text, and then waits for keypress. If self.EEG is true, display text for only 200ms,
                # but also wait for keypress
                self.stimulus_text.text = text
                if self.EEG == "TRUE":
                    event.clearEvents("keyboard")
                    record_keypress = True
                    num_frames = self.time_to_frames(200)
                    for frameN in range(num_frames):
                        self.window.flip()
                        if frameN == 0:
                            self.send_eeg_trigger(eeg_trigger)
                        if record_keypress:
                            key_press = event.getKeys(keyList=self.KEY_LIST, timeStamped=False)
                            if ("1" in key_press) or ("num_1" in key_press) or ("num_end" in key_press) or \
                                    ("end" in key_press):
                                self.key_press = 1
                                self.send_eeg_trigger(int(self.key_press))
                                self.reaction_time = round(timer.getTime() * 1000, 4)
                                record_keypress = False
                            elif ("2" in key_press) or ("num_down" in key_press) or ("num_2" in key_press) or \
                                    ("down" in key_press):
                                self.key_press = 2
                                self.send_eeg_trigger(int(self.key_press))
                                self.reaction_time = round(timer.getTime() * 1000, 4)
                                record_keypress = False
                    self.stimulus_text.text = " "
                    for frameN in range(self.time_to_frames(self.TIME_OUT * 1000 - 200)):  # self.TIME_OUT is in
                        # seconds, so converting to ms; also subtracting 200 because TIME_OUT counter starts on
                        # presentation of target
                        self.window.flip()
                        if record_keypress:
                            key_press = event.getKeys(keyList=self.KEY_LIST, timeStamped=False)
                            if ("1" in key_press) or ("num_1" in key_press) or ("num_end" in key_press) or \
                                    ("end" in key_press):
                                self.key_press = 1
                                self.send_eeg_trigger(201)
                                self.reaction_time = round(timer.getTime() * 1000, 4)
                                record_keypress = False
                            elif ("2" in key_press) or ("num_down" in key_press) or ("num_2" in key_press) or \
                                    ("down" in key_press):
                                self.key_press = 2
                                self.send_eeg_trigger(202)
                                self.reaction_time = round(timer.getTime() * 1000, 4)
                                record_keypress = False
                    if record_keypress:
                        self.key_press = "None"
                        self.send_eeg_trigger(203)  # 203 for null response
                        self.reaction_time = round(timer.getTime() * 1000, 4)
                else:
                    self.window.flip()
                    if self.TASK == "NAMING":
                        self.get_audio(text)
                    else:
                        self.get_keypress(timer)

    def display_trial(self, trial_series):
        """
        Each trial is made up of a sequence of events. This procedure groups events.
        For now, this procedure displays text only.
        """
        for i, event_time_pair in enumerate(self.trial_config_list):
            self.current_event_num = i+1  # i+1 so that current_event_num starts at 1
            self.related = trial_series[6]
            self.item_num = trial_series[0]
            self.corr_resp = trial_series[5]

            event_name = event_time_pair[0]
            wait = event_time_pair[1]
            event_content = trial_series[i+1]  # i+1 because you don't want to start with the item_num
            eeg_trigger = ""

            if event_name == "Fixation":
                if int(self.corr_resp) == 1:
                    eeg_trigger = 151
                elif int(self.corr_resp) == 2:
                    eeg_trigger = 150
                self.display_text(num_frames=wait, text=event_content, eeg_trigger=eeg_trigger)
            elif event_name == "Prime":
                if self.TASK == "CONCRETENESS DECISION":
                    if int(self.related) == 0:
                        eeg_trigger = 160
                    if int(self.related) == 1:
                        eeg_trigger = 161
                elif self.TASK == "CATEGORY DECISION":
                    if int(self.related) == 0:
                        eeg_trigger = 170
                    if int(self.related) == 1:
                        eeg_trigger = 171
                self.display_text(num_frames=wait, text=event_content, eeg_trigger=eeg_trigger)
            elif event_name == "Mask":  # no mask in eeg, so no trigger
                self.display_text(num_frames=wait, text=event_content)
            if event_name == "Target":
                eeg_trigger = int(self.item_num)
                self.display_text(text=event_content, response=True, eeg_trigger=eeg_trigger)
            elif event_name == "ITI":
                self.display_text(num_frames=wait)

        self.write_log()

    def experiment(self):
        """
        This procedure puts together the pieces of the experiment into one whole.
        """
        practice_df, block_df_list = self.partition_stimuli()
        # task = self.condition_dict["items"].split("_")[0]  # TODO NEEDS TO BE TASK FROM CONFIG
        task = self.TASK

        # this chunk converts ms to num frames
        for i, event_time_pair in enumerate(self.trial_config_list):
            try:
                event_time_pair[1] = self.time_to_frames(event_time_pair[1])
            except ValueError:
                pass

        if self.EEG == "TRUE":
            # this chunk shows introductory instructions
            self.display_instructions("Stimuli/Instructions/main_instructionsEEG.txt")
            try:
                self.display_instructions("Stimuli/Instructions/task_instructions1EEG.txt", task)
                self.display_instructions("Stimuli/Instructions/task_instructions2EEG.txt", task)
                self.display_instructions("Stimuli/Instructions/task_instructions3.txt", task)
            except KeyError:
                print("No corresponding instructions found.")
                pass

            # this chunk displays the practice block if there is one
            if len(practice_df) > 0:
                self.current_block_num = 0
                self.display_instructions("Stimuli/Instructions/practice_instructions.txt")
                self.display_block(practice_df)
                self.display_instructions("Stimuli/Instructions/start_testEEG.txt")

        elif self.EEG == "FALSE":
            # this chunk shows introductory instructions
            self.display_instructions("Stimuli/Instructions/main_instructions.txt")
            try:
                self.display_instructions("Stimuli/Instructions/task_instructions1.txt", task)
                self.display_instructions("Stimuli/Instructions/task_instructions2.txt", task)
                self.display_instructions("Stimuli/Instructions/task_instructions3.txt", task)
            except KeyError:
                print("No corresponding instructions found.")
                pass

            # this chunk displays the practice block if there is one
            if len(practice_df) > 0:
                self.current_block_num = 0
                self.display_instructions("Stimuli/Instructions/practice_instructions.txt")
                self.display_block(practice_df)
                self.display_instructions("Stimuli/Instructions/start_test.txt")

        # this chunk starts the test blocks
        num_blocks = len(block_df_list)
        for i in range(num_blocks):  # block_num 0 is PRACTICE
            self.current_block_num = i + 1  # for logging and alignment purposes
            block_name = self.BLOCK_NAMES_LIST[1:][self.current_block_num]
            self.display_instructions("Stimuli/Instructions/block_instructions.txt", block_name)
            self.display_block(block_df_list[self.current_block_num])
            if self.current_block_num != num_blocks-1:  # if not last block
                self.display_instructions("Stimuli/Instructions/block_break.txt")

        # participant is done!
        if self.EEG == "TRUE":
            self.display_instructions("Stimuli/Instructions/endEEG.txt")
        elif self.EEG == "FALSE":
            self.display_instructions("Stimuli/Instructions/end.txt")

    def get_audio(self, text):
        mic = voicekey.OnsetVoiceKey(sec=self.TIME_OUT,
                                     more_processing=True,
                                     zero_crossings=True,
                                     file_out="Output/Data/{}/{}.wav".format(self.FILE_NAME.split("/")[0]
                                                                             , text))
        mic.start()
        core.wait(self.TIME_OUT)
        mic.stop()

        # use librosa to detect onset time. Not ideal bc tuned to instruments, but better than voicekey
        # x, sr = librosa.load("Output/Data/{}/{}.wav".format(self.FILE_NAME.split("/")[0], text))
        # # grab first peak as voice onset
        # onset_frame = librosa.onset.onset_detect(y=x, sr=sr, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1)[0]
        # onset_time = librosa.frames_to_time(onset_frame)
        # self.reaction_time = round(onset_time * 1000, 4)
        ## self.reaction_time = round(mic.event_onset*1000, 4)
        self.reaction_time = "???"
        self.key_press = "VOICE"

    def get_keypress(self, timer):  #TODO can instantiate timer inside method, but need to change EEG too
        self.key_press = event.waitKeys(keyList=self.KEY_LIST, maxWait=self.TIME_OUT)
        if self.key_press is None:
            self.key_press = "None"
            self.reaction_time = round(timer.getTime() * 1000, 4)
        else:
            self.key_press = self.key_press[0]
            if self.key_press == "num_1" or "num_end":
                self.key_press = 1
            elif self.key_press == "num_2" or "num_down":
                self.key_press = 2
            self.reaction_time = round(timer.getTime() * 1000, 4)

    def load_df(self, file_path):
        """
        Reads csv files and returns them as pandas dataframes.
        """
        df = pd.read_csv(file_path, header=0, skip_blank_lines=True)
        # df = df.drop(df.columns[0], axis=1)
        return df

    def load_dict(self, config_file):
        """
        Reads csv files (two columns only!) and returns them as dictionaries.
        """
        output = {}
        f = open(config_file)
        for line in f:
            line = (line.strip("\n")).split(",")
            try:
                output[line[0]] = line[1]
            except:
                print('ERROR in' + config_file + 'in Row {}'.format(line))
                sys.exit(2)
        return output

    def load_list(self, file_path):
        """
        Reads csv files and returns them as lists.
        """
        output = []
        f = open(file_path)
        for line in f:
            line = (line.strip("\n")).split(",")
            output.append(line)
        return output

    def partition_stimuli(self):
        """
        This method takes self.stimuli_df and partitions it, returning Practice and Test blocks. Test blocks will be
        appropriately named if relevant. If not, they will just be called TEST.
        """
        num_blocks = int(self.exp_config_dict["BLOCKS"])
        num_trials = int(len(self.stimuli_df))

        block_df_list = []  # a list of dataframes per block-- each block gets one

        if num_blocks > 0:  # num_blocks not only quantifies the number of blocks-- it is also our indicator for if
                            # blocks have special names
            for i in range(num_blocks-1):
                self.BLOCK_NAMES_LIST.append(self.BLOCK_NAMES_LIST[-1])  # gives each block a name to refer to for
                # instruction-showing purposes in the self.experiment() function
            block_list = []
            current_block = 1
            num_practice_trials = 0
            for i in range(num_trials):  # for later assigment of block numbers to self.stimuli_df
                if self.stimuli_df.loc[i, "Block_Name"] == 'PRACTICE':
                    num_practice_trials += 1
                    block_list.append(0)
                    continue
                block_list.append(current_block)
                if current_block < num_blocks:
                    current_block += 1
                else:
                    current_block = 1
            if self.RAND_BLOCKS == "TRUE":
                copy = block_list[num_practice_trials:]
                random.shuffle(copy)
                block_list[num_practice_trials:] = copy
            for i in range(len(self.stimuli_df)):  # assigns each trial to a block in self.stimuli_df for partitioning
                self.stimuli_df.loc[i, "Block_Num"] = block_list[i]

            practice_df = self.stimuli_df[self.stimuli_df["Block_Num"] == 0]
            practice_df = practice_df.reset_index(drop=True)
            for i in range(num_blocks):
                block_df = self.stimuli_df[self.stimuli_df["Block_Num"] == i+1]
                if self.RAND_WITHIN_BLOCKS == "TRUE":  # randomizes trials within blocks if executed
                    block_df = block_df.sample(frac=1).reset_index(drop=True)
                    practice_df = practice_df.sample(frac=1).reset_index(drop=True)
                block_df_list.append(block_df)

            return practice_df, block_df_list

        else:  # <=0 indicates that blocks have special names. This is important for instruction showing purposes, etc.
            if self.RAND_BLOCKS == "TRUE":
                copy = self.BLOCK_NAMES_LIST[1:]
                random.shuffle(copy)
                self.BLOCK_NAMES_LIST[1:] = copy
            for i in range(num_trials):
                block_name = self.stimuli_df.loc[i, "Block_Name"]
                self.stimuli_df.loc[i, "Block_Num"] = self.BLOCK_NAMES_LIST.index(block_name)
            practice_df = self.stimuli_df[self.stimuli_df["Block_Num"] == 0]
            practice_df = practice_df.reset_index(drop=True)

            for i in range(1, len(self.BLOCK_NAMES_LIST)):  # assigns each trial to a block in self.stimuli_df for partitioning
                block_df = self.stimuli_df[self.stimuli_df["Block_Num"] == i]
                if self.RAND_WITHIN_BLOCKS == "TRUE":
                    block_df = block_df.sample(frac=1).reset_index(drop=True)
                    practice_df = practice_df.sample(frac=1).reset_index(drop=True)
                block_df_list.append(block_df)

            return practice_df, block_df_list

    def quit(self):  # quits the experiment
        core.quit()

    def send_eeg_trigger(self, trigger):
        """
        Sends meaningful codes to the EEG-Computer for data analysis. Kara and I have agreed on the following.

        Fixation:       Yes (151) | No (150)
        Prime:          CONCRETENESS DECISION   Related (161) | Unrelated (160)
                        CATEGORY DECISION       Related (171) | Unrelated (170)
        Target:         Item-code for target (1~128)
        Response:       1 (201) | 2 (202) | NULL (203)
        """
        if self.EEG != "TRUE":
            return
        else:
            if trigger is None:
                return
            self.port.setData(trigger)
            core.wait(0.0005)
            self.port.setData(0)

    def time_to_frames(self, num_milliseconds):
        """
        Approximates number of frames from time in milliseconds.
        """
        num_milliseconds = float(num_milliseconds)
        ms_per_frame = 1000.0/self.screen_refresh_rate
        num_frames_to_wait = round(num_milliseconds/ms_per_frame)  # round() returns an int that is closest

        return num_frames_to_wait

    def write_log(self):
        row = []
        row.extend((self.EXP_NAME, self.SUBJECT_ID, self.ITEM_LIST, self.CONDITION))
        row.extend((self.current_block_num, self.current_trial_num))
        row.extend(self.current_trial_series[0:-1])
        row.extend((self.key_press, self.reaction_time))

        with open('Output/Data/' + self.FILE_NAME + '.csv', 'a', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(row)
