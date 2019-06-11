from __future__ import division
import matplotlib
matplotlib.use("TkAgg") # Use this on Mac
from tkinter import filedialog
from tkinter import *
import synappy as syn
from os import listdir
from os.path import isfile, join
import xlwt
import ntpath


class ABFProcessor(object):

    def __init__(self):


        self.events = []
        self.FIND_STIM_EVENTS = True
        self.FIND_SPONTANEOUS_EVENTS = False

        self.master = Tk()
        self.master.geometry("850x550")
        self.gen_widgets()
        self.master.mainloop()



    def gen_widgets(self):

        mainFrame = Frame(self.master)

        helpMessage = """
        Click the button to select a directory of abf files to analyze. 
        An xls file will be generated where each line represents an event.
        
        Each event will contain:
        
            - FILE_NAME
            - TRACE_NUM
            - STIM_NUM
            - TAO
            - BASELINE_OFFSET
            - NORMALIZED_PEAK_AMPLITUDE
            - PEAK_AMPLITUDE_INDEX
            - PEAK_TIME_FROM_STIM_ON
            - ???
            - LATENCY_SECONDS
            - MEAN_BASELINE
            - STDEV_BASELINE
        
        NOTE1: Each event in the file can be uniquely identified by a combination of FILE_NAME, TRACE_NUM and STIM_NUM.
            This property allows easy implementation of excel macros or python functions for calculating arbitrary 
            metrics from arbitrary bins of events.
            
        NOTE2: The .abf files in the selected directory MUST contain a TTL 0-5V stimulation channel."""

        helpLabel = Label(mainFrame, text=helpMessage, justify=LEFT, relief=GROOVE, padx=5, pady=15)
        helpLabel.config(font=("Courier", 9))
        processFilesButton = Button(mainFrame, width=12, height=2, text="Process ABFs", font="Verdana 30 bold", borderwidth=6, command=self.process_files, bg="RoyalBlue4",
                                    activebackground="RoyalBlue3")
        helpLabel.pack()
        processFilesButton.pack(anchor=CENTER, pady=(10, 0))
        mainFrame.pack(expand=True, side=TOP)


    def select_directory(self):
        return filedialog.askdirectory()

    def path_leaf(self, path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)


    # Takes a synwrapper object and dumps each stimulation event into a .xls
    # Note: This function assumes the synwrapper object has already called <add_events('stim')>
    # and <add_all()>. This function will crash otherwise.
    def dump_synwrapper_to_xls(self, event, fileNames, directoryPath):


        colNames = ['FILE_NAME', 'TRACE_NUM', 'STIM_NUM', 'TAO', 'BASELINE_OFFSET', 'NORMALIZED_PEAK_AMPLITUDE',
                    'PEAK_AMPLITUDE_INDEX', 'PEAK_TIME_FROM_STIM_ON', '???', 'LATENCY_SECONDS','LATENCY_INDEX',
                    'MEAN_BASELINE', 'STDEV_BASELINE']

        # Initialize workbook
        book = xlwt.Workbook(encoding="utf-8")
        sheet1 = book.add_sheet("STIMS")

        for i in range(0, len(colNames)):

            sheet1.write(0, i, colNames[i])
            sheet1.col(i).width = 256 * 32


        stimIndex = 0
        # for each file
        for i in range(0, len(event.decay)):
            # for each trace
            for x in range(0, len(event.decay[i])):
                # for each stim
                for y in range(0, len(event.decay[i][x])):

                    tao = event.decay[i][x][y][0]
                    baselineOffset = event.decay[i][x][y][1]
                    normalizedPeakAmplitude = event.height[i][x][y][0]
                    peakAmplitudeIndex = event.height[i][x][y][1]
                    timeOfPeakAmplitudeFromStim = event.height[i][x][y][2]
                    weDontKnowWhatThisValueIs = event.height[i][x][y][3]
                    latencySeconds = event.latency[i][x][y][0]
                    latencyIndex = event.latency[i][x][y][1]
                    meanBaseline = event.baseline[i][x][y][0]
                    stdevBaseline = event.baseline[i][x][y][1]


                    sheet1.write(stimIndex + 1, 0, str(fileNames[i]))
                    sheet1.write(stimIndex + 1, 1, str(x))
                    sheet1.write(stimIndex + 1, 2, str(y))
                    sheet1.write(stimIndex + 1, 3, str(tao))
                    sheet1.write(stimIndex + 1, 4, str(baselineOffset))
                    sheet1.write(stimIndex + 1, 5, str(normalizedPeakAmplitude))
                    sheet1.write(stimIndex + 1, 6, str(peakAmplitudeIndex))
                    sheet1.write(stimIndex + 1, 7, str(timeOfPeakAmplitudeFromStim))
                    sheet1.write(stimIndex + 1, 8, str(weDontKnowWhatThisValueIs))
                    sheet1.write(stimIndex + 1, 9, str(latencySeconds))
                    sheet1.write(stimIndex + 1, 10, str(latencyIndex))
                    sheet1.write(stimIndex + 1, 11, str(meanBaseline))
                    sheet1.write(stimIndex + 1, 12, str(stdevBaseline))
                    stimIndex += 1

        savePath = directoryPath + "/stims.xls"
        book.save(savePath)


    def process_files(self):

        # Select a directory and get a list of files in that directory
        directoryPath = self.select_directory()
        fileNames = [f for f in listdir(directoryPath) if isfile(join(directoryPath, f))]

        # Filter to get only .abf files and get full path of each .abf
        filePaths = []
        for fileName in fileNames:
            if fileName.endswith('.abf'):
                filePaths.append(directoryPath + "/" + fileName)


        # load .abf files into synwrapper
        event = syn.load(filePaths)

        # Algorithmic magic to identify events. (stim events are found using TTL 0-5V channel (not so magical), while spontaenous events
        # are found using derivatives and magic).
        if self.FIND_STIM_EVENTS:
            syn.add_events(event, event_type='stim', stim_thresh=2)
        if self.FIND_SPONTANEOUS_EVENTS:
            syn.add_events(event, event_type='spontaneous', spont_filtsize=25, spont_threshampli=3, spont_threshderiv=-1.2, savgol_polynomial=3)

        # More algorithmic magic for identifying various features of the events found in previous step.
        event.add_all(event_direction='down', latency_method='max_height')


        # Get filename of all ABF files
        fileNames = []
        for filePath in filePaths:
            fileNames.append(self.path_leaf(filePath))

        self.dump_synwrapper_to_xls(event, fileNames, directoryPath)
        self.events.append(event)




abfProc = ABFProcessor()

