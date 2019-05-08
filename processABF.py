from __future__ import division
from neo import AxonIO
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import *
import os



class ABFProcessor(object):

    def __init__(self):

        self.master = Tk()
        self.master.geometry("640x480")
        self.gen_widgets()
        self.master.mainloop()


    def gen_widgets(self):

        mainFrame = Frame(self.master)
        processFolderButton = Button(mainFrame, state=DISABLED, width=20, height=3, text="Process Folder", font="Verdana 30 bold", borderwidth=6, command=lambda: self.process_folder())
        processFileButton = Button(mainFrame, width=20, height=3, text="Process File", font="Verdana 30 bold", borderwidth=6, command=lambda: self.process_file())
        self.genGraphsState = BooleanVar()
        genGraphsStateButton = Checkbutton(mainFrame, width=40, height=4, text="Generate Graphs", variable=self.genGraphsState)

        processFolderButton.pack(anchor=CENTER)
        processFileButton.pack(anchor=CENTER)
        genGraphsStateButton.pack(anchor=CENTER)
        mainFrame.pack(expand=True)



    def process_file(self):

        filePath = self.select_abf_file()
        if filePath != "":
            traces, dt, nb_steps, nb_sweeps = self.load_abf(filePath)
            self.process_traces(traces, dt, nb_steps, filePath)

    def process_folder(self):
        directoryPath = self.select_directory()
        print(directoryPath)


    def select_directory(self):
        return filedialog.askdirectory()


    def select_abf_file(self):
        return filedialog.askopenfilename(initialdir="./", title="Select file",
                                   filetypes=(("abf files", "*.abf"), ("all files", "*.*")))


    def load_abf(self, ABFPath):

        original_file = AxonIO(filename=ABFPath)
        data = original_file.read_block(lazy=False)
        nb_steps = len(data.segments[0].analogsignals[0])
        nb_sweeps = len(data.segments)
        fs = np.array(data.segments[0].analogsignals[0].sampling_rate, dtype=int)
        dt = 1 / fs
        traces = np.zeros((nb_steps, nb_sweeps))

        for sw_i in range(nb_sweeps):
            traces[:, sw_i] = np.ravel(np.array(data.segments[sw_i].analogsignals[0]))

        # Arrange matrix in sane way
        traces = traces.transpose()

        return traces, dt, nb_steps, nb_sweeps



    def gen_trace_plot(self, xPoints, yPoints, title, savePath):


        plt.figure(figsize=(20, 10))
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Current (pA)')
        plt.plot(xPoints, yPoints, 'b', alpha=0.7)
        plt.savefig(savePath + title + '.svg')
        plt.close()


    def process_traces(self, traces, dt, nb_steps, abfPath):

        startX = int(0 / dt)
        endX = nb_steps - 1
        numSamples = traces.shape[1]
        numSubSamplePoints = numSamples
        numTraces = traces.shape[0]
        timePoints = np.linspace(startX, endX, numSamples) * dt
        traceSampleIndexes = np.linspace(startX, endX, numSubSamplePoints).astype(np.int)
        subSampleTimePoints = np.linspace(startX, endX, numSubSamplePoints) * dt

        for traceIndex in range(0, numTraces):

            trace = traces[traceIndex]
            self.process_trace(trace, timePoints, traceIndex, dt)

            if self.genGraphsState.get():
                outputPath = os.path.dirname(abfPath) + "/"
                self.gen_trace_plot(subSampleTimePoints, trace[traceSampleIndexes], str(traceIndex), outputPath)

        print("Processing complete!")

    def process_trace(self, trace, timePoints, traceIndex, dt):

        gradients = np.gradient(trace)
        maxGradientIndex = np.argmax(gradients)
        minGradientIndex = np.argmin(gradients)

        timeOfMaxGradient = timePoints[maxGradientIndex]
        timeOfMinGradient = timePoints[minGradientIndex]
        delta = timeOfMinGradient - timeOfMaxGradient

        if 0.001 > delta > 0:

            timeStepsUntilBaseline = int((timePoints[maxGradientIndex] - 0.05) / (dt))
            X1_timeBaseline = timePoints[timeStepsUntilBaseline]
            Y1_currentBaseline = trace[timeStepsUntilBaseline]

            timeStepsUntilEventOver = timeStepsUntilBaseline + int(0.3 / dt)
            eventCurrents = trace[minGradientIndex + 100:timeStepsUntilEventOver]
            Y2_eventCurrentMin = np.amin(eventCurrents)
            X2_eventCurrentMinTime = timePoints[np.argmin(eventCurrents) + minGradientIndex - 1]

            peakCurrent = Y2_eventCurrentMin - Y1_currentBaseline
            riseTime = X2_eventCurrentMinTime - X1_timeBaseline

            print("TRACE " + str(traceIndex) + "\n" +
                  "PEAK_CURRENT= " + str(peakCurrent) + "(pA)\n" +
                  "RISE_TIME= " + str(riseTime) + "(s)\n")

        else:
            print("TRACE " + str(traceIndex) + "\n" +
                  "No peak found!\n")




abfProc = ABFProcessor()



