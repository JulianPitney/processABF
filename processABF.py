from __future__ import division
from neo import AxonIO
import numpy as np
import matplotlib
matplotlib.use("TkAgg") # Use this on Mac
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import *
import os
import xlwt


GUI_ENABLE = False

class ABFProcessor(object):

    def __init__(self):

        if GUI_ENABLE:
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



    def gen_trace_plot(self, xPoints, yPoints, traceNumber, savePath, baselineIndex, baselineEndIndex, peakIndex, eventCurrentBaselineIndex, maxGradientIndex, minGradientIndex):

        fig, ax = plt.subplots(figsize=(40, 10))  # note we must use plt.subplots, not plt.subplot
        plt.text(xPoints[baselineIndex], yPoints[baselineIndex], "BCS", fontsize=10)
        plt.text(xPoints[baselineEndIndex], yPoints[baselineEndIndex], "BCE", fontsize=10)
        plt.text(xPoints[peakIndex], yPoints[peakIndex], "P", fontsize=10)
        plt.text(xPoints[eventCurrentBaselineIndex], yPoints[eventCurrentBaselineIndex], "BT")
        plt.text(xPoints[maxGradientIndex], yPoints[maxGradientIndex], "MAG", fontsize=10)
        plt.text(xPoints[minGradientIndex], yPoints[minGradientIndex], "MIG", fontsize=10)

        # BCS = Start of baseline chunk used for average
        # BCE = End of baseline chunk used for average
        # P = Peak
        # BT = Time point used for baseline
        # MAG = max gradient point
        # MIG = min gradient point

        ax.plot(xPoints, yPoints, 'b', alpha=0.7)
        fig.savefig(savePath + str(traceNumber) + '.svg')
        plt.close(fig)

    def process_traces(self, traces, dt, nb_steps, abfPath):

        # Gen excel workbook for current file
        book = xlwt.Workbook(encoding="utf-8")
        sheet1 = book.add_sheet("TRACES")
        sheet1.write(0, 0, "TRACE_NUMBER")
        sheet1.write(0, 1, "PEAK_CURRENT(pA)")
        sheet1.write(0, 2, "RISE_TIME(s)")
        sheet1.write(0, 3, "DECAY")
        sheet1.col(0).width = 256 * 20
        sheet1.col(1).width = 256 * 20
        sheet1.col(2).width = 256 * 20
        sheet1.col(3).width = 256 * 20
        savePath = os.path.splitext(abfPath)[0]

        startX = int(0 / dt)
        endX = nb_steps - 1
        numSamples = traces.shape[1]
        numSubSamplePoints = numSamples
        numTraces = traces.shape[0]
        timePoints = np.linspace(startX, endX, numSamples) * dt
        traceSampleIndexes = np.linspace(startX, endX, numSubSamplePoints).astype(np.int)
        subSampleTimePoints = np.linspace(startX, endX, numSubSamplePoints) * dt

        for traceIndex in range(0, numTraces):

            print("TRACE" + str(traceIndex))
            trace = traces[traceIndex]
            peakCurrent, riseTime, decay, baselineIndex, baselineEndIndex, peakIndex, eventCurrentBaselineIndex, maxGradientIndex, minGradientIndex = self.process_trace(trace, timePoints, dt)

            if peakCurrent == None or riseTime == None or decay == None:
                sheet1.write(traceIndex + 1, 0, "NOT_FOUND")
                sheet1.write(traceIndex + 1, 1, "NOT_FOUND")
                sheet1.write(traceIndex + 1, 2, "NOT_FOUND")
                sheet1.write(traceIndex + 1, 3, "NOT_FOUND")
            else:
                sheet1.write(traceIndex + 1, 0, traceIndex + 1)
                sheet1.write(traceIndex + 1, 1, peakCurrent)
                sheet1.write(traceIndex + 1, 2, riseTime)
                sheet1.write(traceIndex + 1, 3, decay)

            if self.genGraphsState.get():
                self.gen_trace_plot(subSampleTimePoints, trace[traceSampleIndexes], traceIndex + 1, savePath, baselineIndex, baselineEndIndex, peakIndex, eventCurrentBaselineIndex, maxGradientIndex, minGradientIndex)

        book.save(savePath + ".xls")
        print("Processing complete!")

    def check_none(obj):

        if obj == None:
            print("Error: Object was empty")
            return True
        else:
            return False

    def process_trace(self, trace, timePoints, dt):

        if check_none(trace):
            return -1


        # Find the points of max and min rate of change.
        # This should reliably find the spike generated
        # by the electrode at the start of stimulation.
        gradients = np.gradient(trace)
        if check_none(gradients):
            return -1

        maxGradientIndex = np.argmax(gradients)
        minGradientIndex = np.argmin(gradients)
        timeOfMaxGradient = timePoints[maxGradientIndex]
        timeOfMinGradient = timePoints[minGradientIndex]
        # Check that the distance in time between the max and min rate of change
        # points if less than a small amount. The spike generated by the electrode 
        # should be very brief.
        delta = timeOfMinGradient - timeOfMaxGradient

        if 0.001 > delta > 0:

            # Take a 100ms chunk of plot starting 0.12s before the spike and 
            # ending 0.02s before the spike. We will average this chunk and call it
            # the baseline current.
            baselineStartIndex = int((timePoints[maxGradientIndex] - 0.12) / (dt))
            baselineStopIndex = int((timePoints[maxGradientIndex] - 0.02) /(dt))
            Y1_currentBaseline = np.average(trace[baselineStartIndex:baselineStopIndex])
           
            # Assume the end of the event is no more than 300ms after the end of the electrode spike.
            eventEndIndex = minGradientIndex + int(0.3 / dt)
            # Get a chunk of the plot starting just after the end of the electrode spike and ending at the end
            # of the event.
            eventCurrents = trace[minGradientIndex + 50:eventEndIndex]
            # The minimum of this chunk is the event peak.
            Y2_eventCurrentMin = np.amin(eventCurrents)
            eventCurrentMinIndex = np.argmin(eventCurrents) + minGradientIndex + 50

            # Get the time where the event peak occured.
            X2_eventCurrentMinTime = timePoints[np.argmin(eventCurrents) + minGradientIndex + 50]

            # We now have a better constraint on where the event occurred since we identified the peak
            # earlier. Update the eventCurrents accordingly.
            eventCurrents = trace[minGradientIndex + 50:eventCurrentMinIndex]

            # Find the point in time where the event began. We search the newly updated event
            # currents for the max, which represents the max current just before the current begins to dip.
            X1_eventCurrentMaxIndex = np.argmax(eventCurrents) + minGradientIndex + 50
            X1_timeBaseline = timePoints[X1_eventCurrentMaxIndex]


            # Calculate values of interest
            peakCurrent = abs(Y2_eventCurrentMin - Y1_currentBaseline)
            riseTime = X2_eventCurrentMinTime - X1_timeBaseline
            decay = 0

            return peakCurrent, riseTime, decay, baselineStartIndex, baselineStopIndex, eventCurrentMinIndex, X1_eventCurrentMaxIndex, maxGradientIndex, minGradientIndex
        else:
            # No spike found 
            return None, None, None, 0, 0, 0, 0, 0, 0




abfProc = ABFProcessor()
traces, dt, nb_steps, nb_sweeps = abfProc.load_abf("C:\\Projects\\processABF\\Kieran_Data\\19507004.abf")
startX = int(0 / dt)
endX = nb_steps - 1
numSamples = traces.shape[1]
timePoints = np.linspace(startX, endX, numSamples) * dt

num = 0
for trace in traces:

    abfProc.gen_trace_plot(timePoints, trace, num, ".", 0, 0, 0)
    num += 1


