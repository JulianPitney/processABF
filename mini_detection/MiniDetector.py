#%% IMPORT MODULES

from __future__ import division

import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import scipy.signal as signal


#%% CREATE MiniDetector CLASS

class MiniDetector(object):
    """Class for detecting mEPSCs or mIPSCs in Recording objects or similar arrays.
    """

    def __init__(self, rec, clean_data = True, baseline_range = (4500, 5000),
        remove_n_sweeps = 0, I_channel = 0, dt = 0.1):

        self.rec = rec
        self.dt = dt

        if clean_data:
            self.I = self._clean_data(rec, baseline_range, remove_n_sweeps, I_channel)

    def set_name(self, name):
        self.name = name

    @staticmethod
    def _clean_data(rec, baseline_range, remove_n_sweeps, I_channel):
        """
        Clean up Recording object containing minis.

        Inputs:
            baseline_range -- tuple of ints
                Range to use for baseline subtraction in timesteps.
                Discards points before the end of the baseline.
            remove_n_sweeps -- int
                Dicard this may sweeps from the beginning of the recording.
                Set to zero to discard none.
            I_channel -- int
                Current channel of the Recording to clean.

        Returns a matrix with the cleaned I channel.
        """

        baseline_slice = slice(baseline_range[0], baseline_range[1])

        I = rec[I_channel, :, :] - rec[I_channel, baseline_slice, :].mean(axis = 0)
        I_sub = I[baseline_range[1]:, remove_n_sweeps:]

        return I_sub

    @staticmethod
    def _multipoint_gradient(x, no_points, axis = 0):
        """
        Compute a multipoint gradient over a specified axis of a matrix.

        Implemented using convolution with edges padded with zeros.

        Inputs:
            x -- 1D or 2D array
                Matrix or vector over which to compute the gradient.
            no_points -- even int >= 2
                Number of points to use to compute the gradient.
            axis -- int
                Axis over which to compute the gradient.

        Returns:
            Matrix with same shape as x with the first order multipoint gradient.
        """

        if no_points < 2 or no_points % 2 != 0:
            raise ValueError('no_points must be an even integer >= 2.')
        if x.ndim > 2:
            raise ValueError('not defined for x with more than 2 dimensions')

        kernel = -1 / np.concatenate(
            (np.arange(-(no_points - 1), 0, 2), np.arange(1, no_points + 1, 2))
        )
        kernel /= no_points // 2

        if x.ndim > 1:
            gradient = np.empty_like(x)
            slc = [slice(None)] * x.ndim
            for i in range(x.shape[1 - axis]):
                slc[1 - axis] = i
                gradient[slc] = np.convolve(x[slc].flatten(), kernel, 'same')
        else:
            gradient = np.convolve(x.flatten(), kernel, 'same')
            gradient = np.reshape(gradient, x.shape)

        return gradient

    def compute_gradient(self, no_points):
        """
        Compute the multipoint gradient of I for each sweep.
        Creates an attribute named `grad` containing the gradient.
        """
        self.grad = self._multipoint_gradient(self.I, no_points, axis = 0)

    @staticmethod
    def _find_peaks(x, height = None, threshold = None, distance = None,
        prominence = None, width = None, wlen = None, rel_height = 0.5, axis = 0):
        """Vectorized wrapper for scipy.signal.find_peaks.
        x must be a 1D or 2D array.
        Vectorized over axis.
        See scipy.signal.find_peaks docs for other args.
        """

        if x.ndim > 2:
            raise ValueError('x has {} dimensions, but a 1D or 2D array is required.'.format(x.ndim))

        elif x.ndim == 2:

            peaks = []
            properties = []

            slc = [slice(None)] * x.ndim
            for i in range(x.shape[1 - axis]):
                slc[1 - axis] = i
                peaks_tmp, props_tmp = signal.find_peaks(
                    x[slc], height, threshold, distance, prominence, width, wlen,
                    rel_height
                )
                peaks.append(peaks_tmp)
                properties.append(props_tmp)

        else:
            peaks, properties = signal.find_peaks(
                x.flatten(), height, threshold, distance, prominence, width, wlen,
                rel_height
            )

        return peaks, properties


    def find_grad_peaks(self, height_SD = 3.5, distance = 50, width = 5):
        """Find peaks in the gradient.
        Creates attributes named `peak_threshold` (in mV/ms) and `peaks` (ms).
        Ensures that `peaks` is a list of arrays.
        """

        self.peak_threshold = np.float64(self.grad.std()) * height_SD
        peaks, _ = self._find_peaks(
            self.grad, height = self.peak_threshold, distance = distance, width = width
        )

        # Convert peaks to time from indices.
        try:
            # Check whether peaks is array-like.
            # Throws an AttributeError if peaks is not array-like (e.g. a list)
            assert peaks.ndim > 0
            peaks = self.ind_to_t(peaks)
            peaks = [peaks]
        except AttributeError:
            # If peaks is a list of arrays, multiply each array by dt.
            assert peaks[0].ndim > 0 # Check that the first entry is an array.
            peaks = [self.ind_to_t(p) for p in peaks]

        # Check output type.
        assert type(peaks) is list
        assert peaks[0].ndim > 0

        self.peaks = peaks

    @staticmethod
    def _extract_sections(x, list_of_locs, window = (-50, 200), axis = 0, locs_are_timestamps = True, dt = None):
        """Pull sections from `x` based on `list_of_inds`.

        Inputs:
            x -- 2D array
            list_of_inds -- list of arrays
                Indices to around which to pull sections.
                Length of list_of_inds must match the selected axis.
            window -- tuple of ints
                Range around list_of_inds entries around which to pull sections.
            axis -- int
                Axis over which to pull sections.
            locs_are_timestamps -- bool
                Optionally, convert locs from timestamps to inds.
        """

        # Input checks.
        if x.ndim != 2:
            raise TypeError('x must be 2D, but is {} dimensional instead.'.format(x.ndim))
        elif x.shape[1-axis] != len(list_of_locs):
            raise ValueError(
                'shape of x along axis {} ({}) and no. of entries in `list_of_locs`'
                ' ({}) do not match.'.format(
                1-axis, x.shape[1-axis], len(list_of_locs)
            ))
        if locs_are_timestamps and dt is None:
            raise ValueError('If `locs_are_timestamps` is True, `dt` must be specified.')

        # Extract sections.
        sections = []
        slc = [slice(None)] * x.ndim

        for i in range(len(list_of_locs)):

            if locs_are_timestamps:
                inds = (list_of_locs[i] / dt).round().astype(np.int32)
            else:
                inds = list_of_locs[i]

            slc[1-axis] = i

            for ind in inds:
                slc[axis] = slice(ind + window[0], ind + window[1])
                tmp_nan = np.empty(slc[axis].stop - slc[axis].start)
                tmp_nan[:] = np.nan

                tmp_sec = x[slc]
                tmp_nan[:len(tmp_sec)] = tmp_sec

                sections.append(tmp_nan)

        assert len(sections) > 1, 'sections list has length {}'.format(len(sections))

        sections = np.array(sections)
        assert sections.ndim > 1, 'sections has shape {}'.format(sections.shape)
        if axis == 0:
            sections = sections.T

        return sections

    def extract_minis(self, window = (-50, 200)):
        self.minis = self._extract_sections(
            self.I, self.peaks, window = window, axis = 0,
            locs_are_timestamps = True, dt = self.dt
        )

    ### Extract mini parameters
    def inter_mini_intervals(self, sort = False):
        IMIs = []
        for x in self.peaks:
            IMIs.extend(np.diff(x))

        if sort:
            IMIs = np.sort(IMIs)

        return np.array(IMIs)

    def amplitudes(self, baseline_width = None, extract_max = True, sort = False):
        if baseline_width is not None and baseline_width > 1:
            minis_ = (np.copy(self.minis)
                - self.minis[:baseline_width, :].mean(axis = 0))
        else:
            minis_ = np.copy(self.minis)

        if extract_max:
            amplis = np.nanmax(minis_, axis = 0)
        else:
            amplis = np.nanmin(minis_, axis = 0)

        if sort:
            amplis = np.sort(amplis)

        return amplis

    ### Simple properties, support vectors & related methods
    @property
    def no_sweeps(self):
        return self.I.shape[1]

    @property
    def no_channels(self):
        return self.rec.shape[0]

    @property
    def no_minis(self):
        return self.minis.shape[1]

    @property
    def t_vec(self):
        return np.arange(0, (self.I.shape[0] - 0.5) * self.dt, self.dt)

    @property
    def t_mat(self):
        return np.tile(self.t_vec[:, np.newaxis], (1, self.no_sweeps))

    @property
    def t_arr(self):
        return np.tile(self.t_mat[np.newaxis, :, :], (self.no_channels, 1, 1))

    def t_to_ind(self, t):
        return (np.copy(t) / self.dt).round().astype(np.int32)

    def ind_to_t(self, ind):
        return np.copy(ind) * self.dt


    ### Plotting methods

    def plot_rec(self):
        self.rec.plot(downsample = 1)

    def plot_signal(self, sweeps = 'all', show_grad = True, show_grad_peaks = True):

        if sweeps == 'all':
            sweeps = slice(None)
            no_sweeps_plt = self.no_sweeps
        else:

            try:
                assert len(sweeps) == 2
                sweeps = slice(sweeps[0], sweeps[1])
                no_sweeps_plt = sweeps[1] - sweeps[0]
            except TypeError:
                # Catch case when sweeps is an int
                sweeps = sweeps
                no_sweeps_plt = 1

        plt.figure()

        ax = plt.subplot(111)
        plt.plot(self.t_mat[:, sweeps], self.I[:, sweeps], 'k-', alpha = min(1, 3/no_sweeps_plt))

        if show_grad:
            ax2 = ax.twinx()
            ax2.plot(self.t_mat[:, sweeps], self.grad[:, sweeps], 'r-', alpha = min(0.9, 5/no_sweeps_plt))
        if show_grad_peaks:
            ax2.axhline(self.peak_threshold, color = 'k', lw = 0.5, ls = '--', dashes = (5, 5))

            if no_sweeps_plt > 1:
                tmp_grad = np.copy(self.grad[:, sweeps])
                for i, sw in enumerate(self.peaks[sweeps]):
                    ax2.plot(sw, tmp_grad[(sw / self.dt).round().astype(np.int32), i], 'go')

            else:
                ax2.plot(self.peaks[sweeps], self.grad[(self.peaks[sweeps] / self.dt).round().astype(np.int32), sweeps], 'go')



        plt.xlabel('Time (ms)')
        plt.ylabel('I')
        plt.show()

    def plot_minis(self, bl_subtract = None):
        plt.figure()
        plt.subplot(111)
        t_vec = np.arange(0, (self.minis.shape[0] - 0.5) * self.dt, self.dt)
        y = np.copy(self.minis)
        if bl_subtract is not None:
            y -= y[:bl_subtract, :].mean(axis = 0)
        plt.plot(
            np.tile(t_vec[:, np.newaxis], (1, self.no_minis)), y,
            'k-', lw = 0.5, alpha = 0.7
        )
        plt.plot(
            t_vec, np.nanmean(y, axis = 1), 'r-'
        )
        plt.show()

    ### Methods for saving stuff.
    def pickle_detector(self, fname):
        """Pickle the current instance of MiniDector.
        """
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    def pickle_minis(self, fname):
        """Pickle detected minis as a numpy array.
        """
        with open(fname, 'wb') as f:
            pickle.dump(self.minis, f)
