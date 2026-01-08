"""Fichier contenant toutes les fonctions utilitaire GPU/CPU"""

import os
# import datetime
import math

import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.signal
import scipy.interpolate

from numba import cuda, njit, prange

os.environ['NUMBA_CUDA_MAX_PENDING_DEALLOCS_COUNT'] = '15'


# ======================================================================================================================
# FUNCTIONS
# ======================================================================================================================

# === Function for data loading ===

def read_linear_transducer_data_standard(folder_path):
    '''
    Simple function to read data from a linear transducer in standard format.

    :param folder_path: path to the folder
    :return: data, dictionary with metadata
    '''

    data_path = os.path.join(folder_path, 'data.npy')
    data_raw = np.load(data_path) # RF data (numpy) in format (N_Tx x N_Rx x N_t),
    # with N_Tx the number of steering angles, N_Rx the number of elements and N_t the number of time samples

    metadata_path = os.path.join(folder_path, 'metadata.h5')
    dict_out = {}
    with h5py.File(metadata_path, 'r') as metadata_file:
        dict_out['angles'] = np.float64(np.array(metadata_file['angles']))  # Steering angles [rad]
        dict_out['t_coord'] = np.float64(metadata_file['t_coord'])  # time coordinates [s] (t=0 is defined as when the
        # center of the pulse is emitted from the point (x=0, z=0) situated in the center of the transducer)
        dict_out['c0'] = np.float64(metadata_file['c0'])  # beamforming SoS [m/s] (used to define the steering delays)
        dict_out['f0'] = np.float64(metadata_file['f0'])  # center frequency of the probe [Hz]
        dict_out['pitch'] = np.float64(metadata_file['pitch'])  # pitch of the probe [m]
        dict_out['n_elem'] = np.int32(metadata_file['n_elem'])  # number of elements of the probe
        dict_out['elem_width'] = np.float64(metadata_file['elem_width'])  # width of an element [m]
        dict_out['apod_tx'] = np.float64(metadata_file['apod_tx'])  # apodization weights in transmit
        dict_out['apod_rx'] = np.float64(metadata_file['apod_rx'])  # apodization weights in receive

    return data_raw, dict_out

# === Functions for preprocessing ===


def compute_hilbert_fir(data, filter_size=21, dtype=np.complex128):
    '''
    Get the analytic signal using a FIR Hilbert transform

    :param data: the input data (Hilbert transform performed on the last dimension)
    :param filter_size: size of the filter used
    :param dtype: the dtype of the output data
    :return: the analytic signal of data
    '''

    # Get the filter
    beta = 8

    fc = 1
    t = fc / 2 * np.arange((1 - filter_size) / 2, filter_size / 2)
    fir_hilbert = np.sinc(t) * np.exp(1j * np.pi * t)

    # Check input length
    sig_shape = data.shape
    sig_len = sig_shape[-1]  # only axis=-1 supported!
    if sig_len < filter_size:
        raise ValueError('Signal length must be larger than filter length')

    if filter_size % 2 == 0:
        raise ValueError('Must be odd')

    # Multiply ideal filter with tapered window
    fir_hilbert *= scipy.signal.windows.kaiser(filter_size, beta)
    fir_hilbert /= np.sum(fir_hilbert.real)

    sig_ana = np.zeros(shape=sig_shape, dtype=dtype)
    #   Reshape array to iterate (generalize to any dim)
    x_it = data.reshape(-1, sig_shape[-1])
    sa_it = sig_ana.reshape(-1, sig_shape[-1])
    for s, sa in zip(x_it, sa_it):
        sa[:] = np.convolve(s, fir_hilbert, 'same')

    return sig_ana


def apply_exponential_tgc(data, t_coord, att_db_s):
    '''
    Apply an exponential TGC (Time-Gain Compensation) on the data without an offset

    :param data: data (..., N_t)
    :param t_coord: time coordinates (in s, N_t)
    :param att_db_s: attenuation value (in dB/s)
    :return: the modified data
    '''

    val_acc = 10**(att_db_s * t_coord / 20)

    return data * val_acc

###########
### GPU ###
###########


# === Functions for the delay-and-sum ===

def get_beamformer_npw_linear_transducer_Tukey_phase_screen(
        angles_tx, sensors_x, t_coord, t_upsampling, x_coord_beam, z_coord_beam, c0,
        apod_tukey_angle_max, apod_tukey_cosine_frac,
        return_one_per_angle=False, thread_number=256, pos_coord_batch=5):
    '''
    Function to get the basic delay-and-sum beamformer

    :param angles_tx: steering angles
    :param sensors_x: x positions of sensors [m]
    :param t_coord: time coordinates of the data [s]
    :param t_upsampling: upsampling parameter for the interpolation
    :param x_coord_beam: x coordinates of the beamforming grid [m]
    :param z_coord_beam: z coordinates of the beamforming grid [m]
    :param c0: beamforming SoS [m/s]
    :param apod_tukey_angle_max: maximum angle for the Tukey apodization [rad]
    :param apod_tukey_cosine_frac: cosine fraction for the Tukey apodization
    :param return_one_per_angle: if we do not want to perform compounding
    :param thread_number: number of threads
    :param pos_coord_batch: batch size
    :return:
    '''

    n_channel = 1

    # define the parameters
    delta_t_up = (t_coord[1] - t_coord[0]) / t_upsampling
    n_t_up = (t_coord.shape[0] - 1) * t_upsampling + 1
    t0 = t_coord[0]

    n_elem = sensors_x.shape[0]

    n_loop_1 = np.int32(np.ceil((pos_coord_batch * n_elem) / thread_number))
    n_loop_2 = np.int32(np.ceil((pos_coord_batch * n_channel) / thread_number))

    n_x = x_coord_beam.shape[0]
    n_z = z_coord_beam.shape[0]

    delta_x = x_coord_beam[1] - x_coord_beam[0]
    delta_z = z_coord_beam[1] - z_coord_beam[0]

    x0 = x_coord_beam[0]
    z0 = z_coord_beam[0]
    block_number_beamformer = np.int32(np.ceil((n_x * n_z) / pos_coord_batch))

    # define the operator
    @cuda.jit(max_registers=4)
    def cuda_operator(data_gpu_in, img_gpu_out, angle_tx):

        # allocate the shared arrays
        t_id_down_all = cuda.shared.array(shape=(pos_coord_batch, n_elem), dtype=np.int32)
        t_id_up_all = cuda.shared.array(shape=(pos_coord_batch, n_elem), dtype=np.int32)
        t_interp_down_all = cuda.shared.array(shape=(pos_coord_batch, n_elem), dtype=np.float64)
        t_interp_up_all = cuda.shared.array(shape=(pos_coord_batch, n_elem), dtype=np.float64)
        o_tx_rx_all = cuda.shared.array(shape=(pos_coord_batch, n_elem), dtype=np.float64)

        # get the ids
        block_id = cuda.blockIdx.x
        thread_id = cuda.threadIdx.x

        # first loop: compute the interpolation coeficients
        for loop_id in range(n_loop_1):

            # get the ids and test
            full_id = loop_id * thread_number + thread_id

            elem_id = full_id % n_elem
            pos_batch_id = full_id // n_elem

            if pos_batch_id >= pos_coord_batch:
                break

            pos_id_full = block_id * pos_coord_batch + pos_batch_id
            z_id = pos_id_full % n_z
            x_id = pos_id_full // n_z

            if x_id >= n_x:
                break

            # get the t interpolation
            x_local = x_id * delta_x + x0
            z_local = z_id * delta_z + z0
            elem_local = sensors_x[elem_id]

            d_tx = z_local * math.cos(angle_tx) + x_local * math.sin(angle_tx)
            d_rx = math.sqrt((x_local - elem_local) ** 2 + z_local ** 2)

            t_local = (d_tx + d_rx) / c0
            t_local_norm = (t_local - t0) / delta_t_up
            t_id_down = np.int32(math.floor(t_local_norm))
            t_id_up = t_id_down + 1
            t_interp_up = t_local_norm - t_id_down
            t_interp_down = 1 - t_interp_up

            t_id_down_all[pos_batch_id, elem_id] = t_id_down
            t_id_up_all[pos_batch_id, elem_id] = t_id_up
            t_interp_down_all[pos_batch_id, elem_id] = t_interp_down
            t_interp_up_all[pos_batch_id, elem_id] = t_interp_up

            theta_angle = math.atan2(x_coord_beam[x_id] - sensors_x[elem_id], z_coord_beam[z_id])

            val = (math.pi / apod_tukey_cosine_frac) * (1 - abs(theta_angle) / apod_tukey_angle_max)
            val = min(max(val, 0), np.pi)
            o_local = (1 - math.cos(val)) / 2
            o_tx_rx_all[pos_batch_id, elem_id] = o_local

        cuda.syncthreads()

        # second loop: berform the adjoint
        for loop_id in range(n_loop_2):
            full_id = loop_id * thread_number + thread_id

            pos_batch_id = full_id // n_channel
            channel_id = full_id % n_channel

            if pos_batch_id >= pos_coord_batch:
                break

            pos_id_full = block_id * pos_coord_batch + pos_batch_id
            z_id = pos_id_full % n_z
            x_id = pos_id_full // n_z

            if x_id >= n_x:
                break

            val_local = 0j
            for elem_id in range(n_elem):

                t_id_down_local = t_id_down_all[pos_batch_id, elem_id]
                t_id_up_local = t_id_up_all[pos_batch_id, elem_id]
                t_interp_down_local = t_interp_down_all[pos_batch_id, elem_id]
                t_interp_up_local = t_interp_up_all[pos_batch_id, elem_id]

                o_val_local = o_tx_rx_all[pos_batch_id, elem_id]

                val_local_small = 0j
                if t_id_down_local >= 0 and t_id_down_local < n_t_up:
                    val_local_small += (t_interp_down_local * data_gpu_in[channel_id, elem_id, t_id_down_local])
                if t_id_up_local >= 0 and t_id_up_local < n_t_up:
                    val_local_small += (t_interp_up_local * data_gpu_in[channel_id, elem_id, t_id_up_local])

                val_local += val_local_small * o_val_local

            img_gpu_out[channel_id, x_id, z_id] = val_local

    # define the function
    def beamform(data):

        if return_one_per_angle:
            img_out = np.zeros((angles_tx.shape[0], x_coord_beam.shape[0], z_coord_beam.shape[0]), dtype=np.complex128)
        else:
            img_out = np.zeros((x_coord_beam.shape[0], z_coord_beam.shape[0]), dtype=np.complex128)

        img_gpu_out = cuda.to_device(np.zeros((1, x_coord_beam.shape[0], z_coord_beam.shape[0]), dtype=np.complex128))

        for tx_id, angle_tx in enumerate(angles_tx):

            data_local = data[tx_id, :, :]

            data_local_interp_real = scipy.interpolate.interp1d(t_coord, np.real(data_local), kind='cubic', fill_value="extrapolate")
            data_local_interp_imag = scipy.interpolate.interp1d(t_coord, np.imag(data_local), kind='cubic', fill_value="extrapolate")
            data_local_up = data_local_interp_real(delta_t_up * np.arange(n_t_up) + t0) +\
                            1j * data_local_interp_imag(delta_t_up * np.arange(n_t_up) + t0)

            data_local_gpu = cuda.to_device(data_local_up[np.newaxis, :, :])

            cuda_operator[block_number_beamformer, thread_number](data_local_gpu, img_gpu_out, angle_tx)

            if return_one_per_angle:
                img_out[tx_id, :, :] = img_gpu_out.copy_to_host()[0, :, :]
            else:
                img_out += img_gpu_out.copy_to_host()[0, :, :]

        return img_out

    return beamform

# === Functions for the adaptive beamforming ===


def get_select_patch_window_cuda_function(angles_tx, z_coord_small, z_size_patch, x_patch_pos, z_patch_pos,
                                          max_registers=2):
    '''
    Get the GPU function to select patches from the image
    :param angles_tx: Tx angles
    :param z_coord_small: patch coordinates
    :param x_size_patch_cuda: number of patches in the x directions
    :param z_size_patch: number of patches in the z direction
    :param window_val: float32 window of size N_z_small x N_z_small
    :param x_patch_pos: x id of window center in the image, size N_x
    :param z_patch_pos: x id of window center in the image, size N_z
    :return: the function
    '''

    n_tx = angles_tx.shape[0]
    n_z_small = z_coord_small.shape[0]
    window_radius_pixels = (n_z_small + 1)//2

    x_patch_pos = np.ascontiguousarray(x_patch_pos)
    z_patch_pos = np.ascontiguousarray(z_patch_pos)

    @cuda.jit(max_registers=max_registers)
    def select_patch_window_cuda(patch_out, img_in, window_val_in, x_size_patch_cuda, x_patch_offset_input, x_patch_offset_output):
        '''
        Extract patches from images and window them

        :param patch_out: complex64 GPU patches of size N_x x N_z x N_tx x N_z_small x N_z_small
        :param img_in: complex64 GPU image of size N_x_img x N_z_img
        :param window_in: float32 window of size N_z_small x N_z_small
        :param x_patch_pos_in: x id of window center in the image, size N_x
        :param z_patch_pos_in: x id of window center in the image, size N_z
        :param x_patch_offset_input: x offset of the patch in the image
        :param x_patch_offset_output: x offset in the output patch array
        :return:
        '''

        grid_id = cuda.grid(1)

        x_id = grid_id // (z_size_patch * n_tx)
        z_id = (grid_id % (z_size_patch * n_tx)) // n_tx
        tx_id = (grid_id % (z_size_patch * n_tx)) % n_tx

        if x_id >= x_size_patch_cuda:
            return

        x_beam_id_local = x_patch_pos[x_id + x_patch_offset_input]
        z_beam_id_local = z_patch_pos[z_id]

        start_id_x_local = (x_beam_id_local - (window_radius_pixels - 1))
        start_id_z_local = (z_beam_id_local - (window_radius_pixels - 1))

        for x_small_id in range(n_z_small):
            for z_small_id in range(n_z_small):
                patch_val_local = img_in[tx_id, start_id_x_local + x_small_id, start_id_z_local + z_small_id]
                patch_val_local *= window_val_in[x_small_id, z_small_id]
                patch_out[x_patch_offset_output + x_id, z_id, tx_id, x_small_id, z_small_id] = patch_val_local

    return select_patch_window_cuda


def get_patch_radon_transform_rx_cuda_function(angles_tx, z_coord_small, angles_rx, k0, z_size_patch,
                                                thread_number=1024, max_registers=4, M_first_sum=12):
    '''
    Get the GPU function to compute the Radon tranform of a series of patches

    :param angles_tx: Tx angles
    :param z_coord_small: coord of the patch
    :param angles_rx: Rx angles
    :param k0: base spatial frequency
    :param z_size_patch: number of patches in the z direction
    :param thread_number: number of threads
    :param max_registers: number of registers
    :return: the function
    '''


    n_tx = angles_tx.shape[0]
    n_z_small = z_coord_small.shape[0]
    n_rx = angles_rx.shape[0]

    z0 = z_coord_small[0]
    delta_z = z_coord_small[1] - z_coord_small[0]

    loop_number_tx_z_small = np.int32(np.ceil(n_tx * n_z_small / thread_number))

    angles_tx = np.ascontiguousarray(angles_tx)

    l_val = 6
    z_p = np.sqrt(3) - 2

    @cuda.jit(max_registers=max_registers)
    def patch_radon_transform_rx_cuda(win_rad_filt_out, win_rad_local, patches_in, mat_filt_in, x_patch_offset_input,
                                      x_patch_offset_output):
        '''
        Function to compute the Radon transform of patches

        :param win_rad_filt_out: complex64 gpu array of size N_x x N_z x N_tx x N_rx x N_z_small-1
        :param win_rad_local: complex64 gpu array of size N_x x N_z x N_tx x N_rx x N_z_small
        :param patches_in: complex64 gpu array of size N_x x N_z x N_tx x N_z_small x N_z_small
        :param mat_filt_in: complex64 matrix for the filtering
        :param x_patch_offset_input: offset of the x patch in the input
        :param x_patch_offset_output: offset of the x patch in the output
        :return:
        '''

        # determine the patch id
        block_id = cuda.blockIdx.x
        patch_id_x = block_id // z_size_patch
        patch_id_z = block_id % z_size_patch

        thread_id = cuda.threadIdx.x

        # perform the prefilter for interpolation
        for loop_id in range(loop_number_tx_z_small):
            full_id = loop_id * thread_number + thread_id
            tx_id = full_id // n_z_small
            x_small_id = full_id % n_z_small

            if tx_id >= n_tx:
                break

            z_p_power = 1.
            val_sum = 0j
            for m_id in range(M_first_sum):
                z_p_power *= z_p
                val_sum += patches_in[
                               x_patch_offset_input + patch_id_x, patch_id_z, tx_id, x_small_id, m_id] * z_p_power

            # forward pass
            for z_id in range(n_z_small):
                if z_id == 0:
                    patches_in[x_patch_offset_input + patch_id_x, patch_id_z, tx_id, x_small_id, z_id] = \
                        l_val * (patches_in[
                                     x_patch_offset_input + patch_id_x, patch_id_z, tx_id, x_small_id, z_id] + val_sum)
                else:
                    patches_in[x_patch_offset_input + patch_id_x, patch_id_z, tx_id, x_small_id, z_id] = \
                        l_val * patches_in[x_patch_offset_input + patch_id_x, patch_id_z, tx_id, x_small_id, z_id] + \
                        z_p * patches_in[x_patch_offset_input + patch_id_x, patch_id_z, tx_id, x_small_id, z_id - 1]

            # backward pass
            for z_id in range(n_z_small):
                z_id_local = n_z_small - 1 - z_id
                if z_id_local == n_z_small - 1:
                    patches_in[x_patch_offset_input + patch_id_x, patch_id_z, tx_id, x_small_id, z_id_local] = \
                        - z_p / (1 - z_p) * patches_in[
                            x_patch_offset_input + patch_id_x, patch_id_z, tx_id, x_small_id, z_id_local]
                else:
                    patches_in[x_patch_offset_input + patch_id_x, patch_id_z, tx_id, x_small_id, z_id_local] = \
                        z_p * (patches_in[
                                   x_patch_offset_input + patch_id_x, patch_id_z, tx_id, x_small_id, z_id_local + 1] -
                               patches_in[x_patch_offset_input + patch_id_x, patch_id_z, tx_id, x_small_id, z_id_local])
        cuda.syncthreads()


        for loop_id in range(loop_number_tx_z_small):
            full_id = loop_id * thread_number + thread_id
            tx_id = full_id // n_z_small
            z_small_id = full_id % n_z_small

            if tx_id >= n_tx:
                break

            z_p_power = 1.
            val_sum = 0j
            for m_id in range(M_first_sum):
                z_p_power *= z_p
                val_sum += patches_in[x_patch_offset_input + patch_id_x, patch_id_z, tx_id, m_id, z_small_id] *\
                           z_p_power

            # forward pass
            for x_id in range(n_z_small):
                if x_id == 0:
                    patches_in[x_patch_offset_input + patch_id_x, patch_id_z, tx_id, x_id, z_small_id] = \
                        l_val * (patches_in[
                                     x_patch_offset_input + patch_id_x, patch_id_z, tx_id, x_id, z_small_id] + val_sum)
                else:
                    patches_in[x_patch_offset_input + patch_id_x, patch_id_z, tx_id, x_id, z_small_id] = \
                        l_val * patches_in[x_patch_offset_input + patch_id_x, patch_id_z, tx_id, x_id, z_small_id] + \
                        z_p * patches_in[x_patch_offset_input + patch_id_x, patch_id_z, tx_id, x_id - 1, z_small_id]

            # backward pass
            for x_id in range(n_z_small):
                x_id_local = n_z_small - 1 - x_id
                if x_id_local == n_z_small - 1:
                    patches_in[x_patch_offset_input + patch_id_x, patch_id_z, tx_id, x_id_local, z_small_id] = \
                        - z_p / (1 - z_p) * patches_in[
                            x_patch_offset_input + patch_id_x, patch_id_z, tx_id, x_id_local, z_small_id]
                else:
                    patches_in[x_patch_offset_input + patch_id_x, patch_id_z, tx_id, x_id_local, z_small_id] = \
                        z_p * (patches_in[
                                   x_patch_offset_input + patch_id_x, patch_id_z, tx_id, x_id_local + 1, z_small_id] -
                               patches_in[x_patch_offset_input + patch_id_x, patch_id_z, tx_id, x_id_local, z_small_id])
        cuda.syncthreads()

        for loop_id in range(loop_number_tx_z_small):
            full_id = loop_id * thread_number + thread_id
            tx_id = full_id // n_z_small
            z_small_id = full_id % n_z_small

            if tx_id >= n_tx:
                break

            for rx_id in range(n_rx):
                angle_mid_local = (angles_rx[rx_id] + angles_tx[tx_id]) / 2
                cos_mid_local = math.cos(angle_mid_local)
                # tan_mid_local = math.tan(angle_mid_local)
                sin_mid_local = math.sin(angle_mid_local)

                val_out = 0j
                for x_small_id in range(n_z_small):

                    x_val_rot = z_coord_small[z_small_id] * sin_mid_local + z_coord_small[x_small_id] * cos_mid_local
                    z_val_rot = z_coord_small[z_small_id] * cos_mid_local - z_coord_small[x_small_id] * sin_mid_local

                    x_val_rot_norm = (x_val_rot - z0) / delta_z
                    z_val_rot_norm = (z_val_rot - z0) / delta_z

                    x_id_down = np.int32(math.floor(x_val_rot_norm))
                    z_id_down = np.int32(math.floor(z_val_rot_norm))


                    interp_x_down_down = math.pow(2 - math.fabs(x_val_rot_norm - (x_id_down - 1)), 3) / 6
                    interp_x_down = (2 / 3 - math.pow(math.fabs(x_val_rot_norm - x_id_down), 2) *
                                      (2 - math.fabs(x_val_rot_norm - x_id_down)) / 2)
                    interp_x_up = (2 / 3 - math.pow(math.fabs(x_val_rot_norm - (x_id_down + 1)), 2) *
                                      (2 - math.fabs(x_val_rot_norm - (x_id_down + 1))) / 2)
                    interp_x_up_up = math.pow(2 - math.fabs(x_val_rot_norm - (x_id_down + 2)), 3) / 6

                    interp_z_down_down = math.pow(2 - math.fabs(z_val_rot_norm - (z_id_down - 1)), 3) / 6
                    interp_z_down = (2 / 3 - math.pow(math.fabs(z_val_rot_norm - z_id_down), 2) *
                        (2 - math.fabs(z_val_rot_norm - z_id_down)) / 2)
                    interp_z_up = (2 / 3 - math.pow(math.fabs(z_val_rot_norm - (z_id_down + 1)), 2) *
                        (2 - math.fabs(z_val_rot_norm - (z_id_down + 1))) / 2)
                    interp_z_up_up = math.pow(2 - math.fabs(z_val_rot_norm - (z_id_down + 2)), 3) / 6


                    val_local = 0j
                    for shift_x in [-1, 0, 1, 2]:
                        if shift_x == -1:
                            interp_x_local = interp_x_down_down
                        elif shift_x == 0:
                            interp_x_local = interp_x_down
                        elif shift_x == 1:
                            interp_x_local = interp_x_up
                        else:
                            interp_x_local = interp_x_up_up

                        for shift_z in [-1, 0, 1, 2]:
                            if shift_z == -1:
                                interp_z_local = interp_z_down_down
                            elif shift_z == 0:
                                interp_z_local = interp_z_down
                            elif shift_z == 1:
                                interp_z_local = interp_z_up
                            else:
                                interp_z_local = interp_z_up_up

                            if x_id_down + shift_x > 0 and x_id_down + shift_x < n_z_small and\
                               z_id_down + shift_z > 0 and z_id_down + shift_z < n_z_small:
                                val_local += interp_x_local * interp_z_local *\
                                             patches_in[x_patch_offset_input + patch_id_x, patch_id_z, tx_id,
                                                        x_id_down + shift_x, z_id_down + shift_z]

                    val_out += val_local

                win_rad_local[patch_id_x, patch_id_z, tx_id, rx_id, z_small_id] = val_out
        cuda.syncthreads()

        for loop_id in range(loop_number_tx_z_small):
            full_id = loop_id * thread_number + thread_id
            tx_id = full_id // n_z_small
            z_small_id = full_id % n_z_small

            if tx_id >= n_tx:
                break

            if z_small_id == n_z_small - 1:
                continue

            for rx_id in range(n_rx):
                val = 0j
                for z_id_sum in range(n_z_small):

                    val += win_rad_local[patch_id_x, patch_id_z, tx_id, rx_id, z_id_sum] * \
                           mat_filt_in[z_small_id, z_id_sum]


                shift_2 = k0 * (z_coord_small[z_small_id] + delta_z / 2)
                cos_shift_2 = math.cos(shift_2)
                sin_shift_2 = math.sin(shift_2)
                val *= (cos_shift_2 + 1j * sin_shift_2)

                win_rad_filt_out[patch_id_x + x_patch_offset_output, patch_id_z, tx_id, rx_id, z_small_id] = val

    return patch_radon_transform_rx_cuda


def get_decomposition_function_gpu(angles_tx, angles_rx, angles_mid, z_coord_small_out, reg_param, z_size_patch,
                                   tx_dominates, n_iter, thread_number=1024):
    '''
    Function to compute the decomposition of the patches

    :param angles_tx: Tx angles (N_tx)
    :param angles_rx: Rx angles (N_rx)
    :param angles_mid: mid angles (N_mid)
    :param z_coord_small_out: z coordinates of the sinograms
    :param reg_param: regularization parameter
    :param z_size_patch: number of patches along the z axis
    :param tx_dominates: if the Tx angles dominates
    :param n_iter: number of iterations
    :param thread_number: number of threads
    :return: the decomposition function
    '''

    n_tx = angles_tx.shape[0]
    n_rx = angles_rx.shape[0]
    n_mid = angles_mid.shape[0]
    n_z_out = z_coord_small_out.shape[0]

    loop_number_full = np.int32(np.ceil(n_tx * n_rx * n_z_out / thread_number))
    loop_number_f = np.int32(np.ceil((n_mid * n_z_out + 1) / thread_number))
    loop_number_tx_rx_plus_mid = np.int32(np.ceil((n_tx * n_rx + n_mid) / thread_number))
    loop_number_tx_rx = np.int32(np.ceil((n_tx * n_rx) / thread_number))

    n_tx_rx_shape = (n_tx, n_rx)

    delta_mid = angles_mid[1] - angles_mid[0]

    if tx_dominates:
        angular_resolution_upsampling = np.int32(np.round((angles_rx[1] - angles_rx[0]) /
                                                          (angles_tx[1] - angles_tx[0])))
    else:
        angular_resolution_upsampling = np.int32(np.round((angles_tx[1] - angles_tx[0]) /
                                                          (angles_rx[1] - angles_rx[0])))


    @cuda.jit(max_registers=10)
    def get_decomposition_gpu(f_out_real, f_out_imag, u_tx_out, u_rx_out, g_in):

        tx_norm2_shared = cuda.shared.array(shape=1, dtype=np.float32)
        rx_norm2_shared = cuda.shared.array(shape=1, dtype=np.float32)
        f_norm2_shared = cuda.shared.array(shape=1, dtype=np.float32)
        u_tx_real_shared = cuda.shared.array(shape=n_tx, dtype=np.float32)
        u_tx_imag_shared = cuda.shared.array(shape=n_tx, dtype=np.float32)
        u_rx_real_shared = cuda.shared.array(shape=n_rx, dtype=np.float32)
        u_rx_imag_shared = cuda.shared.array(shape=n_rx, dtype=np.float32)
        f_val_down_shared = cuda.shared.array(shape=n_mid, dtype=np.float32)
        tx_val_down_shared = cuda.shared.array(shape=n_tx, dtype=np.float32)
        rx_val_down_shared = cuda.shared.array(shape=n_rx, dtype=np.float32)

        f_shift_g_sum_real_shared = cuda.shared.array(shape=n_tx_rx_shape, dtype=np.float32)
        f_shift_g_sum_imag_shared = cuda.shared.array(shape=n_tx_rx_shape, dtype=np.float32)
        f_shift_2_sum_shared = cuda.shared.array(shape=n_tx_rx_shape, dtype=np.float32)


        block_id = cuda.blockIdx.x
        thread_id = cuda.threadIdx.x

        # determine the patch ids
        patch_x_id = block_id // z_size_patch
        patch_z_id = block_id % z_size_patch


        # determine the norms of u_tx and u_rx
        if thread_id == 0:
            tx_norm2_shared[0] = 0
            rx_norm2_shared[0] = 0
        cuda.syncthreads()

        if thread_id < n_tx:
            u_tx_real_shared[thread_id] = 1
            u_tx_imag_shared[thread_id] = 0
            cuda.atomic.add(tx_norm2_shared, 0, u_tx_real_shared[thread_id] ** 2 + u_tx_imag_shared[thread_id] ** 2)
        elif thread_id < n_tx + n_rx:
            u_rx_real_shared[thread_id - n_tx] = 1
            u_rx_imag_shared[thread_id - n_tx] = 0
            cuda.atomic.add(rx_norm2_shared, 0, u_rx_real_shared[thread_id - n_tx] ** 2 +
                            u_rx_imag_shared[thread_id - n_tx] ** 2)
        cuda.syncthreads()


        for iter_id in range(n_iter):
            # === Compute the value of f ===

            # set val down to 0
            if thread_id < n_mid:
                f_val_down_shared[thread_id] = 0

            # compute val down
            for loop_id in range(loop_number_tx_rx_plus_mid):
                full_id_local = loop_id * thread_number + thread_id

                if full_id_local < n_tx * n_rx:
                    tx_id = full_id_local // n_rx
                    rx_id = full_id_local % n_rx
                    if tx_dominates:
                        mid_id_local = tx_id + rx_id * angular_resolution_upsampling
                    else:
                        mid_id_local = rx_id + tx_id * angular_resolution_upsampling
                    cuda.atomic.add(f_val_down_shared, mid_id_local,
                                    (u_tx_real_shared[tx_id] ** 2 + u_tx_imag_shared[tx_id] ** 2) *
                                    (u_rx_real_shared[rx_id] ** 2 + u_rx_imag_shared[rx_id] ** 2))
                elif full_id_local < n_tx * n_rx + n_mid:
                    cuda.atomic.add(f_val_down_shared, full_id_local - n_tx * n_rx,
                                    reg_param * tx_norm2_shared[0] * rx_norm2_shared[0] * delta_mid)
            cuda.syncthreads()

            # set f to 0
            for loop_id in range(loop_number_f):
                full_id_local = loop_id * thread_number + thread_id
                if full_id_local < n_mid * n_z_out:
                    mid_id = full_id_local // n_z_out
                    z_out_id = full_id_local % n_z_out
                    f_out_real[patch_x_id, patch_z_id, mid_id, z_out_id] = 0
                    f_out_imag[patch_x_id, patch_z_id, mid_id, z_out_id] = 0
                elif full_id_local == n_mid * n_z_out:
                    f_norm2_shared[0] = 0
            cuda.syncthreads()

            # compute f
            for loop_id in range(loop_number_full):
                full_id_local = loop_id * thread_number + thread_id

                tx_id = full_id_local // (n_rx * n_z_out)
                if tx_id < n_tx:
                    full_id_local_1down = full_id_local % (n_rx * n_z_out)
                    rx_id = full_id_local_1down // n_z_out
                    z_out_id = full_id_local_1down % n_z_out
                    if tx_dominates:
                        mid_id_local = tx_id + rx_id * angular_resolution_upsampling
                    else:
                        mid_id_local = rx_id + tx_id * angular_resolution_upsampling

                    val_local = (u_tx_real_shared[tx_id] - 1j * u_tx_imag_shared[tx_id]) *\
                                (u_rx_real_shared[rx_id] - 1j * u_rx_imag_shared[rx_id]) *\
                                g_in[patch_x_id, patch_z_id, tx_id, rx_id, z_out_id]
                    val_local = val_local / f_val_down_shared[mid_id_local]
                    cuda.atomic.add(f_out_real, (patch_x_id, patch_z_id, mid_id_local, z_out_id), val_local.real)
                    cuda.atomic.add(f_out_imag, (patch_x_id, patch_z_id, mid_id_local, z_out_id), val_local.imag)
            cuda.syncthreads()

            # compute the norm of f
            for loop_id in range(loop_number_f):
                full_id_local = loop_id * thread_number + thread_id
                if full_id_local < n_mid * n_z_out:
                    mid_id = full_id_local // n_z_out
                    z_out_id = full_id_local % n_z_out
                    cuda.atomic.add(f_norm2_shared, 0, f_out_real[patch_x_id, patch_z_id, mid_id, z_out_id]**2 +
                                    f_out_imag[patch_x_id, patch_z_id, mid_id, z_out_id]**2)
            cuda.syncthreads()

            # === Compute the value of u_tx and  u_rx ===

            # set to 0 the intermediary variables
            for loop_id in range(loop_number_tx_rx):
                full_id_local = loop_id * thread_number + thread_id
                if full_id_local < n_tx * n_rx:
                    tx_id = full_id_local // n_rx
                    rx_id = full_id_local % n_rx

                    f_shift_g_sum_real_shared[tx_id, rx_id] = 0
                    f_shift_g_sum_imag_shared[tx_id, rx_id] = 0
                    f_shift_2_sum_shared[tx_id, rx_id] = 0
            cuda.syncthreads()

            # compute the intermediary variables
            for loop_id in range(loop_number_full):
                full_id_local = loop_id * thread_number + thread_id

                tx_id = full_id_local // (n_rx * n_z_out)
                if tx_id < n_tx:
                    full_id_local_1down = full_id_local % (n_rx * n_z_out)
                    rx_id = full_id_local_1down // n_z_out
                    z_out_id = full_id_local_1down % n_z_out
                    if tx_dominates:
                        mid_id_local = tx_id + rx_id * angular_resolution_upsampling
                    else:
                        mid_id_local = rx_id + tx_id * angular_resolution_upsampling

                    val_local = (f_out_real[patch_x_id, patch_z_id, mid_id_local, z_out_id] -
                                 1j * f_out_imag[patch_x_id, patch_z_id, mid_id_local, z_out_id]) *\
                                (g_in[patch_x_id, patch_z_id, tx_id, rx_id, z_out_id])
                    cuda.atomic.add(f_shift_g_sum_real_shared, (tx_id, rx_id), val_local.real)
                    cuda.atomic.add(f_shift_g_sum_imag_shared, (tx_id, rx_id), val_local.imag)
                    val_2_local = f_out_real[patch_x_id, patch_z_id, mid_id_local, z_out_id]**2 +\
                                  f_out_imag[patch_x_id, patch_z_id, mid_id_local, z_out_id]**2
                    cuda.atomic.add(f_shift_2_sum_shared, (tx_id, rx_id), val_2_local)
            cuda.syncthreads()

            if thread_id < n_tx:
                u_tx_real_shared[thread_id] = 0
                u_tx_imag_shared[thread_id] = 0
                tx_val_down_shared[thread_id] = 0
            cuda.syncthreads()

            # compute the val up and down for u_tx
            for loop_id in range(loop_number_tx_rx):
                full_id_local = loop_id * thread_number + thread_id

                if full_id_local < n_tx * n_rx:
                    tx_id = full_id_local // n_rx
                    rx_id = full_id_local % n_rx

                    val_local = (f_shift_g_sum_real_shared[tx_id, rx_id] +
                                 1j * f_shift_g_sum_imag_shared[tx_id, rx_id]) *\
                                (u_rx_real_shared[rx_id] - 1j * u_rx_imag_shared[rx_id])
                    cuda.atomic.add(u_tx_real_shared, tx_id, val_local.real)
                    cuda.atomic.add(u_tx_imag_shared, tx_id, val_local.imag)

                    val_2_local = f_shift_2_sum_shared[tx_id, rx_id] *\
                                  (u_rx_real_shared[rx_id]**2 + u_rx_imag_shared[rx_id]**2)
                    cuda.atomic.add(tx_val_down_shared, tx_id, val_2_local)
            cuda.syncthreads()

            # compute the the values of u_tx
            if thread_id < n_tx:
                u_tx_real_shared[thread_id] = u_tx_real_shared[thread_id] /\
                    (tx_val_down_shared[thread_id] + reg_param * delta_mid * rx_norm2_shared[0] * f_norm2_shared[0])
            elif thread_id < 2 * n_tx:
                u_tx_imag_shared[thread_id - n_tx] = u_tx_imag_shared[thread_id - n_tx] /\
                    (tx_val_down_shared[thread_id - n_tx] +
                     reg_param * delta_mid * rx_norm2_shared[0] * f_norm2_shared[0])
            elif thread_id == 2 * n_tx:
                tx_norm2_shared[0] = 0
            cuda.syncthreads()

            # compute the norm of u_tx
            if thread_id < n_tx:
                cuda.atomic.add(tx_norm2_shared, 0, u_tx_real_shared[thread_id] ** 2 + u_tx_imag_shared[thread_id] ** 2)
            cuda.syncthreads()

            if thread_id < n_rx:
                u_rx_real_shared[thread_id] = 0
                u_rx_imag_shared[thread_id] = 0
                rx_val_down_shared[thread_id] = 0
            cuda.syncthreads()

            # compute the val up and down for u_tx
            for loop_id in range(loop_number_tx_rx):
                full_id_local = loop_id * thread_number + thread_id

                if full_id_local < n_tx * n_rx:
                    tx_id = full_id_local // n_rx
                    rx_id = full_id_local % n_rx

                    val_local = (f_shift_g_sum_real_shared[tx_id, rx_id] +
                                 1j * f_shift_g_sum_imag_shared[tx_id, rx_id]) *\
                                (u_tx_real_shared[tx_id] - 1j * u_tx_imag_shared[tx_id])
                    cuda.atomic.add(u_rx_real_shared, rx_id, val_local.real)
                    cuda.atomic.add(u_rx_imag_shared, rx_id, val_local.imag)

                    val_2_local = f_shift_2_sum_shared[tx_id, rx_id] *\
                                  (u_tx_real_shared[tx_id]**2 + u_tx_imag_shared[tx_id]**2)
                    cuda.atomic.add(rx_val_down_shared, rx_id, val_2_local)
            cuda.syncthreads()

            # compute the the values of u_rx
            if thread_id < n_rx:
                u_rx_real_shared[thread_id] = u_rx_real_shared[thread_id] / \
                      (rx_val_down_shared[thread_id] + reg_param * delta_mid * tx_norm2_shared[0] *
                       f_norm2_shared[0])
            elif thread_id < 2 * n_rx:
                u_rx_imag_shared[thread_id - n_rx] = u_rx_imag_shared[thread_id - n_rx] / \
                     (rx_val_down_shared[thread_id - n_rx] + reg_param * delta_mid * tx_norm2_shared[0] *
                      f_norm2_shared[0])
            elif thread_id == 2 * n_rx:
                rx_norm2_shared[0] = 0
            cuda.syncthreads()

            # compute the norm of u_rx
            if thread_id < n_rx:
                cuda.atomic.add(rx_norm2_shared, 0, u_rx_real_shared[thread_id] ** 2 + u_rx_imag_shared[thread_id] ** 2)
            cuda.syncthreads()

        # === Out values ===
        if thread_id < n_tx:
            u_tx_out[patch_x_id, patch_z_id, thread_id] =\
                (u_tx_real_shared[thread_id] + 1j * u_tx_imag_shared[thread_id]) / math.sqrt(tx_norm2_shared[0])
        elif thread_id < n_tx + n_rx:
            u_rx_out[patch_x_id, patch_z_id, thread_id - n_tx] =\
                (u_rx_real_shared[thread_id - n_tx] + 1j * u_rx_imag_shared[thread_id - n_tx]) /\
                math.sqrt(rx_norm2_shared[0])

        # reweight f
        for loop_id in range(loop_number_f):
            full_id_local = loop_id * thread_number + thread_id

            mid_id = full_id_local // n_z_out
            if mid_id < n_mid:
                z_out_id = full_id_local % n_z_out
                f_out_real[patch_x_id, patch_z_id, mid_id, z_out_id] *= math.sqrt(tx_norm2_shared[0]) *\
                                                                        math.sqrt(rx_norm2_shared[0])
                f_out_imag[patch_x_id, patch_z_id, mid_id, z_out_id] *= math.sqrt(tx_norm2_shared[0]) *\
                                                                        math.sqrt(rx_norm2_shared[0])

    return get_decomposition_gpu


def get_patch_backprojection_mid_window_cuda_function(
        angles_mid, z_coord_small_out, z_size_patch, thread_number= 1024, max_registers=4, M_first_sum=12):
    '''
    Get the patch backprojection function

    :param angles_mid: mid angles
    :param z_coord_small_out: z coord of the out patch
    :param z_size_patch: number of patches in the z dimension
    :param max_registers: number of registers used
    :return:
    '''

    z_coord_small_out_size = z_coord_small_out.shape[0]


    z_small_min = z_coord_small_out[0]
    delta_z = z_coord_small_out[1] - z_coord_small_out[0]

    n_mid = angles_mid.shape[0]

    loop_number_1 = np.int32(np.ceil(angles_mid.shape[0] / thread_number))
    loop_number_2 = np.int32(np.ceil(z_coord_small_out.shape[0] ** 2 / thread_number))

    l_val = 6
    z_p = np.sqrt(3) - 2

    @cuda.jit(max_registers=max_registers)
    def patch_backprojection_gpu(patch_out, array_real_in, array_imag_in, window_val_out_in,
                                 x_offset_in, x_offset_out):

        # prefilter the data

        # determine the patch id
        block_id = cuda.blockIdx.x
        patch_id_x = block_id // z_size_patch
        patch_id_z = block_id % z_size_patch

        thread_id = cuda.threadIdx.x
        for loop_id in range(loop_number_1):
            mid_id = loop_id * thread_number + thread_id

            if mid_id >= n_mid:
                break

            # compute the first element
            z_p_power = 1.
            val_sum_real = 0
            val_sum_imag = 0
            for m_id in range(M_first_sum):
                z_p_power *= z_p
                val_sum_real += array_real_in[x_offset_in + patch_id_x, patch_id_z, mid_id, m_id] * z_p_power
                val_sum_imag += array_imag_in[x_offset_in + patch_id_x, patch_id_z, mid_id, m_id] * z_p_power

            # forward pass
            for z_id in range(z_coord_small_out_size):
                if z_id == 0:
                    array_real_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id] = l_val * \
                        (array_real_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id] + val_sum_real)
                    array_imag_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id] = l_val * \
                        (array_imag_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id] + val_sum_imag)

                else:
                    array_real_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id] = \
                        l_val * array_real_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id] + \
                        z_p * array_real_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id - 1]
                    array_imag_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id] = \
                        l_val * array_imag_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id] + \
                        z_p * array_imag_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id - 1]

            # backward pass
            for z_id in range(z_coord_small_out_size):
                z_id_local = z_coord_small_out_size - 1 - z_id
                if z_id_local == z_coord_small_out_size - 1:
                    array_real_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id_local] = -(z_p / (1 - z_p)) * \
                          (array_real_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id_local])
                    array_imag_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id_local] = -(z_p / (1 - z_p)) * \
                          (array_imag_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id_local])
                else:
                    array_real_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id_local] = z_p * \
                          (array_real_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id_local + 1] -
                           array_real_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id_local])
                    array_imag_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id_local] = z_p * \
                          (array_imag_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id_local + 1] -
                           array_imag_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id_local])
        cuda.syncthreads()


        # perform the backprojection
        for loop_id in range(loop_number_2):
            full_id = loop_id * thread_number + thread_id

            x_small_id = full_id // z_coord_small_out_size
            z_small_id = full_id % z_coord_small_out_size

            if x_small_id >= z_coord_small_out_size:
                break

            x_local = z_coord_small_out[x_small_id]
            z_local = z_coord_small_out[z_small_id]

            val_out = 0j
            for mid_id in range(n_mid):

                mid_angle_local = angles_mid[mid_id]
                z_val_local = z_local * math.cos(mid_angle_local) + x_local * math.sin(mid_angle_local)
                z_norm = (z_val_local - z_small_min) / delta_z

                z_id_down = np.int32(math.floor(z_norm))
                z_id_down_down = z_id_down - 1
                z_id_up = z_id_down + 1
                z_id_up_up = z_id_up + 1

                val_real_local = 0
                val_imag_local = 0

                if z_id_down_down >= 0 and z_id_down_down < z_coord_small_out_size:
                    interp_local = (2 - abs(z_norm - z_id_down_down)) ** 3 / 6
                    val_real_local += array_real_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id_down_down] * \
                                      interp_local
                    val_imag_local += array_imag_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id_down_down] * \
                                      interp_local

                if z_id_down >= 0 and z_id_down < z_coord_small_out_size:
                    interp_local = (2 / 3 - abs(z_norm - z_id_down) ** 2 * (2 - abs(z_norm - z_id_down)) / 2)
                    val_real_local += array_real_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id_down] * \
                                      interp_local
                    val_imag_local += array_imag_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id_down] * \
                                      interp_local

                if z_id_up >= 0 and z_id_up < z_coord_small_out_size:
                    interp_local = (2 / 3 - abs(z_norm - z_id_up) ** 2 * (2 - abs(z_norm - z_id_up)) / 2)
                    val_real_local += array_real_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id_up] * \
                                      interp_local
                    val_imag_local += array_imag_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id_up] * \
                                      interp_local

                if z_id_up_up >= 0 and z_id_up_up < z_coord_small_out_size:
                    interp_local = (2 - abs(z_norm - z_id_up_up)) ** 3 / 6
                    val_real_local += array_real_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id_up_up] * \
                                      interp_local
                    val_imag_local += array_imag_in[x_offset_in + patch_id_x, patch_id_z, mid_id, z_id_up_up] * \
                                      interp_local

                val_out += (val_real_local + 1j * val_imag_local)

            patch_out[x_offset_out + patch_id_x, patch_id_z, x_small_id, z_small_id] = \
                val_out * window_val_out_in[x_small_id, z_small_id]

    return patch_backprojection_gpu


def get_reconstruct_image_functions_gpu(x_coord_patch, z_coord_patch, z_coord_small, norm_thresh, thread_number):
    '''
    Get the different GPU functions for the image reconstruction

    :param x_coord_patch: x coordinates of the patches
    :param z_coord_patch: z coordinates of the patches
    :param z_coord_small: z coordinates with respect to the center of the patch
    :param norm_thresh: threshold for the normalization
    :param thread_number: number of threads
    :return: the function to reconstruct an image, the function to compute shifts, the function to get normalization,
    the function to normalize, the block numbers for the three function, the coordinates of the output image
    '''


    # define the upsampling factors
    upsampling_factor_x = np.int32(
        np.round((x_coord_patch[1] - x_coord_patch[0]) / (z_coord_small[1] - z_coord_small[0])))
    if np.abs(upsampling_factor_x - (
            (x_coord_patch[1] - x_coord_patch[0]) / (z_coord_small[1] - z_coord_small[0]))) > 1e-6:
        raise ValueError('Resolution of x_coord is not a multiple of the resolution of z_coord_small')

    upsampling_factor_z = np.int32(
        np.round((z_coord_patch[1] - z_coord_patch[0]) / (z_coord_small[1] - z_coord_small[0])))
    if np.abs(upsampling_factor_z - (
            (z_coord_patch[1] - z_coord_patch[0]) / (z_coord_small[1] - z_coord_small[0]))) > 1e-6:
        raise ValueError('Resolution of z_coord is not a multiple of the resolution of z_coord_small')

    # define the output coordinates
    x_coord_out = np.arange((x_coord_patch.shape[0] - 1) * upsampling_factor_x + z_coord_small.shape[0]) * \
                  (z_coord_small[1] - z_coord_small[0]) + x_coord_patch[0] - \
                  (z_coord_small.shape[0] - 1) / 2 * (z_coord_small[1] - z_coord_small[0])

    z_coord_out = np.arange((z_coord_patch.shape[0] - 1) * upsampling_factor_z + z_coord_small.shape[0]) * \
                  (z_coord_small[1] - z_coord_small[0]) + z_coord_patch[0] - \
                  (z_coord_small.shape[0] - 1) / 2 * (z_coord_small[1] - z_coord_small[0])

    n_z_small_out = z_coord_small.shape[0]
    z_size_patch = z_coord_patch.shape[0]
    x_size_patch = x_coord_patch.shape[0]

    # define the block coordinates
    block_number_1 = np.int32(np.ceil((x_size_patch * z_size_patch * n_z_small_out ** 2) / thread_number))
    block_number_2 = x_size_patch * z_size_patch
    block_number_3 = np.int32(np.ceil(x_coord_out.shape[0] * z_coord_out.shape[0] / thread_number))

    # function to get an image from patches
    @cuda.jit(max_registers=4)
    def reconstruct_image(patches_in, img_real_out, img_imag_out, shift_value):
        full_id = cuda.grid(1)
        patch_id_x = full_id // (z_size_patch * n_z_small_out * n_z_small_out)
        full_id_1down = full_id % (z_size_patch * n_z_small_out * n_z_small_out)
        patch_id_z = full_id_1down // (n_z_small_out * n_z_small_out)
        full_id_2down = full_id_1down % (n_z_small_out * n_z_small_out)
        small_id_x = full_id_2down // (n_z_small_out)
        small_id_z = full_id_2down % (n_z_small_out)

        if patch_id_x >= x_size_patch:
            return

        img_x_id = small_id_x + upsampling_factor_x * patch_id_x
        img_z_id = small_id_z + upsampling_factor_z * patch_id_z

        patches_in[patch_id_x, patch_id_z, small_id_x, small_id_z] *= (shift_value[patch_id_x, patch_id_z])

        cuda.atomic.add(img_real_out, (img_x_id, img_z_id),
                        patches_in[patch_id_x, patch_id_z, small_id_x, small_id_z].real)
        cuda.atomic.add(img_imag_out, (img_x_id, img_z_id),
                        patches_in[patch_id_x, patch_id_z, small_id_x, small_id_z].imag)

    loop_number_n_z_n_z = np.int32(np.ceil(n_z_small_out ** 2 / thread_number))

    # function to get the correction factors from an image and patches
    @cuda.jit(max_registers=4)
    def update_shift(patches, img_real, img_imag, shift_value):
        block_id = cuda.blockIdx.x
        thread_id = cuda.threadIdx.x

        patch_x_id = block_id // z_size_patch
        patch_z_id = block_id % z_size_patch

        if thread_id == 0:
            shift_value[patch_x_id, patch_z_id] = 0j
        cuda.syncthreads()

        for loop_id in range(loop_number_n_z_n_z):
            full_id = loop_id * thread_number + thread_id

            small_x_id = full_id // n_z_small_out
            small_z_id = full_id % n_z_small_out

            if small_x_id >= n_z_small_out:
                break

            img_x_id = small_x_id + upsampling_factor_x * patch_x_id
            img_z_id = small_z_id + upsampling_factor_z * patch_z_id

            val_local = (img_real[img_x_id, img_z_id] + 1j * img_imag[img_x_id, img_z_id]) - \
                        patches[patch_x_id, patch_z_id, small_x_id, small_z_id]
            val_local *= patches[patch_x_id, patch_z_id, small_x_id, small_z_id].real - \
                         1j * patches[patch_x_id, patch_z_id, small_x_id, small_z_id].imag

            shift_value[patch_x_id, patch_z_id] += val_local
        cuda.syncthreads()

        if thread_id == 0:
            shift_norm = math.sqrt(shift_value[patch_x_id, patch_z_id].real ** 2 +
                                   shift_value[patch_x_id, patch_z_id].imag ** 2)
            shift_value[patch_x_id, patch_z_id] /= shift_norm

    x_coord_out_size = x_coord_out.shape[0]
    z_coord_out_size = z_coord_out.shape[0]

    # function to get the normalization factor
    @cuda.jit(max_registers=4)
    def get_normalization(window_val, norm_val):

        full_id = cuda.grid(1)

        patch_id_x = full_id // (z_size_patch * n_z_small_out * n_z_small_out)
        full_id_1down = full_id % (z_size_patch * n_z_small_out * n_z_small_out)
        patch_id_z = full_id_1down // (n_z_small_out * n_z_small_out)
        full_id_2down = full_id_1down % (n_z_small_out * n_z_small_out)
        small_id_x = full_id_2down // (n_z_small_out)
        small_id_z = full_id_2down % (n_z_small_out)

        if patch_id_x < x_size_patch:
            img_x_id = small_id_x + upsampling_factor_x * patch_id_x
            img_z_id = small_id_z + upsampling_factor_z * patch_id_z

            cuda.atomic.add(norm_val, (img_x_id, img_z_id), window_val[small_id_x, small_id_z] ** 2)

    # function to normalize the image
    @cuda.jit(max_registers=4)
    def normalize_image(img_real, img_imag, norm_val):

        full_id = cuda.grid(1)
        img_x_id = full_id // z_coord_out_size
        img_z_id = full_id % z_coord_out_size

        if img_x_id >= x_coord_out_size:
            return

        if norm_val[img_x_id, img_z_id] > norm_thresh:
            img_real[img_x_id, img_z_id] /= norm_val[img_x_id, img_z_id]
            img_imag[img_x_id, img_z_id] /= norm_val[img_x_id, img_z_id]
        else:
            img_real[img_x_id, img_z_id] = 0
            img_imag[img_x_id, img_z_id] = 0

    return reconstruct_image, update_shift, get_normalization, normalize_image, block_number_1, block_number_2,\
           block_number_3, x_coord_out, z_coord_out

###########
### CPU ###
###########

# === Functions for the delay-and-sum ===

def get_beamformer_npw_linear_transducer_Tukey_phase_screen_cpu(
        angles_tx, sensors_x, t_coord, t_upsampling, x_coord_beam, z_coord_beam, c0,
        apod_tukey_angle_max, apod_tukey_cosine_frac,
        return_one_per_angle=False):
    
    # Pre-computation of constants
    delta_t_up = (t_coord[1] - t_coord[0]) / t_upsampling
    n_t_up = (t_coord.shape[0] - 1) * t_upsampling + 1
    t0 = t_coord[0]
    n_elem = sensors_x.shape[0]
    n_x = x_coord_beam.shape[0]
    n_z = z_coord_beam.shape[0]
    
    # Compile the inner loop for CPU
    @njit(parallel=True, fastmath=True)
    def beamform_cpu_kernel(data_in, angle_tx):
        # Output buffer for this specific angle
        img_out_local = np.zeros((n_x, n_z), dtype=np.complex128)
        
        # Loop over grid pixels (parallelized)
        for x_id in prange(n_x):
            x_val = x_coord_beam[x_id]
            for z_id in range(n_z):
                z_val = z_coord_beam[z_id]
                
                # Compute TX delay
                d_tx = z_val * math.cos(angle_tx) + x_val * math.sin(angle_tx)
                
                val_pixel = 0j
                
                # Loop over sensors (Summation)
                for elem_id in range(n_elem):
                    elem_x = sensors_x[elem_id]
                    
                    # Compute RX delay
                    d_rx = math.sqrt((x_val - elem_x)**2 + z_val**2)
                    
                    # Total time and index
                    t_total = (d_tx + d_rx) / c0
                    t_idx_float = (t_total - t0) / delta_t_up
                    
                    # Linear Interpolation
                    if 0 <= t_idx_float < (n_t_up - 1):
                        t_id_down = int(t_idx_float)
                        t_frac = t_idx_float - t_id_down
                        
                        sample = (1 - t_frac) * data_in[0, elem_id, t_id_down] + \
                                 t_frac * data_in[0, elem_id, t_id_down + 1]
                        
                        # Apodization (Tukey)
                        theta_angle = math.atan2(x_val - elem_x, z_val)
                        if apod_tukey_angle_max > 0:
                            ratio = abs(theta_angle) / apod_tukey_angle_max
                            if ratio > 1:
                                weight = 0.0
                            elif ratio < (1 - apod_tukey_cosine_frac):
                                weight = 1.0
                            else:
                                weight = 0.5 * (1 + math.cos(math.pi / apod_tukey_cosine_frac * (ratio - 1 + apod_tukey_cosine_frac)))
                        else:
                            weight = 1.0 if abs(theta_angle) < apod_tukey_angle_max else 0.0 # Fallback
                            
                        val_pixel += sample * weight
                        
                img_out_local[x_id, z_id] = val_pixel
        return img_out_local

    def beamform(data):
        if return_one_per_angle:
            img_out = np.zeros((angles_tx.shape[0], x_coord_beam.shape[0], z_coord_beam.shape[0]), dtype=np.complex128)
        else:
            img_out = np.zeros((x_coord_beam.shape[0], z_coord_beam.shape[0]), dtype=np.complex128)

        for tx_id, angle_tx in enumerate(angles_tx):
            # Upsample data for this angle
            data_local = data[tx_id, :, :]
            # Simple linear upsampling for speed on CPU, cubic is slow
            # If high quality needed, keep scipy but it's slow inside loop.
            # Here we do pre-interpolation outside the kernel
            
            # Using scipy for consistency with original code for the time interpolation
            data_local_interp_real = scipy.interpolate.interp1d(t_coord, np.real(data_local), kind='linear', fill_value="extrapolate")
            data_local_interp_imag = scipy.interpolate.interp1d(t_coord, np.imag(data_local), kind='linear', fill_value="extrapolate")
            t_new = delta_t_up * np.arange(n_t_up) + t0
            data_local_up = data_local_interp_real(t_new) + 1j * data_local_interp_imag(t_new)
            
            # Add new axis to match dimensions expected by kernel
            data_in_kernel = data_local_up[np.newaxis, :, :]
            
            # Run kernel
            img_angle = beamform_cpu_kernel(data_in_kernel, angle_tx)
            
            if return_one_per_angle:
                img_out[tx_id, :, :] = img_angle
            else:
                img_out += img_angle
                
        return img_out

    return beamform

# === Functions for the adaptive beamforming (CPU Port) ===

def get_select_patch_window_cpu_function(angles_tx, z_coord_small, x_patch_pos, z_patch_pos):
    
    n_tx = angles_tx.shape[0]
    n_z_small = z_coord_small.shape[0]
    window_radius_pixels = (n_z_small + 1)//2

    @njit(parallel=True)
    def select_patch_window_cpu(img_in, window_val_in, x_size_batch, x_patch_offset_input):
        # We assume patch_out is allocated outside or we create a list? 
        # Numba parallel works best with pre-allocated arrays.
        # Calculating output shape:
        z_size_patch = z_patch_pos.shape[0]
        patch_out = np.zeros((x_size_batch, z_size_patch, n_tx, n_z_small, n_z_small), dtype=np.complex64)
        
        for x_id in prange(x_size_batch):
            global_x_idx = x_id + x_patch_offset_input
            x_center = x_patch_pos[global_x_idx]
            
            for z_id in range(z_size_patch):
                z_center = z_patch_pos[z_id]
                
                start_x = x_center - (window_radius_pixels - 1)
                start_z = z_center - (window_radius_pixels - 1)
                
                for tx_id in range(n_tx):
                    for i in range(n_z_small):
                        for j in range(n_z_small):
                            # Boundary check (simple clamping or zeroing could be added)
                            val = img_in[tx_id, start_x + i, start_z + j]
                            patch_out[x_id, z_id, tx_id, i, j] = val * window_val_in[i, j]
                            
        return patch_out

    return select_patch_window_cpu

def get_patch_radon_transform_rx_cpu_function(angles_tx, z_coord_small, angles_rx, k0, M_first_sum=12):
    
    n_tx = angles_tx.shape[0]
    n_z_small = z_coord_small.shape[0]
    n_rx = angles_rx.shape[0]
    z0 = z_coord_small[0]
    delta_z = z_coord_small[1] - z_coord_small[0]
    l_val = 6.0
    z_p = np.sqrt(3) - 2

    @njit(parallel=True, fastmath=True)
    def patch_radon_transform_rx_cpu(patches_in, mat_filt_in):
        # patches_in shape: (Batch_X, Batch_Z, Tx, Z_small, Z_small)
        batch_x_size = patches_in.shape[0]
        batch_z_size = patches_in.shape[1]
        
        # Output array
        # Size: Batch_X, Batch_Z, Tx, Rx, Z_small (filtered)
        win_rad_filt_out = np.zeros((batch_x_size, batch_z_size, n_tx, n_rx, n_z_small-1), dtype=np.complex64)
        
        # Temporary array for Radon (Rx dim)
        # We need this per thread effectively, but since we parallelize outer loops, we allocate inside
        
        for bx in prange(batch_x_size):
            for bz in range(batch_z_size):
                
                # Working copy for this patch to do in-place filtering
                patch_local = patches_in[bx, bz].copy()
                
                # --- Filter 1 (along Z dim of patch) ---
                for tx_id in range(n_tx):
                    for x_s in range(n_z_small):
                        # Prefilter sum
                        val_sum = 0j
                        z_p_power = 1.0
                        for m_id in range(M_first_sum):
                            z_p_power *= z_p
                            val_sum += patch_local[tx_id, x_s, m_id] * z_p_power
                            
                        # Forward
                        patch_local[tx_id, x_s, 0] = l_val * (patch_local[tx_id, x_s, 0] + val_sum)
                        for z_i in range(1, n_z_small):
                            patch_local[tx_id, x_s, z_i] = l_val * patch_local[tx_id, x_s, z_i] + \
                                                           z_p * patch_local[tx_id, x_s, z_i-1]
                        # Backward
                        patch_local[tx_id, x_s, n_z_small-1] = -z_p/(1-z_p) * patch_local[tx_id, x_s, n_z_small-1]
                        for z_i in range(n_z_small-2, -1, -1):
                            patch_local[tx_id, x_s, z_i] = z_p * (patch_local[tx_id, x_s, z_i+1] - patch_local[tx_id, x_s, z_i])

                # --- Filter 2 (along X dim of patch) ---
                for tx_id in range(n_tx):
                    for z_s in range(n_z_small):
                        val_sum = 0j
                        z_p_power = 1.0
                        for m_id in range(M_first_sum):
                            z_p_power *= z_p
                            val_sum += patch_local[tx_id, m_id, z_s] * z_p_power
                            
                        # Forward
                        patch_local[tx_id, 0, z_s] = l_val * (patch_local[tx_id, 0, z_s] + val_sum)
                        for x_i in range(1, n_z_small):
                            patch_local[tx_id, x_i, z_s] = l_val * patch_local[tx_id, x_i, z_s] + \
                                                           z_p * patch_local[tx_id, x_i-1, z_s]
                        # Backward
                        patch_local[tx_id, n_z_small-1, z_s] = -z_p/(1-z_p) * patch_local[tx_id, n_z_small-1, z_s]
                        for x_i in range(n_z_small-2, -1, -1):
                            patch_local[tx_id, x_i, z_s] = z_p * (patch_local[tx_id, x_i+1, z_s] - patch_local[tx_id, x_i, z_s])

                # Compute all Radon Projections for this patch
                # Shape: (Tx, Rx, Z_small)
                radon_temp = np.zeros((n_tx, n_rx, n_z_small), dtype=np.complex64)
                
                for tx_id in range(n_tx):
                    for rx_id in range(n_rx):
                        angle_mid = (angles_rx[rx_id] + angles_tx[tx_id]) / 2
                        cos_mid = math.cos(angle_mid)
                        sin_mid = math.sin(angle_mid)
                        
                        for z_out_id in range(n_z_small):
                            val_acc = 0j
                            for x_s in range(n_z_small):
                                x_coord = z_coord_small[x_s]
                                z_coord = z_coord_small[z_out_id]
                                x_rot = z_coord * sin_mid + x_coord * cos_mid
                                z_rot = z_coord * cos_mid - x_coord * sin_mid
                                x_norm = (x_rot - z0) / delta_z
                                z_norm = (z_rot - z0) / delta_z
                                x_idx = int(math.floor(x_norm))
                                z_idx = int(math.floor(z_norm))
                                if (x_idx >= 1) and (x_idx < n_z_small - 2) and (z_idx >= 1) and (z_idx < n_z_small - 2):
                                    # Inlined Spline for speed
                                    for i in range(-1, 3):
                                        d_x = abs(x_norm - (x_idx + i))
                                        if d_x < 1: wx = 2/3 - d_x**2 * (2 - d_x)/2
                                        elif d_x < 2: wx = (2 - d_x)**3 / 6
                                        else: wx = 0.0
                                        
                                        for j in range(-1, 3):
                                            d_z = abs(z_norm - (z_idx + j))
                                            if d_z < 1: 
                                                wz = 2/3 - d_z**2 * (2 - d_z)/2
                                            elif d_z < 2: 
                                                wz = (2 - d_z)**3 / 6
                                            else: 
                                                wz = 0.0
                                            
                                            val_acc += wx * wz * patch_local[tx_id, x_idx+i, z_idx+j]
                            radon_temp[tx_id, rx_id, z_out_id] = val_acc

                #Apply Matrix Filter
                for tx_id in range(n_tx):
                    for rx_id in range(n_rx):
                        for z_s in range(n_z_small - 1):
                            val = 0j
                            for k in range(n_z_small):
                                val += radon_temp[tx_id, rx_id, k] * mat_filt_in[z_s, k]
                            
                            shift_2 = k0 * (z_coord_small[z_s] + delta_z / 2)
                            phasor = math.cos(shift_2) + 1j * math.sin(shift_2)
                            win_rad_filt_out[bx, bz, tx_id, rx_id, z_s] = val * phasor

        return win_rad_filt_out

    return patch_radon_transform_rx_cpu


def get_decomposition_function_cpu(angles_tx, angles_rx, angles_mid, z_coord_small_out, reg_param, tx_dominates, n_iter):
    
    n_tx = angles_tx.shape[0]
    n_rx = angles_rx.shape[0]
    n_mid = angles_mid.shape[0]
    n_z_out = z_coord_small_out.shape[0]
    delta_mid = angles_mid[1] - angles_mid[0]

    if tx_dominates:
        angular_resolution_upsampling = int(round((angles_rx[1] - angles_rx[0]) / (angles_tx[1] - angles_tx[0])))
    else:
        angular_resolution_upsampling = int(round((angles_tx[1] - angles_tx[0]) / (angles_rx[1] - angles_rx[0])))

    @njit(parallel=True, fastmath=True)
    def decomposition_cpu(g_in):
        # g_in: (Batch_X, Batch_Z, Tx, Rx, Z_out)
        batch_x = g_in.shape[0]
        batch_z = g_in.shape[1]
        
        # Outputs
        f_out_real = np.zeros((batch_x, batch_z, n_mid, n_z_out), dtype=np.float32)
        f_out_imag = np.zeros((batch_x, batch_z, n_mid, n_z_out), dtype=np.float32)
        u_tx_out = np.zeros((batch_x, batch_z, n_tx), dtype=np.complex64)
        u_rx_out = np.zeros((batch_x, batch_z, n_rx), dtype=np.complex64)
        
        for bx in prange(batch_x):
            for bz in range(batch_z):
                # Initialize variables per patch
                u_tx = np.zeros((n_tx), dtype=np.complex64)
                u_tx[0:n_tx] = 1.0 + 0j
                u_rx = np.zeros((n_rx), dtype=np.complex64)
                u_rx[0:n_rx] = 1.0 + 0j
                
                f_current = np.zeros((n_mid, n_z_out), dtype=np.complex64)
                
                # Iterations
                for itr in range(n_iter):
                    
                    # 1. Update f
                    tx_norm = np.sum(np.abs(u_tx)**2)
                    rx_norm = np.sum(np.abs(u_rx)**2)
                    
                    # Denominator
                    denom_f = np.zeros(n_mid, dtype=np.float32)
                    for tx in range(n_tx):
                        for rx in range(n_rx):
                            if tx_dominates: mid = tx + rx * angular_resolution_upsampling
                            else: mid = rx + tx * angular_resolution_upsampling
                            denom_f[mid] += (np.abs(u_tx[tx])**2) * (np.abs(u_rx[rx])**2)
                    
                    denom_f += reg_param * tx_norm * rx_norm * delta_mid
                    
                    # Numerator
                    f_current[:] = 0
                    for tx in range(n_tx):
                        for rx in range(n_rx):
                            if tx_dominates: mid = tx + rx * angular_resolution_upsampling
                            else: mid = rx + tx * angular_resolution_upsampling
                            
                            # Correction: use np.conj() instead of .conj()
                            val = np.conj(u_tx[tx]) * np.conj(u_rx[rx]) * g_in[bx, bz, tx, rx, :]
                            f_current[mid, :] += val
                    
                    for m in range(n_mid):
                        if denom_f[m] > 1e-9:
                            f_current[m, :] /= denom_f[m]
                        else:
                            f_current[m, :] = 0
                            
                    f_norm = np.sum(np.abs(f_current)**2)

                    # Update u_tx
                    denom_tx = np.zeros(n_tx, dtype=np.float32)
                    num_tx = np.zeros(n_tx, dtype=np.complex64)
                    
                    # Precompute intersection
                    for tx in range(n_tx):
                        for rx in range(n_rx):
                            if tx_dominates: mid = tx + rx * angular_resolution_upsampling
                            else: mid = rx + tx * angular_resolution_upsampling
                            
                            # Term 1: <f, g>
                            # Correction: use np.conj() instead of .conj()
                            dot_fg = np.sum(np.conj(f_current[mid, :]) * g_in[bx, bz, tx, rx, :])
                            num_tx[tx] += dot_fg * np.conj(u_rx[rx])
                            
                            # Term 2: |f|^2
                            norm_f_mid = np.sum(np.abs(f_current[mid, :])**2)
                            denom_tx[tx] += norm_f_mid * (np.abs(u_rx[rx])**2)
                            
                    denom_tx += reg_param * delta_mid * rx_norm * f_norm
                    
                    for tx in range(n_tx):
                         if denom_tx[tx] > 1e-9: u_tx[tx] = num_tx[tx] / denom_tx[tx]
                    
                    tx_norm = np.sum(np.abs(u_tx)**2) # Update norm
                    
                    # Update u_rx
                    denom_rx = np.zeros(n_rx, dtype=np.float32)
                    num_rx = np.zeros(n_rx, dtype=np.complex64)
                    
                    for tx in range(n_tx):
                        for rx in range(n_rx):
                            if tx_dominates: mid = tx + rx * angular_resolution_upsampling
                            else: mid = rx + tx * angular_resolution_upsampling
                            
                            # Correction: use np.conj()
                            dot_fg = np.sum(np.conj(f_current[mid, :]) * g_in[bx, bz, tx, rx, :])
                            num_rx[rx] += dot_fg * np.conj(u_tx[tx])
                            
                            norm_f_mid = np.sum(np.abs(f_current[mid, :])**2)
                            denom_rx[rx] += norm_f_mid * (np.abs(u_tx[tx])**2)
                            
                    denom_rx += reg_param * delta_mid * tx_norm * f_norm
                    
                    for rx in range(n_rx):
                         if denom_rx[rx] > 1e-9: u_rx[rx] = num_rx[rx] / denom_rx[rx]

                # Finalize outputs
                tx_norm_final = math.sqrt(np.sum(np.abs(u_tx)**2))
                rx_norm_final = math.sqrt(np.sum(np.abs(u_rx)**2))
                
                if tx_norm_final > 0: u_tx_out[bx, bz, :] = u_tx / tx_norm_final
                if rx_norm_final > 0: u_rx_out[bx, bz, :] = u_rx / rx_norm_final
                
                f_out_real[bx, bz, :, :] = f_current.real * tx_norm_final * rx_norm_final
                f_out_imag[bx, bz, :, :] = f_current.imag * tx_norm_final * rx_norm_final
        
        return f_out_real, f_out_imag

    return decomposition_cpu

def get_patch_backprojection_mid_window_cpu_function(angles_mid, z_coord_small_out, M_first_sum=12):
    
    z_coord_small_out_size = z_coord_small_out.shape[0]
    z_small_min = z_coord_small_out[0]
    delta_z = z_coord_small_out[1] - z_coord_small_out[0]
    n_mid = angles_mid.shape[0]
    l_val = 6
    z_p = np.sqrt(3) - 2

    @njit(parallel=True, fastmath=True)
    def patch_backprojection_cpu(f_real, f_imag, window_val_out):
        # f inputs: (Batch_X, Batch_Z, N_mid, Z_out)
        bx_size = f_real.shape[0]
        bz_size = f_real.shape[1]
        
        patch_out = np.zeros((bx_size, bz_size, z_coord_small_out_size, z_coord_small_out_size), dtype=np.complex64)
        
        for bx in prange(bx_size):
            for bz in range(bz_size):
                
                # Combine real/imag and work on local copy
                arr_local = f_real[bx, bz] + 1j * f_imag[bx, bz]
                
                # --- Filtering ---
                for mid_id in range(n_mid):
                    val_sum = 0j
                    z_p_power = 1.0
                    for m in range(M_first_sum):
                        z_p_power *= z_p
                        val_sum += arr_local[mid_id, m] * z_p_power
                    
                    # Forward
                    arr_local[mid_id, 0] = l_val * (arr_local[mid_id, 0] + val_sum)
                    for z in range(1, z_coord_small_out_size):
                        arr_local[mid_id, z] = l_val * arr_local[mid_id, z] + z_p * arr_local[mid_id, z-1]
                        
                    # Backward
                    arr_local[mid_id, -1] = -z_p/(1-z_p) * arr_local[mid_id, -1]
                    for z in range(z_coord_small_out_size-2, -1, -1):
                        arr_local[mid_id, z] = z_p * (arr_local[mid_id, z+1] - arr_local[mid_id, z])
                        
                # --- Backprojection ---
                for x_s in range(z_coord_small_out_size):
                    x_loc = z_coord_small_out[x_s]
                    for z_s in range(z_coord_small_out_size):
                        z_loc = z_coord_small_out[z_s]
                        
                        val_pixel = 0j
                        for mid_id in range(n_mid):
                            angle = angles_mid[mid_id]
                            # Projection coord
                            z_proj = z_loc * math.cos(angle) + x_loc * math.sin(angle)
                            z_norm = (z_proj - z_small_min) / delta_z
                            
                            z_idx = int(math.floor(z_norm))
                            if (z_idx >= 1) and (z_idx < z_coord_small_out_size - 2):
                                # Cubic Spline
                                for k in range(-1, 3):
                                    d = abs(z_norm - (z_idx + k))
                                    if d < 1: w = 2/3 - d**2 * (2-d)/2
                                    elif d < 2: w = (2-d)**3 / 6
                                    else: w = 0.0
                                    val_pixel += w * arr_local[mid_id, z_idx+k]
                                    
                        patch_out[bx, bz, x_s, z_s] = val_pixel * window_val_out[x_s, z_s]
                        
        return patch_out
        
    return patch_backprojection_cpu

# === Reconstruct Image Functions (CPU) ===

def get_reconstruct_image_functions_cpu(x_coord_patch, z_coord_patch, z_coord_small, norm_thresh):
    
    upsampling_factor_x = int(round((x_coord_patch[1] - x_coord_patch[0]) / (z_coord_small[1] - z_coord_small[0])))
    upsampling_factor_z = int(round((z_coord_patch[1] - z_coord_patch[0]) / (z_coord_small[1] - z_coord_small[0])))

    # Output coords
    x_coord_out = np.arange((x_coord_patch.shape[0] - 1) * upsampling_factor_x + z_coord_small.shape[0]) * \
                  (z_coord_small[1] - z_coord_small[0]) + x_coord_patch[0] - \
                  (z_coord_small.shape[0] - 1) / 2 * (z_coord_small[1] - z_coord_small[0])

    z_coord_out = np.arange((z_coord_patch.shape[0] - 1) * upsampling_factor_z + z_coord_small.shape[0]) * \
                  (z_coord_small[1] - z_coord_small[0]) + z_coord_patch[0] - \
                  (z_coord_small.shape[0] - 1) / 2 * (z_coord_small[1] - z_coord_small[0])

    n_z_small_out = z_coord_small.shape[0]
    z_size_patch = z_coord_patch.shape[0]
    x_size_patch = x_coord_patch.shape[0]

    # Reconstruct Image (Scatter) - Serial to avoid race conditions
    @njit
    def reconstruct_image(patches_in, shift_value):
        img_out = np.zeros((x_coord_out.shape[0], z_coord_out.shape[0]), dtype=np.complex64)
        
        for px in range(x_size_patch):
            for pz in range(z_size_patch):
                shift = shift_value[px, pz]
                
                for sx in range(n_z_small_out):
                    for sz in range(n_z_small_out):
                        val = patches_in[px, pz, sx, sz] * shift
                        
                        gx = sx + upsampling_factor_x * px
                        gz = sz + upsampling_factor_z * pz
                        
                        img_out[gx, gz] += val
        return img_out

    # Update Shift
    @njit(parallel=True)
    def update_shift(patches, img_in, shift_value):
        # In place update of shift_value
        for px in prange(x_size_patch):
            for pz in range(z_size_patch):
                val_acc = 0j
                for sx in range(n_z_small_out):
                    for sz in range(n_z_small_out):
                        gx = sx + upsampling_factor_x * px
                        gz = sz + upsampling_factor_z * pz
                        
                        diff = img_in[gx, gz] - patches[px, pz, sx, sz]
                        
                        # Correction: use np.conj()
                        val_acc += diff * np.conj(patches[px, pz, sx, sz])
                
                shift_value[px, pz] += val_acc
                
                # Normalize
                nrm = abs(shift_value[px, pz])
                if nrm > 0:
                    shift_value[px, pz] /= nrm
                else:
                    shift_value[px, pz] = 1.0 # default

    # Get Normalization
    @njit
    def get_normalization(window_val):
        norm_map = np.zeros((x_coord_out.shape[0], z_coord_out.shape[0]), dtype=np.float32)
        for px in range(x_size_patch):
            for pz in range(z_size_patch):
                for sx in range(n_z_small_out):
                    for sz in range(n_z_small_out):
                        gx = sx + upsampling_factor_x * px
                        gz = sz + upsampling_factor_z * pz
                        norm_map[gx, gz] += window_val[sx, sz]**2
        return norm_map

    # Normalize Image
    @njit(parallel=True)
    def normalize_image(img_in, norm_map):
        for i in prange(img_in.shape[0]):
            for j in range(img_in.shape[1]):
                if norm_map[i, j] > norm_thresh:
                    img_in[i, j] /= norm_map[i, j]
                else:
                    img_in[i, j] = 0
        return img_in

    return reconstruct_image, update_shift, get_normalization, normalize_image, x_coord_out, z_coord_out




###############
### DISPLAY ###
###############

# === Functions for the display ===

def bmode_simple(image_complex, img_coord_x, img_coord_z, dynamic_range=60, title=None, out_path=None,dpi=300):
    """
    Simple function to plot a B-mode image

    :param image_complex: complex image we want to plot
    :param img_coord_x: x_coordinates of the grid
    :param img_coord_z: z coordinates of the grid
    :param dynamic_range: dynamic range of the B-mode image

    """

    bmode = 20 * np.log10(np.abs(image_complex))
    bmode = bmode - np.max(bmode)

    kwargs = {'vmin': -dynamic_range, 'vmax': 0, 'cmap':'gray', 'extent': [np.min(img_coord_x), np.max(img_coord_x),
                                                                           np.max(img_coord_z), np.min(img_coord_z)]}

    fig, ax = plt.subplots(sharey=True, sharex=True)
    ax.matshow(bmode.T, **kwargs)
    ax.set_xlabel('x-axis [m]')
    ax.set_ylabel('z-axis [m]')
    ax.xaxis.set_ticks_position('bottom')
    ax.axes.set_facecolor(color='k')
    if title is not None:
        ax.set_title(title)

    plt.show()
