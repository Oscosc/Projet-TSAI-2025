import numpy as np
import os
import h5py
import datetime
import math
import matplotlib.pyplot as plt

import scipy.signal
import scipy.interpolate

from numba import cuda

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

# ======================================================================================================================
# SCRIPT
# ======================================================================================================================


# === Set the parameters ====

# define the input and output folders
folder_in = 'example_abdominal_wall_data'

folder_out = 'example_abdominal_wall_data_out'

# parameters of the data preprocessing
angle_downsample_factor = 10  # Downsample factor for the angle
angle_downsample_start = 17  # Starting index for the downsampling
attenuation_tgc = 3e5  # Supposed attenuation for the TGC, in dB per second

# parameters of the beamforming grid
x_size_beam = 1150  # x size of the beamforming grid
z_size_beam = 1300  # z size of the beamforming grid
delta_lambda_fraction = 8  # Spacing for the beamforming grid as a fraction of the center wavelength
min_z_beam = 5e-4  # Minimum z coordinates for the beamforming

# parameters of the apodization during beamforming
tukey_angle = 0.25 # Cosine fraction parameter for for the tukey apodization
max_angle = np.deg2rad(42)  # Max angle for the tukey apodization

# parameter of the windowed Radon transform
angle_rx_max = np.deg2rad(36)  # The maximum Rx angle
angular_resolution_upsampling = 2  # The upsampling factor of Rx with respect to Tx (or opposite id tx_dominates is true)
tx_dominates = False  # If we want the Tx angles to be the largest in term of angular resolution
patch_stride = 24  # Downsampling factor for the patch grid compared to the image grid
window_radius = 2e-3  # Radius of the window for the phase extraction
window_type = 'Tukey'  # Type of the window (must be 'circular' or 'Hann' or 'Tukey')
window_tukey_cosine_fraction = 0.5  # Cosine fraction for the Tukey window if the window type is Tukey

# parameters of the tensor decomposition
reg_param = 1  # Regularization parameter of the inversion
low_rank_n_iter = 20  # Number of iterations for the low rank approximation

# parameters of the postprocessing
norm2_thresh = 1e-1
out_n_iter = 10

# parameters for the GPU
thread_number = 256
x_size_batch = 1


# === Load and preprocess the data ===
data_raw, dict_full = read_linear_transducer_data_standard(folder_in)

t_coord = dict_full['t_coord']
angles_tx_raw = dict_full['angles']
c0 = dict_full['c0']
f0 = dict_full['f0']
pitch = dict_full['pitch']
n_elem = dict_full['n_elem']

x_sensor_list = (np.arange(n_elem) * pitch -
                 (n_elem - 1) / 2 * pitch)

# sort the angles if necessary
sort_angles = np.argsort(angles_tx_raw)
angles_tx_raw = angles_tx_raw[sort_angles]
data_raw = data_raw[sort_angles, :, :]

# downsample if necessary
if angle_downsample_start == 0:
    angles_tx = angles_tx_raw[::angle_downsample_factor]
    data = data_raw[::angle_downsample_factor, :, :]
else:
    angles_tx = angles_tx_raw[angle_downsample_start:-angle_downsample_start:angle_downsample_factor]
    data = data_raw[angle_downsample_start:-angle_downsample_start:angle_downsample_factor, :, :]

# get the complex data
if not np.iscomplexobj(data):
    data = compute_hilbert_fir(data)

# apply the tgc
data = apply_exponential_tgc(data, t_coord, attenuation_tgc)

# determine the validity of the tx angles
if np.std(angles_tx[1:] - angles_tx[:-1]) > 1e-12:
    raise ValueError('Difference between transmit angles not zero')

if np.abs(angles_tx[angles_tx.shape[0]//2]) > 1e-6:
    raise ValueError('Tx angles not centered on 0')

if angles_tx[1] < angles_tx[0]:
    raise ValueError('Tx angles are not in increasing order')

# === Beamforming parameters ===

# determine the grid
lambda0 = c0 / f0
delta_beam = lambda0 / delta_lambda_fraction

x_coord_beam = np.arange(x_size_beam) * delta_beam - (x_size_beam - 1)/2 * delta_beam
z_coord_beam = np.arange(z_size_beam) * delta_beam + min_z_beam

# === Determine the angles ===

delta_tx = angles_tx[1] - angles_tx[0]
if tx_dominates:
    delta_rx = delta_tx * angular_resolution_upsampling
else:
    delta_rx = delta_tx / angular_resolution_upsampling

delta_mid = np.minimum(delta_tx, delta_rx)/2

n_rx = 2 * np.int32(np.round(angle_rx_max / delta_rx)) + 1

angles_rx = np.arange(n_rx) * delta_rx - (n_rx - 1)/2 * delta_rx

angle_mid_max = (angles_tx[-1] + angles_rx[-1])/2

angles_mid = np.arange(2 * np.int32(np.round(angle_mid_max / delta_mid)) + 1) * delta_mid -\
             np.int32(np.round(angle_mid_max / delta_mid)) * delta_mid

print(np.rad2deg(angles_mid))
print(np.rad2deg(angles_rx))

# === Patch definition ===

# define the window
window_radius_pixels = np.int32(np.ceil(window_radius / delta_beam))

z_coord_small = delta_beam * np.arange(-window_radius_pixels + 1, window_radius_pixels)
z_radius = np.sqrt(z_coord_small[:, np.newaxis]**2 + z_coord_small**2)
if window_type == 'circular':
    window_val = np.float64(z_radius < window_radius)
elif window_type == 'Hann':
    window_val = np.cos(np.pi * z_radius / window_radius / 2) ** 2 * np.float64(z_radius < window_radius)
elif window_type == 'Tukey':
    if window_tukey_cosine_fraction == 0:
        window_val = np.float64(z_radius < window_radius)
    else:
        window_val = np.cos(np.pi / 2 *
                            (1 - np.minimum(np.maximum((window_radius - z_radius) / (window_tukey_cosine_fraction * window_radius), 0), 1)))**2
else:
    raise ValueError("Unknown window type")


z_coord_small_out = z_coord_small[:-1] + delta_beam / 2
z_radius_out = np.sqrt(z_coord_small_out[:, np.newaxis]**2 + z_coord_small_out**2)
if window_type == 'circular':
    window_val_out = np.float64(z_radius_out < window_radius)
elif window_type == 'Hann':
    window_val_out = np.cos(np.pi * z_radius_out / window_radius / 2) ** 2 * np.float64(z_radius_out < window_radius)
elif window_type == 'Tukey':
    if window_tukey_cosine_fraction == 0:
        window_val_out = np.float64(z_radius < window_radius)
    else:
        window_val_out = np.cos(np.pi / 2 *
                            (1 - np.minimum(np.maximum((window_radius - z_radius_out) / (window_tukey_cosine_fraction * window_radius), 0), 1)))**2
else:
    raise ValueError("Unknown window type")

# define the patch grid
x_size_patch = (x_size_beam - 2 * window_radius_pixels) // patch_stride + 1
z_size_patch = (z_size_beam - 2 * window_radius_pixels) // patch_stride + 1

x_patch_pos = window_radius_pixels + np.arange(x_size_patch) * patch_stride
z_patch_pos = window_radius_pixels + np.arange(z_size_patch) * patch_stride

x_coord_patch = x_coord_beam[x_patch_pos]
z_coord_patch = z_coord_beam[z_patch_pos]

z_patch_pos_all = (z_patch_pos[:, np.newaxis] * np.ones(x_patch_pos.shape[0])).flatten()
x_patch_pos_all = (x_patch_pos * np.ones(z_patch_pos.shape[0])[:, np.newaxis]).flatten()

# === Get the GPU functions ===

k0 = 2 * np.pi / (c0/f0/2)


func_select_patch = get_select_patch_window_cuda_function(angles_tx, z_coord_small, z_size_patch, x_patch_pos, z_patch_pos)
func_rx_radon = get_patch_radon_transform_rx_cuda_function(angles_tx, z_coord_small, angles_rx, k0, z_size_patch, thread_number=thread_number, max_registers=4)
decomposition_gpu = get_decomposition_function_gpu(angles_tx, angles_rx, angles_mid,
    z_coord_small_out, reg_param, z_size_patch, tx_dominates, low_rank_n_iter, thread_number=thread_number)

patch_backprojection_gpu = get_patch_backprojection_mid_window_cuda_function(
    angles_mid, z_coord_small_out, z_size_patch, thread_number=thread_number)

func_get_image, func_get_shift, func_get_norm, func_norm, block_number_end_1, block_number_end_2, block_number_end_3, \
x_coord_out, z_coord_out =\
    get_reconstruct_image_functions_gpu(
        x_coord_patch, z_coord_patch, z_coord_small_out, norm2_thresh, thread_number)



# allocate the GPU memory

window_val_gpu = cuda.to_device(np.float32(window_val))
window_val_out_gpu = cuda.to_device(np.float32(window_val_out))

mat_filt = np.zeros((z_coord_small.shape[0]-1, z_coord_small.shape[0]), np.complex128)
mat_filt[np.arange(z_coord_small.shape[0]-1), np.arange(z_coord_small.shape[0]-1)] = -1j * (-1/(z_coord_small[1] - z_coord_small[0])) + k0/2
mat_filt[np.arange(z_coord_small.shape[0]-1), np.arange(z_coord_small.shape[0]-1)+1] = -1j * (1/(z_coord_small[1] - z_coord_small[0])) + k0/2
mat_filt = mat_filt * np.exp(-1j * k0 * z_coord_small)

mat_filt_gpu = cuda.to_device(np.complex64(mat_filt))

patch_out_gpu = cuda.device_array((x_size_batch, z_size_patch, angles_tx.shape[0], z_coord_small.shape[0],
                                   z_coord_small.shape[0]), dtype=np.complex64)
win_rad_temp_gpu = cuda.to_device(np.ones((x_size_batch, z_size_patch, angles_tx.shape[0], angles_rx.shape[0],
                                        z_coord_small.shape[0]), dtype=np.complex64))

win_rad_filt_gpu = cuda.to_device(np.ones((x_size_batch, z_size_patch, angles_tx.shape[0], angles_rx.shape[0],
                                        z_coord_small_out.shape[0]), dtype=np.complex64))


f_val_real_gpu = cuda.to_device(np.zeros((x_size_batch, z_size_patch, angles_mid.shape[0], z_coord_small_out.shape[0]),
                                         np.float32))
f_val_imag_gpu = cuda.to_device(np.zeros((x_size_batch, z_size_patch, angles_mid.shape[0], z_coord_small_out.shape[0]),
                                         np.float32))
u_tx_gpu = cuda.to_device(np.ones((x_size_batch, z_size_patch, angles_tx.shape[0]), np.complex64))
u_rx_gpu = cuda.to_device(np.ones((x_size_batch, z_size_patch, angles_rx.shape[0]), np.complex64))


patch_final_gpu = cuda.device_array((x_size_patch, z_size_patch, z_coord_small_out.shape[0],
                                     z_coord_small_out.shape[0]), dtype=np.complex64)


shift_map_gpu = cuda.to_device(np.ones(shape=(x_size_patch, z_size_patch), dtype=np.complex128))
img_real_gpu = cuda.device_array(shape=(x_coord_out.shape[0], z_coord_out.shape[0]), dtype=np.float64)
img_imag_gpu = cuda.device_array(shape=(x_coord_out.shape[0], z_coord_out.shape[0]), dtype=np.float64)
norm_val_gpu = cuda.device_array(shape=(x_coord_out.shape[0], z_coord_out.shape[0]), dtype=np.float64)


# === Beamform the images ===

# beamform
beamformer = get_beamformer_npw_linear_transducer_Tukey_phase_screen(
    angles_tx, x_sensor_list, t_coord, 4, x_coord_beam, z_coord_beam, c0,
    apod_tukey_angle_max=max_angle, apod_tukey_cosine_frac=tukey_angle,
    return_one_per_angle=True)
data_beamform = beamformer(data)

print('Beamforming Done')


img_gpu = cuda.to_device(np.complex64(data_beamform))

n_iter_batch_x = np.int32(np.ceil(x_size_patch / x_size_batch))

# === Perform the method ===

u_tx_all = np.zeros((angles_tx.shape[0], x_coord_patch.shape[0], z_coord_patch.shape[0]), np.complex128)
u_rx_all = np.zeros((angles_rx.shape[0], x_coord_patch.shape[0], z_coord_patch.shape[0]), np.complex128)

loss_in_all = np.zeros((x_coord_patch.shape[0], z_coord_patch.shape[0]))
loss_ref_all = np.zeros((x_coord_patch.shape[0], z_coord_patch.shape[0]))


start_batch_id = 0
for batch_id in range(n_iter_batch_x):
    x_size_batch_local = np.minimum(x_size_patch - start_batch_id, x_size_batch)

    block_number_1 = np.int32(np.ceil(x_size_batch_local * z_size_patch * angles_tx.shape[0] / thread_number))
    block_number_2 = x_size_batch_local * z_size_patch

    func_select_patch[block_number_1, thread_number](patch_out_gpu, img_gpu, window_val_gpu, x_size_batch_local,
                                                     start_batch_id, 0)
    func_rx_radon[block_number_2, thread_number](win_rad_filt_gpu, win_rad_temp_gpu, patch_out_gpu, mat_filt_gpu,
                                                 0, 0)

    decomposition_gpu[block_number_2, thread_number](
        f_val_real_gpu, f_val_imag_gpu, u_tx_gpu, u_rx_gpu, win_rad_filt_gpu)


    f_local = f_val_real_gpu.copy_to_host() + 1j * f_val_imag_gpu.copy_to_host()

    u_tx_local = u_tx_gpu.copy_to_host()
    u_rx_local = u_rx_gpu.copy_to_host()
    win_rad_filt = win_rad_filt_gpu.copy_to_host()

    patch_backprojection_gpu[x_size_batch_local * z_size_patch, thread_number](
        patch_final_gpu, f_val_real_gpu, f_val_imag_gpu, window_val_out_gpu, 0, start_batch_id)

    start_batch_id += x_size_batch

# compute the ratio
ratio = np.sqrt(loss_in_all / loss_ref_all)


for iter_id in range(out_n_iter):
    func_get_image[block_number_end_1, thread_number](patch_final_gpu, img_real_gpu, img_imag_gpu, shift_map_gpu)
    func_get_shift[block_number_end_2, thread_number](patch_final_gpu, img_real_gpu, img_imag_gpu, shift_map_gpu)

func_get_image[block_number_end_1, thread_number](patch_final_gpu, img_real_gpu, img_imag_gpu, shift_map_gpu)

v_corr_all = np.complex128(shift_map_gpu.copy_to_host())

func_get_norm[block_number_end_1, thread_number](window_val_out_gpu, norm_val_gpu)
func_norm[block_number_end_3, thread_number](img_real_gpu, img_imag_gpu, norm_val_gpu)

img_out = np.complex128(img_real_gpu.copy_to_host() + 1j * img_imag_gpu.copy_to_host())

print(img_out.shape)


# === Save the data ===

if not os.path.isdir(folder_out):
    os.mkdir(folder_out)

np.save(os.path.join(folder_out, 'data.npy'), img_out)
np.save(os.path.join(folder_out, 'x_coord.npy'), x_coord_out)
np.save(os.path.join(folder_out, 'z_coord.npy'), z_coord_out)

add_param = {'angles_tx_raw': angles_tx_raw, 'angle_downsample_factor': angle_downsample_factor,
             'angle_downsample_start': angle_downsample_start, 'angles_tx': angles_tx, 'window_radius': window_radius,
             'delta_lambda_fraction': delta_lambda_fraction, 'angle_rx_max': angle_rx_max,
             'angular_resolution_upsampling': angular_resolution_upsampling, 'tx_dominates': tx_dominates,
             't_coord': t_coord, 'c0': c0, 'f0': f0, 'lambda0': lambda0, 'angles_mid': angles_mid,
             'x_size_beam': x_size_beam, 'z_size_beam': z_size_beam, 'reg_param': reg_param,
             'min_z_beam': min_z_beam, 'max_angle': max_angle, 'tukey_angle': tukey_angle,
             'x_coord_beam': x_coord_beam, 'z_coord_beam': z_coord_beam,
             'window_tukey_cosine_fraction': window_tukey_cosine_fraction,
             'out_n_iter': out_n_iter, 'thread_number': thread_number, 'x_size_batch': x_size_batch}

add_param_filename = os.path.join(folder_out, 'additional_parameters.h5')
with h5py.File(add_param_filename, 'w') as file:
    for key, value in add_param.items():
        file.create_dataset(key, data=value)

info_filename = os.path.join(folder_out, 'info.txt')
with open(info_filename, 'w') as file:
    file.write('Date: ' + datetime.datetime.now().isoformat() + '\n')
    file.write('Raw data file: ' + folder_in + '\n')
    file.write('Info: Beamformed image using Tx Rx windowed Radon SVD correction and GPU' + '\n')
    file.write('\n')
    file.write('Window_type: ' + window_type + '\n')
    for key, value in add_param.items():
        file.write(key + ' : ' + str(value) + '\n')



bmode_simple(img_out, x_coord_out, z_coord_out)

