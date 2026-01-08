import numpy as np
import os
import h5py
import datetime
import math
import matplotlib.pyplot as plt

import scipy.signal
import scipy.interpolate

from numba import njit, prange

# ======================================================================================================================
# FUNCTIONS
# ======================================================================================================================

# === Function for data loading ===

def read_linear_transducer_data_standard(folder_path):
    '''
    Simple function to read data from a linear transducer in standard format.
    '''
    data_path = os.path.join(folder_path, 'data.npy')
    data_raw = np.load(data_path) 

    metadata_path = os.path.join(folder_path, 'metadata.h5')
    dict_out = {}
    with h5py.File(metadata_path, 'r') as metadata_file:
        dict_out['angles'] = np.float64(np.array(metadata_file['angles']))
        dict_out['t_coord'] = np.float64(metadata_file['t_coord'])
        dict_out['c0'] = np.float64(metadata_file['c0'])
        dict_out['f0'] = np.float64(metadata_file['f0'])
        dict_out['pitch'] = np.float64(metadata_file['pitch'])
        dict_out['n_elem'] = np.int32(metadata_file['n_elem'])
        dict_out['elem_width'] = np.float64(metadata_file['elem_width'])
        dict_out['apod_tx'] = np.float64(metadata_file['apod_tx'])
        dict_out['apod_rx'] = np.float64(metadata_file['apod_rx'])

    return data_raw, dict_out

# === Functions for preprocessing ===

def compute_hilbert_fir(data, filter_size=21, dtype=np.complex128):
    beta = 8
    fc = 1
    t = fc / 2 * np.arange((1 - filter_size) / 2, filter_size / 2)
    fir_hilbert = np.sinc(t) * np.exp(1j * np.pi * t)

    sig_shape = data.shape
    sig_len = sig_shape[-1]
    if sig_len < filter_size:
        raise ValueError('Signal length must be larger than filter length')
    if filter_size % 2 == 0:
        raise ValueError('Must be odd')

    fir_hilbert *= scipy.signal.windows.kaiser(filter_size, beta)
    fir_hilbert /= np.sum(fir_hilbert.real)

    sig_ana = np.zeros(shape=sig_shape, dtype=dtype)
    x_it = data.reshape(-1, sig_shape[-1])
    sa_it = sig_ana.reshape(-1, sig_shape[-1])
    for s, sa in zip(x_it, sa_it):
        sa[:] = np.convolve(s, fir_hilbert, 'same')

    return sig_ana

def apply_exponential_tgc(data, t_coord, att_db_s):
    val_acc = 10**(att_db_s * t_coord / 20)
    return data * val_acc

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
                            weight = 1.0 if abs(theta_angle) < max_angle else 0.0 # Fallback
                            
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

                # --- CORRECT LOGIC FOR RADON + FILTER SEQUENCE ---
                # 1. Compute all Radon Projections for this patch
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
                                            if d_z < 1: wz = 2/3 - d_z**2 * (2 - d_z)/2
                                            elif d_z < 2: wz = (2 - d_z)**3 / 6
                                            else: wz = 0.0
                                            
                                            val_acc += wx * wz * patch_local[tx_id, x_idx+i, z_idx+j]
                            radon_temp[tx_id, rx_id, z_out_id] = val_acc

                # 2. Apply Matrix Filter
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

                    # 2. Update u_tx
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
                    
                    # 3. Update u_rx
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

    # 1. Reconstruct Image (Scatter) - Serial to avoid race conditions
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

    # 2. Update Shift
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

    # 3. Get Normalization
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

    # 4. Normalize Image
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

# === Functions for the display ===

def bmode_simple(image_complex, img_coord_x, img_coord_z, dynamic_range=60, title=None):
    bmode = 20 * np.log10(np.abs(image_complex) + 1e-12)
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

if __name__ == '__main__':
    # === Set the parameters ====
    folder_in = 'example_abdominal_wall_data'
    folder_out = 'example_abdominal_wall_data_out'

    angle_downsample_factor = 10
    angle_downsample_start = 17
    attenuation_tgc = 3e5

    x_size_beam = 1150
    z_size_beam = 1300
    delta_lambda_fraction = 8
    min_z_beam = 5e-4

    tukey_angle = 0.25
    max_angle = np.deg2rad(42)

    angle_rx_max = np.deg2rad(36)
    angular_resolution_upsampling = 2
    tx_dominates = False
    patch_stride = 24
    window_radius = 2e-3
    window_type = 'Tukey'
    window_tukey_cosine_fraction = 0.5

    reg_param = 1
    low_rank_n_iter = 20
    norm2_thresh = 1e-1
    out_n_iter = 10
    
    # Batch size can be larger on CPU as we loop serially or parallelize differently
    x_size_batch = 10 

    # === Load and preprocess ===
    print("Loading data...")
    data_raw, dict_full = read_linear_transducer_data_standard(folder_in)

    t_coord = dict_full['t_coord']
    angles_tx_raw = dict_full['angles']
    c0 = dict_full['c0']
    f0 = dict_full['f0']
    pitch = dict_full['pitch']
    n_elem = dict_full['n_elem']

    x_sensor_list = (np.arange(n_elem) * pitch - (n_elem - 1) / 2 * pitch)

    sort_angles = np.argsort(angles_tx_raw)
    angles_tx_raw = angles_tx_raw[sort_angles]
    data_raw = data_raw[sort_angles, :, :]

    if angle_downsample_start == 0:
        angles_tx = angles_tx_raw[::angle_downsample_factor]
        data = data_raw[::angle_downsample_factor, :, :]
    else:
        angles_tx = angles_tx_raw[angle_downsample_start:-angle_downsample_start:angle_downsample_factor]
        data = data_raw[angle_downsample_start:-angle_downsample_start:angle_downsample_factor, :, :]

    if not np.iscomplexobj(data):
        data = compute_hilbert_fir(data)

    data = apply_exponential_tgc(data, t_coord, attenuation_tgc)

    # === Beamforming parameters ===
    lambda0 = c0 / f0
    delta_beam = lambda0 / delta_lambda_fraction
    x_coord_beam = np.arange(x_size_beam) * delta_beam - (x_size_beam - 1)/2 * delta_beam
    z_coord_beam = np.arange(z_size_beam) * delta_beam + min_z_beam

    # === Angles ===
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

    print("Angles Mid (deg):", np.rad2deg(angles_mid))
    print("Angles Rx (deg):", np.rad2deg(angles_rx))

    # === Patch definition ===
    window_radius_pixels = np.int32(np.ceil(window_radius / delta_beam))
    z_coord_small = delta_beam * np.arange(-window_radius_pixels + 1, window_radius_pixels)
    z_radius = np.sqrt(z_coord_small[:, np.newaxis]**2 + z_coord_small**2)
    
    # Create window
    if window_type == 'Tukey':
         window_val = np.cos(np.pi / 2 * (1 - np.minimum(np.maximum((window_radius - z_radius) / (window_tukey_cosine_fraction * window_radius), 0), 1)))**2
    else:
         window_val = np.float64(z_radius < window_radius)

    z_coord_small_out = z_coord_small[:-1] + delta_beam / 2
    z_radius_out = np.sqrt(z_coord_small_out[:, np.newaxis]**2 + z_coord_small_out**2)
    if window_type == 'Tukey':
        window_val_out = np.cos(np.pi / 2 * (1 - np.minimum(np.maximum((window_radius - z_radius_out) / (window_tukey_cosine_fraction * window_radius), 0), 1)))**2
    else:
        window_val_out = np.float64(z_radius_out < window_radius)

    x_size_patch = (x_size_beam - 2 * window_radius_pixels) // patch_stride + 1
    z_size_patch = (z_size_beam - 2 * window_radius_pixels) // patch_stride + 1
    x_patch_pos = window_radius_pixels + np.arange(x_size_patch) * patch_stride
    z_patch_pos = window_radius_pixels + np.arange(z_size_patch) * patch_stride
    x_coord_patch = x_coord_beam[x_patch_pos]
    z_coord_patch = z_coord_beam[z_patch_pos]

    # === Get CPU Functions ===
    print("Compiling CPU functions...")
    k0 = 2 * np.pi / (c0/f0/2)

    func_select_patch = get_select_patch_window_cpu_function(angles_tx, z_coord_small, x_patch_pos, z_patch_pos)
    
    func_rx_radon = get_patch_radon_transform_rx_cpu_function(angles_tx, z_coord_small, angles_rx, k0)
    
    decomposition_cpu = get_decomposition_function_cpu(angles_tx, angles_rx, angles_mid,
        z_coord_small_out, reg_param, tx_dominates, low_rank_n_iter)

    patch_backprojection_cpu = get_patch_backprojection_mid_window_cpu_function(
        angles_mid, z_coord_small_out)

    reconstruct_img_cpu, update_shift_cpu, get_norm_cpu, norm_img_cpu, x_coord_out, z_coord_out = \
        get_reconstruct_image_functions_cpu(x_coord_patch, z_coord_patch, z_coord_small_out, norm2_thresh)

    # === Beamforming ===
    print("Starting Beamforming (this may take time)...")
    beamformer = get_beamformer_npw_linear_transducer_Tukey_phase_screen_cpu(
        angles_tx, x_sensor_list, t_coord, 4, x_coord_beam, z_coord_beam, c0,
        apod_tukey_angle_max=max_angle, apod_tukey_cosine_frac=tukey_angle,
        return_one_per_angle=True)
    
    # Beamform
    data_beamform = beamformer(data)
    print('Beamforming Done')

    # Prepare for Processing
    # Prepare matrix filter for radon
    mat_filt = np.zeros((z_coord_small.shape[0]-1, z_coord_small.shape[0]), np.complex128)
    mat_filt[np.arange(z_coord_small.shape[0]-1), np.arange(z_coord_small.shape[0]-1)] = -1j * (-1/(z_coord_small[1] - z_coord_small[0])) + k0/2
    mat_filt[np.arange(z_coord_small.shape[0]-1), np.arange(z_coord_small.shape[0]-1)+1] = -1j * (1/(z_coord_small[1] - z_coord_small[0])) + k0/2
    mat_filt = mat_filt * np.exp(-1j * k0 * z_coord_small)

    # Storage for final patches
    patch_final_all = np.zeros((x_size_patch, z_size_patch, z_coord_small_out.shape[0], z_coord_small_out.shape[0]), dtype=np.complex64)

    n_iter_batch_x = int(np.ceil(x_size_patch / x_size_batch))
    
    print(f"Starting Processing: {x_size_patch} patches in X direction to process in {n_iter_batch_x} batches.")
    
    start_batch_id = 0
    for batch_id in range(n_iter_batch_x):
        print(f"Processing Batch {batch_id+1}/{n_iter_batch_x}...")
        x_size_batch_local = min(x_size_patch - start_batch_id, x_size_batch)
        
        # 1. Select Patches
        patch_out = func_select_patch(data_beamform, window_val, x_size_batch_local, start_batch_id)
        
        # 2. Radon Transform
        win_rad_filt = func_rx_radon(patch_out, mat_filt)
        
        # 3. Decomposition
        f_real, f_imag = decomposition_cpu(win_rad_filt)
        
        # 4. Backprojection
        patch_reconstructed = patch_backprojection_cpu(f_real, f_imag, window_val_out)
        
        # Store
        patch_final_all[start_batch_id:start_batch_id+x_size_batch_local] = patch_reconstructed
        
        start_batch_id += x_size_batch

    # === Reconstruction loop ===
    print("Reconstructing Image...")
    shift_map = np.ones((x_size_patch, z_size_patch), dtype=np.complex64)
    
    for iter_id in range(out_n_iter):
        img_temp = reconstruct_img_cpu(patch_final_all, shift_map)
        update_shift_cpu(patch_final_all, img_temp, shift_map)
        
    img_final_raw = reconstruct_img_cpu(patch_final_all, shift_map)
    norm_map = get_norm_cpu(window_val_out)
    img_out = norm_img_cpu(img_final_raw, norm_map)

    # Save
    if not os.path.isdir(folder_out):
        os.mkdir(folder_out)
    np.save(os.path.join(folder_out, 'data.npy'), img_out)
    np.save(os.path.join(folder_out, 'x_coord.npy'), x_coord_out)
    np.save(os.path.join(folder_out, 'z_coord.npy'), z_coord_out)
    
    print("Done. Displaying image.")
    bmode_simple(img_out, x_coord_out, z_coord_out)