""" Docstring for src.run_script"""

import numpy as np
import torch
from compute_corrected_image import *

# ======================================================================================================================
# SCRIPT
# ======================================================================================================================


# === def des param ====

def main(folder_in, folder_out, is_gpu):
    '''
    fonction principale qui lance le script
    '''

    # parameters of the data preprocessing
    angle_downsample_factor = 10 # Facteur de sous-échantillonnage angulaire (pour aller plus vite)
    angle_downsample_start = 17  # Angle de départ
    attenuation_tgc = 3e5  # Atténuation pour le TGC [dB/s]

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
    x_size_batch = 1 if is_gpu else 10 # Batch size can be larger on CPU as we loop serially or parallelize differently


    angles_tx, data, t_coord, c0, f0, x_sensor_list, angles_tx_raw = load_preprocess_data(folder_in,
                                angle_downsample_start, angle_downsample_factor, attenuation_tgc)


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

    print("Angles Mid (deg):", np.rad2deg(angles_mid))
    print("Angles Rx (deg):", np.rad2deg(angles_rx))

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

    if is_gpu:

        func_select_patch, func_rx_radon, decomposition_gpu, patch_backprojection_gpu,\
        func_get_image, func_get_shift, func_get_norm, func_norm, block_number_end_1,\
        block_number_end_2, block_number_end_3,x_coord_out, z_coord_out , window_val_gpu,\
        window_val_out_gpu, mat_filt_gpu, patch_out_gpu, win_rad_temp_gpu, win_rad_filt_gpu,\
        f_val_real_gpu, f_val_imag_gpu, u_tx_gpu, u_rx_gpu, patch_final_gpu, shift_map_gpu,\
            img_real_gpu, img_imag_gpu, norm_val_gpu = gpu(c0,
                                                            f0, 
                                                            angles_tx,
                                                            z_coord_small,
                                                            z_size_patch,
                                                            x_patch_pos,
                                                            z_patch_pos,
                                                            angles_rx, 
                                                            angles_mid, 
                                                            tx_dominates, 
                                                            low_rank_n_iter, 
                                                            thread_number, 
                                                            x_coord_patch, 
                                                            z_coord_patch, 
                                                            z_coord_small_out,
                                                            norm2_thresh,
                                                            x_size_patch,
                                                            window_val,
                                                            window_val_out, 
                                                            x_size_batch
                                                        )

    else:
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
        
        # Prepare for Processing
        # Prepare matrix filter for radon
        mat_filt = np.zeros((z_coord_small.shape[0]-1, z_coord_small.shape[0]), np.complex128)
        mat_filt[np.arange(z_coord_small.shape[0]-1), np.arange(z_coord_small.shape[0]-1)] = -1j * (-1/(z_coord_small[1] - z_coord_small[0])) + k0/2
        mat_filt[np.arange(z_coord_small.shape[0]-1), np.arange(z_coord_small.shape[0]-1)+1] = -1j * (1/(z_coord_small[1] - z_coord_small[0])) + k0/2
        mat_filt = mat_filt * np.exp(-1j * k0 * z_coord_small)


    # === Beamform the images ===

    if is_gpu : 

        # beamform
        beamformer = get_beamformer_npw_linear_transducer_Tukey_phase_screen(
            angles_tx, x_sensor_list, t_coord, 4, x_coord_beam, z_coord_beam, c0,
            apod_tukey_angle_max=max_angle, apod_tukey_cosine_frac=tukey_angle,
            return_one_per_angle=True)
        data_beamform = beamformer(data)

        print('Beamforming Done')


        img_gpu = cuda.to_device(np.complex64(data_beamform))

        n_iter_batch_x = np.int32(np.ceil(x_size_patch / x_size_batch))

    else :

        # === Beamforming ===
        print("Starting Beamforming (this may take time)...")
        beamformer = get_beamformer_npw_linear_transducer_Tukey_phase_screen_cpu(
            angles_tx, x_sensor_list, t_coord, 4, x_coord_beam, z_coord_beam, c0,
            apod_tukey_angle_max=max_angle, apod_tukey_cosine_frac=tukey_angle, 
            return_one_per_angle=True)
        
        # Beamform
        data_beamform = beamformer(data)
        print('Beamforming Done')



    if is_gpu :
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
    
    else:
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


    # === Save the data ===

    if not os.path.isdir(folder_out):
        os.mkdir(folder_out)

    np.save(os.path.join(folder_out, 'data.npy'), img_out)
    np.save(os.path.join(folder_out, 'x_coord.npy'), x_coord_out)
    np.save(os.path.join(folder_out, 'z_coord.npy'), z_coord_out)

    # add_param = {'angles_tx_raw': angles_tx_raw, 'angle_downsample_factor': angle_downsample_factor,
    #             'angle_downsample_start': angle_downsample_start, 'angles_tx': angles_tx, 'window_radius': window_radius,
    #             'delta_lambda_fraction': delta_lambda_fraction, 'angle_rx_max': angle_rx_max,
    #             'angular_resolution_upsampling': angular_resolution_upsampling, 'tx_dominates': tx_dominates,
    #             't_coord': t_coord, 'c0': c0, 'f0': f0, 'lambda0': lambda0, 'angles_mid': angles_mid,
    #             'x_size_beam': x_size_beam, 'z_size_beam': z_size_beam, 'reg_param': reg_param,
    #             'min_z_beam': min_z_beam, 'max_angle': max_angle, 'tukey_angle': tukey_angle,
    #             'x_coord_beam': x_coord_beam, 'z_coord_beam': z_coord_beam,
    #             'window_tukey_cosine_fraction': window_tukey_cosine_fraction,
    #             'out_n_iter': out_n_iter, 'thread_number': thread_number, 'x_size_batch': x_size_batch}

    # add_param_filename = os.path.join(folder_out, 'additional_parameters.h5')
    # with h5py.File(add_param_filename, 'w') as file:
    #     for key, value in add_param.items():
    #         file.create_dataset(key, data=value)

    # info_filename = os.path.join(folder_out, 'info.txt')
    # with open(info_filename, 'w') as file:
    #     file.write('Date: ' + datetime.datetime.now().isoformat() + '\n')
    #     file.write('Raw data file: ' + folder_in + '\n')
    #     file.write('Info: Beamformed image using Tx Rx windowed Radon SVD correction and GPU' + '\n')
    #     file.write('\n')
    #     file.write('Window_type: ' + window_type + '\n')
    #     for key, value in add_param.items():
    #         file.write(key + ' : ' + str(value) + '\n')

    print("Done. Displaying image.")
    bmode_simple(img_out, x_coord_out, z_coord_out)




def load_preprocess_data(folder_in, angle_downsample_start, angle_downsample_factor, attenuation_tgc):

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
    
    return angles_tx, data, t_coord, c0, f0, x_sensor_list, angles_tx_raw


def gpu(c0,
        f0, 
        angles_tx,
        z_coord_small,
        z_size_patch,
        x_patch_pos,
        z_patch_pos,
        angles_rx, 
        angles_mid, 
        tx_dominates, 
        low_rank_n_iter, 
        thread_number, 
        x_coord_patch, 
        z_coord_patch, 
        z_coord_small_out,
        norm2_thresh,
        x_size_patch,
        window_val,
        window_val_out, 
        x_size_batch):
    """_summary_
    """

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

    return func_select_patch, func_rx_radon, decomposition_gpu, patch_backprojection_gpu, func_get_image, func_get_shift, func_get_norm, func_norm, block_number_end_1, block_number_end_2, block_number_end_3,x_coord_out, z_coord_out , window_val_gpu, window_val_out_gpu, mat_filt_gpu, patch_out_gpu, win_rad_temp_gpu, win_rad_filt_gpu, f_val_real_gpu, f_val_imag_gpu, u_tx_gpu, u_rx_gpu, patch_final_gpu, shift_map_gpu, img_real_gpu, img_imag_gpu, norm_val_gpu




if __name__ == """__main__""":

    # input and output folders
    FOLDER_IN = '../data/example_abdominal_wall_data'
    FOLDER_OUT = '../data/example_abdominal_wall_data_out'

    main(folder_in=FOLDER_IN, folder_out=FOLDER_OUT, is_gpu=torch.cuda.is_available())
    bmode_simple(img_out, x_coord_out, z_coord_out)
