# Project: FlowState Utilities
# Description: Global utilities for all nodes.
# Author: Johnathan Chivington
# Contact: johnathan@flowstateengineering.com | youtube.com/@flowstateeng


##
# OUTSIDE IMPORTS
##
import os, sys, time
import folder_paths


def get_mins_and_secs(start_time):
    loading_duration = time.time() - start_time
    loading_mins = int(loading_duration // 60)
    loading_secs = int(loading_duration - loading_mins * 60)
    return loading_duration, loading_mins, loading_secs


def get_vae_list():
        vaes = folder_paths.get_filename_list('vae')
        approx_vaes = folder_paths.get_filename_list('vae_approx')
        sdxl_taesd_enc = False
        sdxl_taesd_dec = False
        sd1_taesd_enc = False
        sd1_taesd_dec = False
        sd3_taesd_enc = False
        sd3_taesd_dec = False
        f1_taesd_enc = False
        f1_taesd_dec = False

        for v in approx_vaes:
            if v.startswith('taesd_decoder.'):
                sd1_taesd_dec = True
            elif v.startswith('taesd_encoder.'):
                sd1_taesd_enc = True
            elif v.startswith('taesdxl_decoder.'):
                sdxl_taesd_dec = True
            elif v.startswith('taesdxl_encoder.'):
                sdxl_taesd_enc = True
            elif v.startswith('taesd3_decoder.'):
                sd3_taesd_dec = True
            elif v.startswith('taesd3_encoder.'):
                sd3_taesd_enc = True
            elif v.startswith('taef1_encoder.'):
                f1_taesd_dec = True
            elif v.startswith('taef1_decoder.'):
                f1_taesd_enc = True
        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append('taesd')
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append('taesdxl')
        if sd3_taesd_dec and sd3_taesd_enc:
            vaes.append('taesd3')
        if f1_taesd_dec and f1_taesd_enc:
            vaes.append('taef1')
        return vaes