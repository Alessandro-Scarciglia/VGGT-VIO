'''
Dictionary with all the application parameters.
'''

params = {

    # Paths
    "model_name": "facebook/VGGT-1B",
    "src_folder": "/home/vision/Desktop/Datasets/CloudSat_NeRF_Datasets/colour_512_XY_12_0d0_training/",
    "dst_folder": "/home/vision/Desktop/Datasets/CloudSat_NeRF_Datasets/colour_512_XY_12_0d0_training/",

    # Flags
    "do_plot": True,
    "plot_gt": False,
    #"do_eval": False,

    # Hardware
    "device": "cuda",

    # VGGT Inference
    "batch_size": 40,

    # Offset
    "offset": [0, 0, 0],

    # Sampling
    "sampling_deg": 3
    
}