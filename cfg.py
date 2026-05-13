'''
Dictionary with all the application parameters.
'''

params = {

    # Paths
    "model_name": "facebook/VGGT-1B",
    "src_folder": "/home/vision/Desktop/Datasets/Landsat8/unposed_dataset/",
    "dst_folder": "/home/vision/Desktop/Datasets/Landsat8/unposed_dataset/",

    # Flags
    "loop_closure": False,

    # Hardware
    "device": "cuda",

    # VGGT Inference
    "batch_size": 20,
    
}