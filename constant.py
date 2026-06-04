
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


category_list = {
                'mvtec': [
                        "bottle",
                        "cable",
                        "capsule",
                        "carpet",
                        "grid",
                        "hazelnut",
                        "leather",
                        "metal_nut",
                        "pill",
                        "screw",
                        "tile",
                        "toothbrush",
                        "transistor",
                        "wood",
                        "zipper"
                        ],
                
                'visa': [
                        "candle",
                        "capsules",
                        "cashew",
                        "chewinggum",
                        "fryum",
                        "macaroni1",
                        "macaroni2",
                        "pcb1",
                        "pcb2",
                        "pcb3",
                        "pcb4",
                        "pipe_fryum"
                        ],
                
                'btad': [
                        '01',
                        '02',
                        '03'
                ],
                
                'mpdd': [
                        'bracket_black',
                        'bracket_brown',
                        'bracket_white',
                        'connector',
                        'metal_plate',
                        'tubes'
                ],
                
                'mvtec3d_depth': [
                            'bagel', 
                            'cable_gland', 
                            'carrot', 
                            'cookie', 
                            'dowel', 
                            'foam', 
                            'peach', 
                            'potato', 
                            'rope', 
                            'tire'
                            ],
                
                'eyecandies_normals': [
                        "CandyCane",
                        "ChocolateCookie",
                        "ChocolatePraline",
                        "Confetto",
                        "GummyBear",
                        "HazelnutTruffle",
                        "LicoriceSandwich",
                        "Lollipop",
                        "Marshmallow",
                        "PeppermintCandy"
                ],
                
                'brainad': [
                        'type1',
                ],
                
                'retina_oct2017': [
                        'type1',
                ],
                
                'wtb': [
                        'type1',
                ],
                
                'leaves': [
                        'type1',
                ],
}


rgb_modal_list = [
        "mvtec",
        "visa",
        "btad",
        "mpdd"
]

depth_modal_list = [
        "mvtec3d_depth_part1",
        "mvtec3d_depth_part2",
        "eyecandies_normals_part1",
        "eyecandies_normals_part2"
]
