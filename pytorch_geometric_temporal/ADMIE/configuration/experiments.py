Layer_name_list = ['conv',
                   'conv_3d',
                   'time_distr_conv',
                   'time_distr_conv_3d',
                   'lstm',
                   'hidden_dense',
                   'dense',
                   'Flatten',
                   'Dropout']
experiments = dict()

experiments['exp_fuzzy1'] = {'row_all': [('dense', {1})],
                             'output': [
                                 ('concatenate', {1}),
                                 ('dense', {1})
                             ],
                             }

# experiments['cnn1'] = {
# 'images': [('conv', [3, 10]),
#            ('Flatten', []),
#            ('dense', {0.25, 0.125}),
#            ('dense', {0.25, 0.125}),
#            ],
#     'nwp': [('conv', 2),
#             ('Flatten', []),
#             ('dense', {256, 128}),
#             ('dense', {64}),
#             ],
#
#     'row_all': [('dense', {4}),
#                 ('dense', {64, }),
#                 ],
#     'row_calendar': [('dense', {64}),
#                      ],
#     'output': [('concatenate', {1}),
#                ('dense', {32})
#                ],
# }
# experiments['cnn2'] = {
#
#     'nwp': [('conv', 2),
#             ('conv', 2),
#             ('Flatten', []),
#             ('dense', {64}),
#             ],
#     'row_obs_calendar': [('dense', {4}),
#                 ('dense', {64, }),
#                 ],
#     'output': [('concatenate', {1}),
#                ('dense', {64})
#                ],
# }
# experiments['cnn3'] = {
#     'images': [('conv', [7, 10]),
#                ('conv', [5, 7]),
#                ('conv', 3),
#                ('Flatten', []),
#                ('dense', {0.12}),
#                ('dense', {64}),
#                ],
#
#     'row_obs_calendar': [('dense', {64}),
#                      ],
#
# 'row': [('dense', {5, 0.5, 1, 2}),
#             ('dense', {64, }),
#             ],

#     'output': [('concatenate', {1}),
#                ('dense', {64})
#                ],
# }

experiments['mlp1'] = {
    # 'row_stats': [('dense', {5, 7, 4, 1024}),
    #             ('dense', {'linear', 64}),
    #             ('dense', {64}),
    #             ],
    'row_all': [('dense', [5, 10]),
                ('dense', {256}),
                ],
    # 'row_calendar': [
    #                  ('dense', {64}),
    #                  ],
    'output': [('concatenate', {1}),
               ('dense', {32})
               ],
}

experiments['mlp2'] = {
    'row_all': [('dense', [2048, 4096]),
                ('dense', {'linear'}),
                ('dense', {512}),
                ],
    # 'row_calendar': [('dense', {5, 100, 20}),
    #              ('dense', {64}),
    #              ],
    'output': [
        ('concatenate', {1}),
        ('dense', {64})
    ],
}

experiments['mlp3'] = {
    'row_all': [('dense', [512, 1024]),
                ('dense', {'linear'}),
                ('dense', {128}),
                ],
    'output': [('concatenate', {1}),
               ('dense', {32})
               ],
}

experiments['mlp4'] = {
    'row_all': [('dense', [1024, 2048]),
                ('dense', {256}),
                ],
    'output': [
        ('concatenate', {1}),
        ('dense', {32})
    ],
}

experiments['mlp5'] = {
    'lstm': [('bi_lstm', [5, 8]),
             ('Flatten', []),
             ('dense', {512}),
             ],
    'output': [
        ('concatenate', {1}),
        ('dense', {128})
    ],
}

experiments['mlp6'] = {
    'lstm': [('bi_lstm', [2, 4]),
             ('Flatten', []),
            ('dense', {256}),
             ],
    'output': [
        ('concatenate', {1}),
        ('dense', {32})
    ],
}

# experiments['distributed_cnn1'] = {
#     'nwp': [('conv', [3, 10]),
#             ('Flatten', []),
#             ('dense', {0.25, 0.125}),
#             ('dense', {0.25, 0.125}),
#             ],
#     # 'row_all': [('dense', {5, 0.5, 1, 2}),
#     #             ('dense', {64, }),
#     #             ],
#     'row_calendar': [('dense', {5, 10, 20}),
#                      ('dense', {64}),
#                      ],
#     'output': [('concatenate', {1}),
#                ('dense', {0.5, 0.25}),
#                ('dense', {0.5, 'linear'}),
#                ('dense', {32, 64, 128})
#                ],
# }

# experiments['distributed_mlp1'] = {
#     'row_nwp': [('dense', {4, 8}),
#                 ('dense', {0.25, 0.5}),
#                 ],
#     # 'row_all': [('dense', {5, 0.5, 1, 2}),
#     #             ('dense', {64, }),
#     #             ],
#     'row_calendar': [('dense', {5, 10, 20}),
#                      ('dense', {64}),
#                      ],
#     'output': [('concatenate', {1}),
#                ('dense', {0.5, 0.25}),
#                ('dense', {0.5, 'linear'}),
#                ('dense', {32, 64, 128})
#                ],
# }
# experiments['distributed_mlp2'] = {
#     'row_nwp': [('dense', {4, 8}),
#                 ('dense', {0.25, 0.5}),
#                 ],
#     # 'row_all': [('dense', {5, 0.5, 1, 2}),
#     #             ('dense', {64, }),
#     #             ],
#     'row_calendar': [('dense', {5, 10, 20}),
#                      ('dense', {64}),
#                      ],
#     'output': [('concatenate', {1}),
#                ('dense', {0.5, 0.25}),
#                ('dense', {0.5, 'linear'}),
#                ('dense', {32, 64, 128})
#                ],
# }
# experiments['mlp_for_combine_data'] = {
#     'row_all': [('dense', {4, 8}),
#                 ('dense', {0.5, 0.75}),
#                 ],
#     'output': [('concatenate', {1}),
#                ('dense', {0.5, 0.75}),
#                ('dense', {32, 64, 128})
#                ],
# }
# experiments['output_combine'] = {
#     'output': [('concatenate', {1}),
#                ('dense', {32})
#                ]
# }
# mlp_for_combine_simple = {
#     'row_all': [('dense', {4, 8}),
#                 ],
#     'output': [('concatenate', {1}),
#                ('dense', {0.5, 0.75}),
#                ('dense', {32, 64, 128})
#                ],
# }
# experiments['distributed_mlp3'] = {
#     'row_all': [('dense', {0.25, 0.125}),
#                 ],
#     'output': [('concatenate', {1}),
#                ('dense', {0.5, 0.25}),
#                ('dense', {0.5, 'linear'}),
#                ('dense', {32, 64, 128})
#                ],
# }

# experiments['lstm1'] = {
#     'lstm': [('lstm', [2, 10]),
#              ('lstm', [2, 5]),
#              ('Flatten', []),
#              ('dense', [3, 7])],
#     'output': [('concatenate', {1}),
#                ('dense', {64, 128})]
# }
# experiments['lstm2'] = {
#     'lstm': [('lstm', [2, 10]),
#              ('Flatten', []),
#              ('dense', [3, 7]),
#              ('Reshape', []),  #: [32, 32]}
#              ('lstm', [2, 10]),
#              ('Flatten', []),
#              ('dense', [3, 7])
#              ],
#     'output': [('concatenate', {1}),
#                ('dense', {64, 128})
#                ]
# }
# experiments['lstm3'] = {
#     'lstm': [('lstm', [2, 10]),
#              ('Flatten', []),
#              ('dense', [3, 7]),
#
#              ],
#     'output': [('concatenate', {1}),
#                ('dense', {12, 64, 128}),
#                ]
# }
# experiments['lstm4'] = {
#     'lstm': [('lstm', {0.5, 1, 2}),
#              ('Flatten', []),
#              ('dense', {1, 0.5}),
#              ('dense', {1, 0.5, 0.25})],
#     'output': [('concatenate', {1}),
#                ('dense', {12, 64, 128})
#                ]
# }
# experiments['transformer'] = {
#     'lstm': [('transformer', 256),
#              ('Flatten', [])],
#     # ('dense', {1, 0.5}),
#     # ('dense', {1, 0.5, 0.25})],
#     'output': [('concatenate', {1}),
#                ('dense', 256),
#                ('dense', 64)
#                ]
# }
# experiments['timm_net'] = {
#     'images': [('vit_net', 1),
#                ('Flatten', []),
#                ('dense', 2048),
#                ('dense', 720),
#                ],
#     'row_calendar': [('dense', 4),
#                          ],
#     'output': [('concatenate', {1}),
#                ('dense', 64),
#                ]
# }
# experiments['timm_net1a'] = {
#     'images': [('vit_net', 1),
#                ('Flatten', []),
#                ('dense', 1024),
#                ('dense', 256),
#                ],
#     'row_calendar': [('dense', 28),
#                      ],
#     'output': [('cross_attention', {1}),
#                ('dense', 64),
#                ]
# }
# experiments['timm_net1b'] = {
#     'images': [('vit_net', 1),
#                ('Flatten', []),
#                ('dense', 2048),
#                ('dense', 720),
#                ],
#     'row_calendar': [('dense', 4),
#                      ],
#     'output': [('concatenate', {1}),
#                ('dense', 64),
#                ]
# }
# experiments['timm_net2a'] = {
#     'images': [('vit_net', 1),
#                ('Flatten', []),
#                ('dense', 1024),
#                ('dense', 256)
#                ],
#     'row_obs_calendar': [('dense', 4),
#                  ('dense', 64)
#                  ],
#     'output': [('cross_attention', {1}),
#                ('dense', 64),
#                ]
# }
# experiments['timm_net2b'] = {
#     'images': [('vit_net', 1),
#                ('Flatten', []),
#                ('dense', 1024),
#                ('dense', 256)
#                ],
#     'row_obs_calendar': [('dense', 4),
#                  ('dense', 64)
#                  ],
#     'output': [('concatenate', {1}),
#                ('dense', 64),
#                ]
# }
# experiments['timm_net3a'] = {
#     'images': [('time_distr_vit_net', 1),
#                ('Flatten', []),
#                ('dense', 2048),
#                ('dense', 720)
#                ],
#     'row_calendar': [('dense', 4),
#                      ],
#     'output': [('concatenate', {1}),
#                ('dense', 64),
#                ]
# }

# experiments['CrossViVit_net'] = {
#     'images': [('vit_net', 1),
#                ('Flatten', []),
#                ('dense', 1024),
#                ('dense', 128)
#                ],
#
#     'nwp': [('conv', {0, 1, 2, 3, 4, 5}),
#             ('conv', {0, 1, 3}),
#             ('Flatten', []),
#             ('dense', {32, 64, 128}),
#             ],
#     'row_calendar': [
#         ('dense', {32, 64, 128}),
#     ],
#     'row_all': [
#         ('dense', {32, 64, 128}),
#     ],
#     'output': [('concatenate', 1),
#                ('dense', 64),
#                ]
# }
# experiments['Time_CrossViVit_net'] = {
#     'images': [('time_distr_vit_net', 1),
#                ('Flatten', []),
#                ('dense', 1024),
#                ('dense', 128)
#                ],
#     'row_calendar': [('dense', 64),
#                      ],
#     'output': [
#         ('concatenate', {1}),
#         ('dense', 64),
#     ]
# }
# experiments['trans_net'] = {
#     'lstm': [('transformer', 128),
#              ('Flatten', []),
#              ('dense', 720),
#              ('dense', 128)
#              ],
#     'output': [
#         ('concatenate', {1}),
#         ('dense', 64),
#     ]
# }
#
# experiments['yolo'] = {
#     'images': [('yolo', 1),
#                ('Flatten', []),
#                ('dense', 1024),
#                ('dense', 256)
#                ],
#     'row_calendar': [('dense', 64),
#                      ],
#     # 'row_obs_nwp': [('dense', 4),
#     #              ('dense', 720),
#     #              ('dense', 128)
#     #              ],
#     'output': [('concatenate', {1}), ('dense', 64),
#                ]
# }
# experiments['unet'] = {
#     'images': [('unet', 1),
#                ('Flatten', []),
#                ('dense', 1024),
#                ('dense', 64)
#                ],
#     'row_calendar': [('dense', 64),
#                      ],
#     # 'row_obs_nwp': [('dense', 4),
#     #              ('dense', 720),
#     #              ('dense', 128)
#     #              ],
#     'output': [('concatenate', {1}), ('dense', 256),
#                ]
# }
#
# experiments['distributed_lstm1'] = {
#     'lstm': [('lstm', 1),
#              ('Flatten', []),
#              ('dense', 0.25),
#              ('dense', 0.5),
#              ],
#     'output': [('concatenate', {1}), ('dense', 0.25),
#                ('dense', 0.5),
#                ('dense', 32)
#                ],
# }
