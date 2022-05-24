# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Copyright 2022 The Hume AI Authors. All Rights Reserved.
# Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0
# International Licence (CC BY-NC-ND) license.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch
import torch.nn as nn


class MultiTask(nn.Module):
    def __init__(self, feat_dimensions):
        super().__init__()

        self.share_layer = nn.Sequential(
                                    nn.Flatten(),
                                    nn.Linear(feat_dimensions, 128),
                                    nn.LayerNorm(128),
                                    nn.LeakyReLU(),
                                    nn.Linear(128, 64),
                                    nn.LayerNorm(64),
                                    nn.LeakyReLU())
        
        self.emotion_layer = nn.Sequential(
            nn.Linear(64, 32), 
            nn.LeakyReLU(), 
            # nn.Linear(32, 16), 
            # nn.LeakyReLU(), 
            nn.Linear(32, 10)
        )

        self.age_layer = nn.Sequential(
            nn.Linear(64, 32), 
            nn.LeakyReLU(), 
            nn.Linear(32, 16), 
            nn.LeakyReLU(),
            # nn.Linear(16, 8), 
            # nn.LeakyReLU(), 
            # nn.Linear(8, 4),
            # nn.LeakyReLU(),
            nn.Linear(16, 1), 
        )

        self.country_layer = nn.Sequential(
            nn.Linear(64, 32), 
            nn.LeakyReLU(), 
            # nn.Linear(32, 16),
            nn.LeakyReLU(), 
            # nn.Linear(16, 8),
            # nn.LeakyReLU(), 
            nn.Linear(32, 4)
        )
        alpha = 0.33
        beta = 0.33
        gamma = round(1 - alpha - beta, 2)
        self.logsigma = nn.Parameter(torch.FloatTensor([-alpha, -beta, -gamma]))

    def forward(self, x):
        h_shared = self.share_layer(x)
        emotion = torch.sigmoid(self.emotion_layer(h_shared))
        age = self.age_layer(h_shared)
        country = self.country_layer(h_shared)
        return [emotion, country, age], self.logsigma
