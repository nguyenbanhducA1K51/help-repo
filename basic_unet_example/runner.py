#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from configs.Config_unet_spleen import get_config
import subprocess

if __name__ == "__main__":
    c = get_config()
    n_epochs = c.n_epochs
    learning_rate = c.learning_rate
    step = 0

    while True:
        result = subprocess.run(['python', 'run_train_pipeline.py',
                                 '--n_epochs', '{}'.format(n_epochs),
                                 '--learning_rate', '{}'.format(learning_rate)])

        if divmod(step, 2)[1] == 0:
            n_epochs = n_epochs + 20
        else:
            learning_rate = learning_rate / 2
        step += 1
