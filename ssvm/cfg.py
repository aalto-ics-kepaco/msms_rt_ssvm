####
#
# The MIT License (MIT)
#
# Copyright 2021 Eric Bach <eric.bach@aalto.fi>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
####

import os

# ------------------
# Caching parameters
# ------------------

# LRU Cache
__lru_cache_max_size = os.getenv("LRU_CACHE_MAX_SIZE", 0)

if __lru_cache_max_size != 0:
    try:
        __lru_cache_max_size = int(__lru_cache_max_size)
    except ValueError:
        if __lru_cache_max_size == "None":
            __lru_cache_max_size = None

assert isinstance(__lru_cache_max_size, int) or __lru_cache_max_size is None

LRU_CACHE_MAX_SIZE = __lru_cache_max_size  # 0 = no caching, None = unlimited caching

# Joblib Memory Cache
JOBLIB_MEMORY_CACHE_LOCATION = os.getenv("JOBLIB_MEMORY_CACHE_LOCATION", None)
