# Copyright (c) 2018-present, Ahmed H. Al-Ghidani.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

__author__ = "Ahmed H. Al-Ghidani"
__copyright__ = "Copyright 2018, The nlpeus Project, https://github.com/AhmedHani/nlpeus"
__license__ = "GPL"
__maintainer__ = "Ahmed H. Al-Ghidani"
__email__ = "ahmed.hani.ibrahim@gmail.com"


import functools
from utils.text_utils import Preprocessor


class TextTransformations(object):

    def __init__(self, *args):
        pass 
    
    def __new__(cls, *args):
        return [trasformation for trasformation in args]

    class CharPad(object):
        
        def __init__(self, size):
            pass
        
        def __new__(cls, size):
            return functools.partial(Preprocessor.char_based_pad, size=size)
    
    class CharTruncate(object):

        def __init__(self, size):
            pass

        def __new__(cls, size):
            return functools.partial(Preprocessor.chat_based_truncate, size=size)
    
    class WordPad(object):
        
        def __init__(self, size):
            pass

        def __new__(cls, size):
            return functools.partial(Preprocessor.word_based_pad, size=size)
    
    class WordTruncate(object):

        def __init__(self, size):
            pass

        def __new__(cls, size):
            return functools.partial(Preprocessor.word_based_truncate, size=size)
    
    class Normalize(object):

        def __init__(self):
            pass

        def __new__(cls):
            return Preprocessor.normalize_text

    class ToLowerCase(object):

        def __init__(self):
            pass

        def __new__(cls):
            return Preprocessor.normalize_text

    class SeparatePunctuations(object):

        def __init__(self):
            pass
        
        def __new__(cls):
            return Preprocessor.saparate_punctuations
    
    class ReplaceApostrophes(object):

        def __init__(self):
            pass
        
        def __new__(cls):
            return Preprocessor.replace_apostrophes

    class AddStartEndTokens(object):
        def __init__(self):
            pass

        def __new__(cls):
            return Preprocessor.add_start_end_tokens
