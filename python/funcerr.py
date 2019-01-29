#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:16:05 2019

Function error handling

@author: Artur Tarassow
"""

# define Python user-defined exceptions
class Error(Exception):
   """Base class for other exceptions"""
   pass

class NoSeason(Error):
   """Raised when the input value has annual freqeuncy"""
   pass

class NoBootCi(Error):
    """ Raised when user wants to compute bootsrap CIs
    which are not supported, yet"""
    pass

class UnequalLength(Error):
    """ Raised when series/vectors are of different length"""
    pass

""" EXAMPLE
    try:
        ...
       if i_num < number:
           raise NoSeason
       break
   except NoSeason
       print("This value has no season!")
       print()
   except ValueTooLargeError:
       print("This value is too large, try again!")
       print()l
"""