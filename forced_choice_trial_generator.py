# -*- coding: utf-8 -*-
"""
This script contains code for randomizing the lure bin pairs for forced-choice 
versions of the Mnemonic Similarity Task as we did in our previous work 
(Huffman and Stark, 2017 Behavioral Neuroscience). Specifically, lure bins
will be selected and matched for the following test formats:
    
    A-A': forced-choice corresponding lure
    B-C': forced-choice noncorresponding lure
    D-X: traditional forced-choice recognition with unrelated foils

In all cases, the lure bins will be matched between pairs of images, which
will help minimize any influence of difficulty of encoding or retrieval.

During encoding, participants will view images from A, B, C, and D.

During the test phase, participants will view images as paired above. Note, in 
all cases, the correct image will be the XYZa.jpg while the incorrect image 
will be the XYZb.jpg image.

Written by Derek J. Huffman in October 2021 (Copyright Derek J. Huffman 2021)
"""

import numpy as np


def generate_matched_bins_imgs(filename, num_cond=5, img_per_cond=35):
    """
    Generate stimulus sets with matched lure bins.
  
    Parameters
    ----------
    filename : string
        A string with the full path and filename of the lure bins file.
    num_cond : integer
        An integer with the number of conditions. Default=5 (A, B, C, D, X for
        the following test formats: A-A', B-C', D-X).
    img_per_cond : integer
        An integer with the number of images per condition. Default=35 (which
        is the maximum number that we can comfortably run for the 5 conditions,
        for FCC: A-A', FCNC: B-C', and FC: D-X).
    
    Returns
    -------
    img_arr : a 2d numpy.array
        A 2d numpy.array with num_cond rows and img_per_cond columns. The array
        will be ordered such that the lure bins will be stratified from lowest
        to highest with img_per_cond / num_cond columns for each lure bin. 
        Additionally, the "pairings" within the lure bins will be randomized,
        which will allow us to easily pair stimuli across conditions (i.e., 
        by combining rows across the same columns).
    lure_bins : a 1d numpy.array
        A 1d numpy.array with the lure bins information.
    """
    img_bin = np.loadtxt(filename)
    imgs = img_bin[:, 0]
    bins = img_bin[:, 1]
    bins_unique = np.unique(bins)
    imgs_per_bin = int(img_per_cond / len(bins_unique))
    img_arr = np.zeros((num_cond, img_per_cond))
    for c_i, bin_j in enumerate(bins_unique):
        rand_img = np.random.permutation(imgs[bins == bin_j])[:img_per_cond]
        rand_img = rand_img.reshape((num_cond, imgs_per_bin))
        img_arr[:, (c_i * imgs_per_bin):((1 + c_i) * imgs_per_bin)] = rand_img
    lure_bins = np.repeat(bins_unique, imgs_per_bin).astype('int')
    return img_arr, lure_bins


def make_three_digits(x, digits=3):
    """
    Transform numpy.array's to have text format with 3 digits.
    
    For example, this function is helpful for formatting 1 to be 001 so that 
    we can convert from the numbers in the lure bin files to the image names
    (e.g., 001a.jpg).
    
    Parameters
    ----------
    x : a numpy.array
        A numpy.array likely numeric that you would like to convert to have
        digits of values (e.g., 1 becomes 001 if digits=3).
    digits : integer
        An integer indicating the desired number of digits for the number
        string. Default=3.
    
    Returns
    -------
    out : a numpy.array
        A numpy.array of the same shape as x, but in a string format with the
        number of digits as indicated by the parameter digits.
    """
    return np.char.zfill(x.astype('int').astype('str'), 3)


def generate_enc_trials(img_cond_arr, num_cond):
    """
    Generate forced-choice encoding trials.
  
    Parameters
    ----------
    img_cond_arr : a 2d numpy.array
        A 2d numpy.array that is already arranged by lure bins (e.g., the
        output from generate_matched_bins_imgs).
    num_cond : integer
        An integer with the number of conditions.
    
    Returns
    -------
    out : a 1d numpy.array
        A 1d numpy.array with the encoding trials (i.e., image numbers).
    """
    trials = np.random.permutation(np.ravel(img_cond_arr[:(num_cond - 1), :]))
    trials = make_three_digits(trials)
    trials = np.char.add(trials, 'a.jpg')
    return trials


def generate_test_trials(img_cond_arr, bins, img_per_cond=35):
    """
    Generate forced-choice test trials.
  
    Parameters
    ----------
    img_cond_arr : a 2d numpy.array
        A 2d numpy.array that is already arranged by lure bins (e.g., the
        output from generate_matched_bins_imgs).
    bins : a 1d numpy.array
        A 1d numpy.array with the lure bin information.
    img_per_cond : integer
        An integer with the number of images per condition. Default=35 (which
        is the maximum number that we can comfortably run for the 5 conditions,
        for FCC: A-A', FCNC: B-C', and FC: D-X).
    
    Returns
    -------
    out : a 2d numpy.array
        A 2d numpy.array with the test trials. The rows of the array are the
        trials and the first two columns represent the images (with left/right 
        locations implied by the columns of the array; i.e., column 0 is the 
        image to be presented on the left side of the screen and column 1 is 
        the image to be presented on the right side of the screen). Column 2
        contains the trial-type information (i.e., A-A', B-C', D-X) and column
        3 contains the lure bin information (i.e., from 1 to 5).
    """
    a_aprime = np.column_stack((img_cond_arr[0, :], img_cond_arr[0, :]))
    b_cprime = np.column_stack((img_cond_arr[1, :], img_cond_arr[2, :]))
    d_x = np.column_stack((img_cond_arr[3, :], img_cond_arr[4, :]))
    trials = np.row_stack((a_aprime, b_cprime, d_x))
    trials = make_three_digits(trials)
    trials = np.char.add(trials, np.array(('a.jpg', 'b.jpg')))
    # randomize the left/right location of the correct image ------------------
    trials = np.array(list(map(np.random.permutation, trials)))
    # add the trial type and lure bin -----------------------------------------
    trial_type = np.repeat(np.array(("A-A'", "B-C'", "D-X")), img_per_cond)
    lure_bins = np.tile(bins, 3)
    trials = np.column_stack((trials, trial_type, lure_bins))
    return np.random.permutation(trials)


def generate_trials(filename, num_cond=5, img_per_cond=35):
    """
    Generate the encoding and forced-choice test trials for the MST.
  
    Parameters
    ----------
    filename : string
        A string with the full path and filename of the lure bins file.
    num_cond : integer
        An integer with the number of conditions. Default=5 (A, B, C, D, X for
        the following test formats: A-A', B-C', D-X).
    img_per_cond : integer
        An integer with the number of images per condition. Default=35 (which
        is the maximum number that we can comfortably run for the 5 conditions,
        for FCC: A-A', FCNC: B-C', and FC: D-X).
    
    Returns
    -------
    enc_trials : a 1d numpy.array
        A 1d numpy.array with the encoding trials (i.e., image names).
    test_trials : a 2d numpy.array
        A 2d numpy.array with the test trials. The rows of the array are the
        trials and the first two columns represent the images (with left/right 
        locations implied by the columns of the array; i.e., column 0 is the 
        image to be presented on the left side of the screen and column 1 is 
        the image to be presented on the right side of the screen). Column 2
        contains the trial-type information (i.e., A-A', B-C', D-X) and column
        3 contains the lure bin information (i.e., from 1 to 5).
    """
    imgs, bins = generate_matched_bins_imgs(filename, num_cond, img_per_cond)
    enc_trials = generate_enc_trials(imgs, num_cond)
    test_trials = generate_test_trials(imgs, bins, img_per_cond)
    return enc_trials, test_trials
    

# -----------------------------------------------------------------------------
# Test to make sure this is working -------------------------------------------
# -----------------------------------------------------------------------------
enc, ret = generate_trials("Documents/git_repos/MST/Set1 bins.txt")

for cond_i in np.unique(ret[:, 2]):
    print(cond_i)
    curr_trials = ret[ret[:, 2] == cond_i, :]
    bins, counts = np.unique(curr_trials[:, 3], return_counts=True)
    print("Bins: %s"  % bins)
    print("Counts: %s" % counts.astype('str'))
    print()
    
# Looks like we're all good!!! ------------------------------------------------
