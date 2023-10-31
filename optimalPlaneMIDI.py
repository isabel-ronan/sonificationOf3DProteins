#!/usr/bin/env python
# coding: utf-8

# import libraries
import pretty_midi
import mir_eval
import numpy as np
import warnings
import os
import random
from midiMaker import *
from musicalFeatures import *

def getAllMIDIForProtein(protein = 1, directory = './proteins/', midiDirectoryName = 'newMIDI', numberOfInstruments = 4):
    '''
    this function returns the best plane-traversed MIDI file from pdb data
    @input protein: index number of the protein in the proteins directory 
    @input directory: directory where all proteins for musical analysis exist
    @input midiDirectoryName: directory where all musical MIDI protein files will be stored
    @input numberOfInstruments: number of separate tracks in the MIDI file
    
    @return: file path of best protein
    @rtype: string
    '''
    entries = os.listdir(directory)

    # get current directory and make new folder
    current_dir = os.getcwd()
    path = os.path.join(current_dir, midiDirectoryName)
    try:
        os.makedirs(path, exist_ok = True)
        print("Directory created successfully.")
    except OSError as error:
        print("Directory cannot be created.")
    

    # initialise musical feature arrays
    featureList = []
    arrayOfPaths = []
    # MIDI scale being used
    majorMidiScaleC = [60, 62, 64, 65, 67, 69, 71, 72]
    # all the functions that must be called
    centerOfMass, coordinateArray, normalised_bFactors = loadProtein(directory + entries[protein])
    listOfDistancesInteger, listOfDistancesFloat = getDistancesFromCenter(centerOfMass, coordinateArray)
    labelledDataX, labelledDataY = mappingDistances(majorMidiScaleC, listOfDistancesInteger, listOfDistancesFloat, coordinateArray)
    knn = trainKNN(labelledDataX, labelledDataY)
    for i in range(20):
        xyz, xyzWithDistances, threshold = getPlaneTraversalData(coordinateArray, listOfDistancesFloat)
        planeVariable = (int((len(xyz)) / 20) * (i+1))
        planePoints, arrayOfLengths, midiToMap = proteinPlaneSweep([planeVariable - 2, planeVariable - 1, planeVariable], xyz, threshold)
        try:
            midiOutput = makeMIDI(listOfDistancesFloat, xyzWithDistances, midiToMap, knn, normalised_bFactors, numberOfInstruments, "./" + midiDirectoryName + "/"+ entries[protein].replace(".pdb", "-") + str(i) + ".mid", writeFile = True, moreRhythmic = True)
        except Exception as e:
            print("MIDI file number ", i, " could not be created.")
            print(e)
        # analyse
        test_midi_path ="./" + midiDirectoryName + "/"+ entries[protein].replace(".pdb", "-") + str(i) + ".mid"
        try:
            # test for corrupted MIDI files
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                features = get_features(test_midi_path, normalized = False)
                featureList.append(features)
                arrayOfPaths.append(test_midi_path)
        except:
                print("This protein could not be musically analysed.")
    return featureList, arrayOfPaths

def getBestPlane(proteinUsed = 1, directoryUsed = './proteins/', midiDirectoryNameUsed = 'newMIDI', numberOfInstruments = 4):
    featureList, arrayOfPaths = getAllMIDIForProtein(protein = proteinUsed, directory = directoryUsed, midiDirectoryName = midiDirectoryNameUsed, numberOfInstruments = numberOfInstruments)
    # determine most musical MIDI file
    bestRhythm = 0
    dominantPitch = 0
    durationIndex = 0
    longestDuration = 0
    rhythmIndex = 0
    dominantPitchIndex = 0
    overallScore = 0
    overallIndex = 0

    for i in range(len(featureList)):
        try:
            averageScore = sum(featureList[i]) / len(featureList[i])
            if averageScore > overallScore:
                overallScore = averageScore
                overallIndex = i
            if featureList[i][4] > longestDuration:
                longestDuration = featureList[i][4]
                durationIndex = i
            if featureList[i][3] > bestRhythm:
                bestRhythm = featureList[i][3]
                rhythmIndex = i
            if featureList[i][2] > dominantPitch:
                dominantPitch = featureList[i][2]
                dominantPitchIndex = i
        except:
            print("Features could not be analysed")
    print("Longest Duration: " + str(longestDuration) + "\nDuration Index: " + str(durationIndex))
    print("Best Rhythm: " + str(bestRhythm) + "\nRhythm Index: " + str(rhythmIndex))
    print("Pitch Dominance: " + str(dominantPitch) + "\nPitch Index: " + str(dominantPitchIndex))
    print("Overall Average: " + str(overallScore) + "\nOverall Index: " + str(overallIndex))

    return arrayOfPaths[overallIndex]





