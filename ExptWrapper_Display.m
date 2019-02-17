% Purpose: run RealTimeGreenEyesDisplay.m
% This script assigns variables and calls the main script to display.

%% experiment setup parameters
clear all; 

debug = 1;
useButtonBox = 0;
fmri = 0; % whether or not you're in the scanning room
rtData = 0;
%% subject parameters
subjectNum = 500;
subjectName = '0110191_greenEyes';
context = 2;
run = 2;
%% make call to script

RealTimeGreenEyesDisplay(debug, useButtonBox, fmri, rtData, subjectNum, subjectName, context, run)