% Display for Green Eyes sample experiment where we just want the subject
% to listen to the story!
% load necessary scanner settings

% wait 5 TR's before starting story?
clear all;
debug=0;
useButtonBox=0;
CONTEXT=1; 
SUBJECT=500;
RUN=1;
subjectName = '0110191_greenEyes';
rtData = 0;
subjectNum = SUBJECT;
runNum = RUN;


addpath(genpath('stimuli'))
% CONTEXT:
% 0 = NEITHER
% 1 = PARANOID
% 2 = CHEATING
% load in audio data and specify where the project is located
% laptop = /Volumes/norman/amennen/github/greenEyes/
basic_path = '/Volumes/norman/amennen/github/greenEyes/';
cd(basic_path);
wavfilename = [basic_path '/stimuli/greenMyeyes_Edited.wav'];
data_path = fullfile(basic_path,'data', ['subject' num2str(SUBJECT)]);
runHeader = fullfile(data_path, ['run' num2str(RUN)]);
if ~exist(runHeader)
    mkdir(runHeader)
end
%%
%initialize system time calls
seed = sum(100*clock); %get random seed
GetSecs;

% open and set-up output file
dataFile = fopen([runHeader '/behavior.txt'],'a');
fprintf(dataFile,'\n*********************************************\n');
fprintf(dataFile,'* GreenEyes v.1.0\n');
fprintf(dataFile,['* Date/Time: ' datestr(now,0) '\n']);
fprintf(dataFile,['* Seed: ' num2str(seed) '\n']);
fprintf(dataFile,['* Subject Number: ' num2str(subjectNum) '\n']);
fprintf(dataFile,['* Subject Name: ' subjectName '\n']);
fprintf(dataFile,['* Run Number: ' num2str(runNum) '\n']);
fprintf(dataFile,['* Use Button Box: ' num2str(useButtonBox) '\n']);
fprintf(dataFile,['* rtData: ' num2str(rtData) '\n']);
fprintf(dataFile,['* debug: ' num2str(debug) '\n']);
fprintf(dataFile,'*********************************************\n\n');

% print header to command window
fprintf('\n*********************************************\n');
fprintf('* GreenEyes v.1.0\n');
fprintf(['* Date/Time: ' datestr(now,0) '\n']);
fprintf(['* Seed: ' num2str(seed) '\n']);
fprintf(['* Subject Number: ' num2str(subjectNum) '\n']);
fprintf(['* Subject Name: ' subjectName '\n']);
fprintf(['* Run Number: ' num2str(runNum) '\n']);
fprintf(['* Use Button Box: ' num2str(useButtonBox) '\n']);
fprintf(['* rtData: ' num2str(rtData) '\n']);
fprintf(['* debug: ' num2str(debug) '\n']);
fprintf('*********************************************\n\n');

%% Initalizing scanner parameters

disdaqs = 15; % how many seconds to drop at the beginning of the run
TR = 1.5; % seconds per volume
% story duration is 11 minutes 52 seconds
% sotry ends at 11 minutes 36.5 seconds
audioDur = 712; % seconds how long the entire autioclip is
runDur = audioDur;
nTRs_run = ceil(runDur/TR);

% so we want 485 TRs total with the beginning 10 TRs
KbName('UnifyKeyNames');
if (~debug) %so that when debugging you can do other things
    %Screen('Preference', 'SkipSyncTests', 1);
   ListenChar(2);  %prevent command window output
%   HideCursor;     %hide mouse cursor  
   
else
    Screen('Preference', 'SkipSyncTests', 2);
end


% display parameters
textColor = 0;
textFont = 'Arial';
textSize = 35;
textSpacing = 25;
fixColor = 0;
backColor = 127;
fixationSize = 4;% pixels
minimumDisplay = 0.25;
LEFT = KbName('1!');
subj_keycode = LEFT;
DEVICENAME = 'Current Designs, Inc. 932';
if useButtonBox && (~debug)
    [index devName] = GetKeyboardIndices;
    for device = 1:length(index)
        if strcmp(devName(device),DEVICENAME)
            DEVICE = index(device);
        end
    end
else
    DEVICE = -1;
end
TRIGGER = '5%';
%TRIGGER ='=+'; %put in for Princeton scanner
TRIGGER_keycode = KbName(TRIGGER);

%% Initialize Screens
    
screenNumbers = Screen('Screens');

% show full screen if real, otherwise part of screen
if debug
    screenNum = 0;
else
    screenNum = screenNumbers(end);
end

%retrieve the size of the display screen
if debug
    screenX = 1200;
    screenY = 1200;
else
    % first just make the screen tiny
    
    [screenX screenY] = Screen('WindowSize',screenNum);
    % put this back in!!!
    windowSize.degrees = [51 30];
    resolution = Screen('Resolution', screenNum);
    %resolution = Screen('Resolution', 0); % REMOVE THIS AFTERWARDS!!
    %windowSize.pixels = [resolution.width/2 resolution.height];
    %screenX = windowSize.pixels(1);
    %screenY = windowSize.pixels(2);
    % new: setting resolution manually
     screenX = 1920;
     screenY = 1080;
%     %to ensure that the images are standardized (they take up the same degrees of the visual field) for all subjects
%     if (screenX ~= ScreenResX) || (screenY ~= ScreenResY)
%         fprintf('The screen dimensions may be incorrect. For screenNum = %d,screenX = %d (not 1152) and screenY = %d (not 864)',screenNum, screenX, screenY);
%     end
end

%create main window
mainWindow = Screen(screenNum,'OpenWindow',backColor,[0 0 screenX screenY]);
ifi = Screen('GetFlipInterval', mainWindow);
slack  = ifi/2;
% details of main window
centerX = screenX/2; centerY = screenY/2;
Screen(mainWindow,'TextFont',textFont);
Screen(mainWindow,'TextSize',textSize);
fixDotRect = [centerX-fixationSize,centerY-fixationSize,centerX+fixationSize,centerY+fixationSize];
%% check audio volume in the scanner

% preview task
% check audio volume
InitializePsychSound(1)
nrchannels = 2;
okayVolume=0;
while ~okayVolume
    freq=44100;
    duration=1;
    snddata = MakeBeep(378, duration, freq);
    dualdata = [snddata;snddata];
    pahandle = PsychPortAudio('Open', [], [], [], freq, nrchannels);
    PsychPortAudio('FillBuffer', pahandle, dualdata);
    % start it immediately
    PsychPortAudio('UseSchedule',pahandle,1);
    PsychPortAudio('AddToSchedule',pahandle,0);
    trigger=GetSecs + 2;
    begin_time = PsychPortAudio('Start', pahandle, [], trigger);
    resp = input('Volume level okay? \n');
    if resp == 1
        okayVolume = 1;
    end
end
%Stop playback:
PsychPortAudio('Stop', pahandle);
% Close the audio device:
PsychPortAudio('Close', pahandle);

%% Load in audio data for story
%
[y, freq] = audioread(wavfilename);
wavedata = y';
nrchannels = size(wavedata,1); % Number of rows

%% show them instructions until they press to begin

% show instructions
Screen(mainWindow,'FillRect',backColor);
Screen('Flip',mainWindow);
FlushEvents('keyDown');

instructCell = getContext(CONTEXT);

% first give context for the story
for instruct=1:length(instructCell)
    tempBounds = Screen('TextBounds',mainWindow,instructCell{instruct});
    if instruct==length(instructCell)
        textSpacing = textSpacing*1.5;
    end
    Screen('drawtext',mainWindow,instructCell{instruct},centerX-tempBounds(3)/2,centerY-tempBounds(4)/5+textSpacing*(instruct-1),textColor);
    clear tempBounds;
end
Screen('Flip',mainWindow);
waitForKeyboard(subj_keycode,DEVICE);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% now here we're adding to say waiting for scanner, hold tight!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
waitMessage = 'Waiting for scanner start, hold tight!';
tempBounds = Screen('TextBounds', mainWindow, waitMessage);
Screen('drawtext',mainWindow,waitMessage,centerX-tempBounds(3)/2,centerY-tempBounds(4)/2,textColor);
Screen('Flip', mainWindow);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% now here we're going to say to stay still once the triggers start coming
% in
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
STILLREMINDER = ['The scan is now starting.\n\nMoving your head even a little blurs the image, so '...
    'please try to keep your head totally still until the scanning noise stops.\n\n Do it for science!'];
STILLDURATION = 6;

% wait for initial trigger
Priority(MaxPriority(screenNum));
%% Wait for first trigger in the scanner
if (~debug )
        timing.trig.wait = WaitTRPulse(TRIGGER_keycode,DEVICE);
        runStart = timing.trig.wait;
        DrawFormattedText(mainWindow,STILLREMINDER,'center','center',textColor,70)
        startTime = Screen('Flip',mainWindow);
        elapsedTime = 0;
        while (elapsedTime < STILLDURATION)
            pause(0.005)
            elapsedTime = GetSecs()-startTime;
        end
else
    runStart = GetSecs;
end
Screen(mainWindow,'FillRect',backColor);
Screen(mainWindow,'FillOval',fixColor,fixDotRect);
Screen('Flip',mainWindow);
Priority(0);

%%
pahandle = PsychPortAudio('Open', [], [], [], freq, nrchannels);
PsychPortAudio('FillBuffer', pahandle, wavedata);

% calculate onset of story
audioOnset = disdaqs;
volStart = 1 + disdaqs/TR ; % this should be on the 11th trigger
timing.plannedOnsets.audioStart = audioOnset + runStart;

% actual playing
% wait for first trigger
[timing.trig.pulses(volStart) runData.pulses(volStart)] = WaitTRPulse(TRIGGER_keycode,DEVICE,timing.plannedOnsets.audioStart);
timing.actualOnsets.audioStart = PsychPortAudio('Start', pahandle, [], timing.plannedOnsets.audioStart,1);
fprintf('delay is %8.8f\n', timing.plannedOnsets.audioStart-timing.actualOnsets.audioStart)

%% Now record all the triggers from the scanner
% calculate onsets for all subsequent TRs in the scanner
% goal: record every trigger during the story
% music starts at 15
musicDur = 18; % how long the music lasts
silenceDur1 = 3;
storyTRs = 25:475;
nTRs_story = length(storyTRs);
nTRs_music = musicDur/TR;
stationTRs = zeros(nTRs_story,1);

runData.loadCategSep = NaN(nTRs_story,1);
runData.categSep = NaN(nTRs_story,1);
runData.feedbackProp = NaN(nTRs_story,1);

timing.plannedOnsets.story = timing.plannedOnsets.audioStart + musicDur + silenceDur1 + [0 cumsum(repmat(TR, 1,nTRs_story-1))];
timing.actualOnsets.story = NaN(nTRs_story,1);
runData.pulses = NaN(nTRs_run,1);
% prepare for trial sequence
% want displayed: run, volume TR, story TR, tonsset dif, pulse, 
fprintf(dataFile,'run\t\tvolume\t\tstoryTR\t\tonsdif\t\tpulse\t\tstation\t\tload\t\tcatsep\t\tFeedback\n');
fprintf('run\t\tvolume\t\tstoryTR\t\tonsdif\t\tpulse\t\tstation\t\tload\t\tcatsep\t\tFeedback\n');


for iTR = 1:nTRs_story
    volCounter = storyTRs(iTR); % what file number this story TR actually is
    [timing.trig.pulses(volCounter) runData.pulses(volCounter)] = WaitTRPulse(TRIGGER_keycode,DEVICE,timing.plannedOnsets.story(iTR));
    timing.actualOnsets.story(iTR) = timing.trig.pulses(volCounter);
    isStation = stationTRs(iTR);
    % print out TR information
    fprintf(dataFile,'%d\t\t%d\t\t%d\t\t%.3f\t\t%d\t\t%d\t\t%d\t\t%.3f\t\t%.3f\n',runNum,volCounter,iTR,timing.actualOnsets.story(iTR)-timing.plannedOnsets.story(iTR),runData.pulses(volCounter),isStation,runData.loadCategSep(iTR),runData.categSep(iTR),runData.feedbackProp(iTR));
    fprintf('%d\t\t%d\t\t%d\t\t%.3f\t\t%d\t\t%d\t\t%d\t\t%.3f\t\t%.3f\n',runNum,volCounter,iTR,timing.actualOnsets.story(iTR)-timing.plannedOnsets.story(iTR),runData.pulses(volCounter),isStation,runData.loadCategSep(iTR),runData.categSep(iTR),runData.feedbackProp(iTR));

end
%%
% Stop playback:
[timing.PPAstop.startTime timing.PPAstop.endPos timing.PPAstop.xruns timing.PPAstop.estStopTime] = PsychPortAudio('Stop', pahandle,1);
%[startTime endPos xruns estStopTime] = PsychPortAudio('Stop', pahandle,0);
% Close the audio device:
PsychPortAudio('Close', pahandle);

%% save everything 
file_name = ['behavior_run' num2str(RUN) '.mat'];
save(fullfile(data_path,file_name),'timing', 'runData');

sca;
ShowCursor;