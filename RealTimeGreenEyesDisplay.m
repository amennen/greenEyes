% Display for Green Eyes sample experiment where we just want the subject
% to listen to the story!
% load necessary scanner settings

% wait 5 TR's before starting story?

function RealTimeGreenEyesDisplay(debug, useButtonBox, fmri, rtData, subjectNum, subjectName, context, runNum)

runData.context = context;
runData.subjectNum = subjectNum;
runData.subjectName = subjectName;
runData.run = runNum;

addpath(genpath('stimuli'))
% CONTEXT:
% 0 = NEITHER
% 1 = PARANOID
% 2 = CHEATING
% load in audio data and specify where the project is located
% laptop = /Volumes/norman/amennen/github/greenEyes/
if fmri == 1
    basic_path ='/Data1/code/greenEyes/';
else
    basic_path = '/Volumes/norman/amennen/github/greenEyes/';
end
cd(basic_path);
wavfilename = [basic_path '/stimuli/greenMyeyes_Edited.wav'];
data_path = fullfile(basic_path,'data', ['subject' num2str(runData.subjectNum)]);
runHeader = fullfile(data_path, ['run' num2str(runData.run)]);
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
if (~debug) %so that when debugging you can do other things
    %Screen('Preference', 'SkipSyncTests', 1);
    %   ListenChar(2);  %prevent command window output
    %   HideCursor;     %hide mouse cursor
    
else
    Screen('Preference', 'SkipSyncTests', 2);
end


% display parameters
textColor = 0;
textFont = 'Arial';
textSize = 30;
textSpacing = 25;
fixColor = 0;
backColor = 127;
fixationSize = 4;% pixels
minimumDisplay = 0.25;
KbName('UnifyKeyNames');
LEFT = KbName('1!');
subj_keycode = LEFT;
DEVICENAME = 'Current Designs, Inc. 932';
% set default device to be -1
DEVICE = -1;
if useButtonBox && (~debug)
    [index devName] = GetKeyboardIndices;
    for device = 1:length(index)
        if strcmp(devName(device),DEVICENAME)
            DEVICE = index(device);
        end
    end
else
    % let's set it to look for the Dell keyboard instead
    DEVICENAME = 'Dell KB216 Wired Keyboard';
    [index devName] = GetKeyboardIndices;
    for device = 1:length(index)
        if strcmp(devName(device),DEVICENAME)
            DEVICE = index(device);
        end
    end
    %DEVICE = -1;
end

%TRIGGER = '5%'; % for Penn/rtAttention experiment at Princeton
TRIGGER ='=+'; %put in for Princeton scanner -- default setup
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
    % for PRINCETON
    screenX = 1280;
    screenY = 720;
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
if fmri
    AUDIO_DEVICENAME = 'HDA Creative: ALC898 Analog (hw:3,0)';
    AUDIO_devices=PsychPortAudio('GetDevices');
    for dev = 1:length(AUDIO_devices)
        devName = AUDIO_devices(dev).DeviceName;
        if strcmp(devName,AUDIO_DEVICENAME)
            AUDIODEVICE = AUDIO_devices(dev).DeviceIndex;
        end
    end
end
    % preview task
    % check audio volume
    % if runData.run == 1
    %     nrchannels = 2;
    %     okayVolume=0;
    %     while ~okayVolume
    %         InitializePsychSound(1)
    %         freq=44100;
    %         duration=1;
    %         snddata = MakeBeep(378, duration, freq);
    %         dualdata = [snddata;snddata];
    %         if ~fmri
    %             pahandle = PsychPortAudio('Open', [], [], [], freq, nrchannels);
    %         else
    %             AUDIO_DEVICENAME = 'HDA Creative: ALC898 Analog (hw:3,0)';
    %             AUDIO_devices=PsychPortAudio('GetDevices');
    %             for dev = 1:length(AUDIO_devices)
    %                 devName = AUDIO_devices(dev).DeviceName;
    %                 if strcmp(devName,AUDIO_DEVICENAME)
    %                     AUDIODEVICE = AUDIO_devices(dev).DeviceIndex;
    %                 end
    %             end
    %             %%%%%%
    %             pahandle = PsychPortAudio('Open', AUDIODEVICE, [], [], freq, nrchannels);
    %         end
    %         PsychPortAudio('FillBuffer', pahandle, dualdata);
    %         % start it immediately
    %         PsychPortAudio('UseSchedule',pahandle,1);
    %         PsychPortAudio('AddToSchedule',pahandle,0);
    %         trigger=GetSecs + 2;
    %         begin_time = PsychPortAudio('Start', pahandle, [], trigger);
    %         resp = input('Volume level okay? \n');
    %         if resp == 1
    %             okayVolume = 1;
    %         end
    %         PsychPortAudio('Close', pahandle);
    %
    %     end
    % end
    %Stop playback:
    % Close the audio device:
    
    %% Load in audio data for story
    %
    
    [y, freq] = audioread(wavfilename);
    wavedata = y';
    nrchannels = size(wavedata,1); % Number of rows
    if ~debug
        ListenChar(2);
    end
    %% show them instructions until they press to begin
    continueInstruct = '\n\n-- Please press your INDEX to continue once you understand these instructions. --';
    % show instructions
    Screen(mainWindow,'FillRect',backColor);
    Screen('Flip',mainWindow);
    FlushEvents('keyDown');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % before anything else just brief them on listening to the story, either
    % for the first time or again
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    firstRun = ['Today you will be listening to a pre-recorded audio story. You will get instructions each time you listen to the story.'];
    if runData.run == 1
        % show the first instructions
        firstInstruct = [firstRun continueInstruct];
        DrawFormattedText(mainWindow,firstInstruct,'center','center',textColor,70,[],[],1.2)
        Screen('Flip',mainWindow);
        waitForKeyboard(subj_keycode,DEVICE);
    end
    
    % now tell them they will listen again and ge
    beforeContext = ['Welcome to the task!\n\nPlease read the following instructions carefully, as they may change each time you listen to the story.' continueInstruct];
    DrawFormattedText(mainWindow,beforeContext,'center','center',textColor,70,[],[],1.2)
    Screen('Flip',mainWindow);
    waitForKeyboard(subj_keycode,DEVICE);
    
    [instructCell strOutput] = getContext(runData.context);
    strOutput=[strOutput continueInstruct];
    DrawFormattedText(mainWindow,strOutput,'center','center',textColor,70,[],[],1.2)
    Screen('Flip',mainWindow);
    waitForKeyboard(subj_keycode,DEVICE);
    
    strOutput2 = [instructCell{end-1} '\n\n\n' instructCell{end}];
    DrawFormattedText(mainWindow,strOutput2,'center','center',textColor,70,[],[],1.2)
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
    if ~fmri
        pahandle = PsychPortAudio('Open', [], [], [], freq, nrchannels);
    else
        pahandle = PsychPortAudio('Open', AUDIODEVICE, [], [], freq, nrchannels);
    end
    PsychPortAudio('FillBuffer', pahandle, wavedata);
    if ~debug
        ListenChar(2);
    end
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
    % WaitSecs(10);
    %% save everything
    file_name = ['behavior_run' num2str(runData.run) '_' datestr(now,30) '.mat'];
    save(fullfile(runHeader,file_name),'timing', 'runData');
    
    sca;
    ShowCursor;
    ListenChar;
end