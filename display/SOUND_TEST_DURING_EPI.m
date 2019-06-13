% PURPOSE: TEST SOUND WITH EXAMPLE FUNCTIONAL SCAN
dbstop if error;
%toml_file = '/Volumes/norman/amennen/github/brainiak/rt-cloud/projects/greenEyes/conf/greenEyes_organized.toml';
toml_file = '/Data1/code/rt-cloud/projects/greenEyes/conf/greenEyes_organized.toml';
addpath(genpath('matlab-toml'));
raw_text = fileread(toml_file);
cfg = toml.decode(raw_text);

if strcmp(cfg.machine, 'intel')
    fmri = 1;
else
    fmri = 0;
end
if fmri == 1 % this is just if you're in the scanner room or not but it will always look for triggers
    repo_path ='/Data1/code/rt-cloud/projects/greenEyes/';
else
    repo_path = '/Volumes/norman/amennen/github/brainiak/rt-cloud/projects/greenEyes/';
end
display_path = [repo_path 'display'];

cd(display_path);
wavfilename = [display_path '/stimuli/test_audio.wav'];
%% check audio volume in the scanner
if fmri
    %AUDIO_DEVICENAME = 'HDA Creative: ALC898 Analog (hw:3,0)';
    AUDIO_DEVICENAME = 'HDA Creative: ALC898 Analog';
    AUDIO_devices=PsychPortAudio('GetDevices');
    for dev = 1:length(AUDIO_devices)
        devName = AUDIO_devices(dev).DeviceName;
        %if strcmp(devName,AUDIO_DEVICENAME)
        if length(devName) > length(AUDIO_DEVICENAME)
            if all(AUDIO_DEVICENAME==devName(1:length(AUDIO_DEVICENAME)))
                AUDIODEVICE = AUDIO_devices(dev).DeviceIndex;
            end
        end
    end
end

%%
[y, freq] = audioread(wavfilename);
wavedata = y';
nrchannels = size(wavedata,1); % Number of rows
if ~fmri
    pahandle = PsychPortAudio('Open', [], [], [], freq, nrchannels);
else
    pahandle = PsychPortAudio('Open', AUDIODEVICE, [], [], freq, nrchannels);
end
PsychPortAudio('FillBuffer', pahandle, wavedata);
%%
tStart = GetSecs;
tOn = PsychPortAudio('Start', pahandle, [], tStart,1);
%WaitSecs(25);
tOff = PsychPortAudio('Stop', pahandle,1);
PsychPortAudio('Close', pahandle);3


