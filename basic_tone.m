function basic_tone(trigger,duration)

% Perform basic initialization of the sound driver:
InitializePsychSound(1);
freq = 44100;
nrchannels = 1;
snddata = MakeBeep(378, duration, freq);
pahandle = PsychPortAudio('Open', [], [], [], freq, nrchannels);
PsychPortAudio('FillBuffer', pahandle, snddata);

% start it immediately
PsychPortAudio('UseSchedule',pahandle,1); 
PsychPortAudio('AddToSchedule',pahandle,0); 
begin_time = PsychPortAudio('Start', pahandle, [], trigger);

end